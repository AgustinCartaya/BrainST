"""Training script for BrainST-vol: the covariate-conditioned diffusion
model that predicts a ROI-volume profile (one scalar per brain structure)
from demographic/clinical covariates (age, sex, diagnosis).

This script wires together:
    - `PrepareTrainingDataset` / `MaxPerSubjectSampler`: the same
      subject-capped longitudinal sampling strategy used by
      `training_brainst_img.py`, but yielding ROI-volume vectors instead
      of image latents.
    - `diffusion_loop`: the reverse-diffusion sampler used both for
      quick qualitative checks (`multi_inference_from_noise`) and for
      null-text-inversion-based reconstruction (`compute_reconstruction_loss`).
    - `validation()`: periodically (a) reconstructs held-out real
      ROI-volume profiles via DDIM inversion and measures reconstruction
      MAE, and (b) synthesizes ROI-volume profiles for a fixed grid of
      covariates and compares their distribution (Frechet distance) to
      the closest matching real subjects.
    - `EMA`: exponential moving average of UNet weights, used for more
      stable validation/inference (identical implementation to the one in
      `training_brainst_img.py`).

Run directly (no CLI): the hard-coded `args_train` dictionary at the
bottom of this file configures the run and is intentionally left
untouched by this documentation pass -- edit it directly to change
paths/hyperparameters.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import glob
import json
import logging
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn.functional as F

# monai
from monai.networks.schedulers.ddpm import DDPMPredictionType

# images
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# mine
import configs as cfg
import src.utils.functions as fc
from src.brainst_vol import instantiate_models, null_inversion
from src.brainst_vol.networks_declaration.rectified_flow import RFlowScheduler
from src.utils import data_normalization, load_dataset

logger = logging.getLogger(__name__)

device_name = cfg.DEVICE
device = torch.device(device_name)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs (NumPy, PyTorch CPU/GPU) for reproducibility.

    Also forces cuDNN into deterministic mode. Note this does not seed
    Python's built-in ``random`` module (used elsewhere via dedicated
    calls / ``torch.Generator`` objects instead).

    Args:
        seed: Seed value applied to all RNGs.

    Side Effects:
        Mutates global RNG state for NumPy and PyTorch, and sets
        ``torch.backends.cudnn.deterministic = True`` /
        ``torch.backends.cudnn.benchmark = False``.
    """
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista


class LoadPaths:
    """Loads the training dataset CSV and assembles per-instance training dicts.

    Simpler counterpart to ``training_brainst_img.py``'s ``LoadPaths``:
    no latent/attention-mask handling, since BrainST-vol trains directly
    on ROI-volume scalars and covariate columns already present in the
    dataset CSV.
    """

    def __init__(self, training_dataset_path_name: str, conditions_keys_ordered: list[str], dataset_filters: dict | None = None):
        """Load the dataset and latents from the specified paths.

        Args:
          training_dataset_path_name: Path to the training dataset.
          conditions_keys_ordered: List of condition keys in the desired order.
          dataset_filters: Optional filters to apply to the dataset in the form of a dictionary where keys are column names and values are lists of values to filter by.
        """
        self.complete_dataset = load_dataset.LoadDataset(training_dataset_path_name, sid_column=cfg.COL_SUBJECT_ID, session_column=cfg.COL_SESSION_ID, age_column=cfg.COL_AGE) 

        self.conditions_keys_ordered = conditions_keys_ordered

        if dataset_filters is not None:
            for column, values_list in dataset_filters.items():
                self.complete_dataset.df = self.complete_dataset.df[self.complete_dataset.df[column].isin(values_list)]


    def get_train_data(self, only_baseline: bool = False) -> list[dict]:
        """Assemble the list of training instances (one dict per subject/session row).

        Args:
            only_baseline: If True, use only each subject's baseline
                (first) session instead of every available session.

        Returns:
            A list of instance dicts, each with keys ``"id"``, ``"age"``,
            ``"sex"``, ``"dx"``, and one entry per key in
            ``self.conditions_keys_ordered``.

        Raises:
            ValueError: If a requested condition key is missing from a
                dataset row.
        """
        if only_baseline:
            complete_df = self.complete_dataset.get_baseline_df()
        else:
            complete_df = self.complete_dataset.df.copy()

        # obtain train 
        complete_df = complete_df[complete_df[cfg.COL_SPLIT] == "train"]

        instances = []  
        for i, row in complete_df.iterrows():
            _instance = {}
            _instance["id"] = row[cfg.COL_SUBJECT_ID]
            _instance["age"] = row[cfg.COL_AGE]
            _instance["sex"] = row[cfg.COL_SEX]
            _instance["dx"] = row[cfg.COL_DX]

            for key in self.conditions_keys_ordered:
                if key in row:
                    _instance[key] = row[key]
                else:
                    raise ValueError(f"Condition key {key} not found in the dataset.")
            instances.append(_instance)
        return instances




class PrepareTrainingDataset(Dataset):
    """PyTorch ``Dataset`` yielding per-subject covariates and ROI-volume targets."""

    def __init__(self, 
                 training_dataset_path_name: str,
                 conditions_keys_ordered: list[str],
                 dataset_filters: dict | None = None,

                 ):
        """Load training instances via :class:`LoadPaths` and store dataset config.

        Args:
            training_dataset_path_name: Path to the training CSV.
            conditions_keys_ordered: Ordered list of ROI-volume column names to predict.
            dataset_filters: Optional column-based row filters (see :class:`LoadPaths`).
        """

        # load data
        data_loader = LoadPaths(training_dataset_path_name, conditions_keys_ordered,
                                  dataset_filters=dataset_filters)
        
        self.train_data = data_loader.get_train_data(only_baseline=False)
        self.conditions_keys_ordered = conditions_keys_ordered

        logger.info(f"Number of training images: {len(self.train_data)}")

        # number of latent in the folder
        self.num_instances = len(self.train_data) 
        self._length = self.num_instances

    def __len__(self) -> int:
        """Number of training instances."""
        return self._length

    def __getitem__(self, index: int) -> dict:
        """Fetch one training example: id, covariates, and ROI-volume targets.

        Args:
            index: Index into the dataset (wrapped modulo ``self.num_instances``).

        Returns:
            A dict with keys ``"id"``, ``"age"`` (1-element tensor),
            ``"sex"`` (1-element tensor), ``"dx"`` (one-hot tensor, 3
            classes), and one entry per condition key (each a 1-element
            tensor).
        """
        # dictionary to store the image and the prompt
        example = {}
        # select latent path name from the list
        instance = self.train_data[index % self.num_instances]


        # obtain the age of the image
        example["id"] = instance["id"]
        example["age"] = torch.tensor([instance["age"]])
        example["sex"] = torch.tensor([instance["sex"]])
        example["dx"] = F.one_hot(torch.tensor(instance["dx"]), num_classes=3)

        # obtain the conditions
        for key in self.conditions_keys_ordered:
            example[key] = torch.tensor([instance[key]])

        return example
    
def collate_fn(examples: list[dict], conditions_keys_ordered: list[str]) -> dict:
    """Batch a list of dataset examples into stacked tensors.

    Args:
        examples: List of per-example dicts as produced by
            :meth:`PrepareTrainingDataset.__getitem__`.
        conditions_keys_ordered: Ordered list of ROI-volume condition keys
            expected in every example.

    Returns:
        A dict with ``"id"`` (list), ``"age"``, ``"sex"``, ``"dx"``
        (stacked tensors), and one stacked tensor per condition key.

    Raises:
        ValueError: If a condition key is missing from the first example.
    """
    res_dict = {}

    res_dict["id"] = [example["id"] for example in examples]
    res_dict["age"] = torch.stack([example["age"] for example in examples])
    res_dict["sex"] = torch.stack([example["sex"] for example in examples])
    res_dict["dx"] = torch.stack([example["dx"] for example in examples])

    for key in conditions_keys_ordered:
        if key in examples[0]:
            # res_dict[key] = torch.tensor([example[key] for example in examples], dtype=torch.float32)
            res_dict[key] = torch.stack([example[key] for example in examples])
        else:
            # If the key is not found, raise an error
            raise ValueError(f"Condition key {key} not found in the examples.")
    return res_dict




class MaxPerSubjectSampler(Sampler):
    """Epoch sampler that caps how many timepoints per subject are drawn each epoch.

    Identical strategy to the one used in ``training_brainst_img.py``:
    prevents subjects with many longitudinal follow-ups from dominating
    training relative to subjects with few sessions.
    """

    def __init__(self, dataset: PrepareTrainingDataset, max_per_subject: int = 3, shuffle: bool = True, generator: torch.Generator | None = None):
        """
        dataset: instancia de PrepareTrainingDataset (con atributo train_data que incluye 'id')
        max_per_subject: número máximo de muestras por sujeto por época

        Args:
            dataset: Dataset whose ``train_data`` list is grouped by subject id.
            max_per_subject: Maximum number of samples drawn per subject per epoch.
            shuffle: Whether to shuffle indices within each subject and across the final epoch order.
            generator: Optional ``torch.Generator`` for reproducible shuffling.
        """
        self.dataset = dataset
        self.max_per_subject = max_per_subject
        self.shuffle = shuffle

        # agrupar índices por sujeto
        self.indices_by_subject = {}
        for idx, instance in enumerate(dataset.train_data):
            sid = instance["id"]
            self.indices_by_subject.setdefault(sid, []).append(idx)
        self.subjects = list(self.indices_by_subject.keys())
        self.generator = generator

    def __iter__(self):
        """Yield a shuffled epoch's worth of dataset indices, capped per subject.

        Returns:
            An iterator over dataset indices for one epoch.
        """
        epoch_indices = []

        for sid in self.subjects:
            indices = self.indices_by_subject[sid]
            if self.shuffle:
                if self.generator is not None:
                    # __permuted_indices = torch.randperm(len(indices), generator=self.generator).tolist()
                    # indices = [indices[i] for i in __permuted_indices]
                    indices = [indices[i] for i in torch.randperm(len(indices), generator=self.generator)]
                else:
                    random.shuffle(indices)
            # tomar hasta max_per_subject
            chosen = indices[:self.max_per_subject]
            epoch_indices.extend(chosen)
        if self.shuffle:
            if self.generator is not None:
                epoch_indices = [epoch_indices[i] for i in torch.randperm(len(epoch_indices), generator=self.generator)]
            else:
                random.shuffle(epoch_indices)
        return iter(epoch_indices)

    def __len__(self) -> int:
        """Total number of samples this sampler yields per epoch."""
        # return len(self.subjects) * self.max_per_subject
        return sum(min(len(self.indices_by_subject[sid]), self.max_per_subject) for sid in self.subjects)



def instantiate_dataset(training_dataset_path_name: str, conditions_keys_ordered: list[str], 
                        batch_size: int, gen_dataloader: torch.Generator, 
                        dataset_filters: dict | None = None, max_timepoints_per_epoch: int = 3) -> torch.utils.data.DataLoader:
    """Build the training ``DataLoader`` with subject-capped sampling.

    Args:
        training_dataset_path_name: Path to the training CSV.
        conditions_keys_ordered: Ordered list of ROI-volume condition column names.
        batch_size: Batch size for the DataLoader.
        gen_dataloader: RNG used by :class:`MaxPerSubjectSampler` for reproducible shuffling.
        dataset_filters: Optional column-based row filters.
        max_timepoints_per_epoch: Passed to :class:`MaxPerSubjectSampler` as ``max_per_subject``.

    Returns:
        A configured ``torch.utils.data.DataLoader`` (8 workers, persistent workers enabled).
    """
    # ---- Data set creation
    train_dataset = PrepareTrainingDataset(
        training_dataset_path_name=training_dataset_path_name,
        conditions_keys_ordered=conditions_keys_ordered,
        dataset_filters=dataset_filters,
    )

    sampler = MaxPerSubjectSampler(train_dataset, max_per_subject=max_timepoints_per_epoch, shuffle=True, generator=gen_dataloader)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda examples: collate_fn(examples, conditions_keys_ordered),
        num_workers=8, 
        persistent_workers=True,
    )
    return train_dataloader








from scipy.linalg import sqrtm
from sklearn.utils import resample


def frechet_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Frechet distance between two multivariate Gaussians fit to ``X`` and ``Y``.

    Standard Frechet Inception Distance (FID) formula applied to
    ROI-volume vectors instead of image-embedding features: fits a
    Gaussian (mean + covariance) to each sample set and computes the
    closed-form 2-Wasserstein distance between them.

    Args:
        X: Real samples, shape ``(n_samples_x, n_features)``.
        Y: Generated samples, shape ``(n_samples_y, n_features)``.
        eps: Small value added to the diagonal of each covariance matrix
            for numerical stability (avoids singular matrices).

    Returns:
        The (squared) Frechet distance between the two fitted Gaussians.
    """
    mu_x, mu_y = X.mean(0), Y.mean(0)
    cov_x, cov_y = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    cov_x += np.eye(cov_x.shape[0])*eps
    cov_y += np.eye(cov_y.shape[0])*eps
    covmean = sqrtm(cov_x.dot(cov_y))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_x - mu_y
    return diff.dot(diff) + np.trace(cov_x + cov_y - 2*covmean)


def evaluate_groups(real_dict: dict, gen_dict: dict, min_n: int = 50, n_bootstrap: int = 500, random_state: int = 0) -> tuple[dict, dict]:
    """Compute per-group and aggregate Frechet-distance metrics with bootstrap confidence intervals.

    Args:
        real_dict: Mapping ``group_key -> np.ndarray`` of real ROI-volume
            samples for that group (``group_key`` is typically a
            ``(age, sex, dx)`` tuple).
        gen_dict: Mapping ``group_key -> np.ndarray`` of generated
            ROI-volume samples for the same groups.
        min_n: Minimum number of samples (in both real and generated sets)
            required to evaluate a group; groups below this are skipped
            (marked with ``"skip": True`` in the per-group results).
        n_bootstrap: Number of bootstrap resamples used to estimate the
            95% confidence interval of the Frechet distance per group.
        random_state: Seed for the bootstrap RNG.

    Returns:
        A tuple ``(results, agg)`` where:
            - ``results`` maps each ``group_key`` to a dict with
              ``n_real``, ``n_gen``, ``frechet``, ``frechet_ci`` (or just
              ``{"skip": True, "n_real", "n_gen"}`` for skipped groups).
            - ``agg`` is a dict of dataset-level aggregate statistics:
              ``groups_evaluated``, ``total_n``, ``frechet_weighted_mean``
              (weighted by ``n_real``), ``frechet_mean``, ``frechet_median``,
              ``frechet_std``, ``worst_group_frechet``.
    """
    rng = np.random.default_rng(random_state)
    results = {}
    for g, X_real in real_dict.items():
        X_gen = gen_dict.get(g)
        if X_gen is None:
            continue
        n_real = len(X_real)
        n_gen = len(X_gen)
        if n_real < min_n or n_gen < min_n:
            results[g] = {"skip": True, "n_real": n_real, "n_gen": n_gen}
            continue
        # compute metrics on full samples
        F_metric = frechet_distance(X_real, X_gen)

        # bootstrap CIs: resample with replacement within group (paired by class)
        F_bs = []
        # M_bs = []
        # A_bs = []
        for _ in range(n_bootstrap):
            xr = resample(X_real, replace=True, n_samples=n_real, random_state=rng.integers(0,2**31-1))
            xg = resample(X_gen, replace=True, n_samples=n_gen, random_state=rng.integers(0,2**31-1))
            F_bs.append(frechet_distance(xr, xg))

        results[g] = {
            "n_real": n_real,
            "n_gen": n_gen,
            "frechet": F_metric,
            "frechet_ci": (np.percentile(F_bs,2.5), np.percentile(F_bs,97.5)),
        }

    # aggregate across groups (weighted by n_real)
    valid = [g for g,v in results.items() if not v.get("skip", False)]
    ns = np.array([results[g]["n_real"] for g in valid], dtype=float)
    # metrics arrays
    F_vals = np.array([results[g]["frechet"] for g in valid])

    total_n = ns.sum()
    weighted = lambda arr: float(np.sum(arr * ns) / total_n) if total_n>0 else float(np.mean(arr))
    agg = {
        "groups_evaluated": len(valid),
        "total_n": int(total_n),
        "frechet_weighted_mean": weighted(F_vals),
        "frechet_mean": float(np.mean(F_vals)),
        "frechet_median": float(np.median(F_vals)),
        "frechet_std": float(np.std(F_vals, ddof=1)),
        "worst_group_frechet": valid[np.argmax(F_vals)] if len(valid)>0 else None,
    }

    return results, agg




@torch.no_grad()
def diffusion_loop(
                    initial_noise: torch.Tensor, 
                    unet: torch.nn.Module, 
                   conditions_model: torch.nn.Module,
                   noise_scheduler, 
                    covars: torch.Tensor, 
                    uncond_embeddings: list[torch.Tensor] | None = None,
                   free_guidance_ratio: float = 3.0,
                   return_noisy_steps: bool = False) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """Run the reverse-diffusion sampling loop over ROI-volume vectors.

    Supports two modes of classifier-free guidance:
        - ``uncond_embeddings is None``: uses a zero embedding as the
          unconditional branch at every step (standard CFG).
        - ``uncond_embeddings`` provided (a list, one per timestep): uses
          these (typically null-text-optimized) embeddings instead,
          enabling accurate reconstruction from inverted latents.

    Args:
        initial_noise: Starting point of the reverse diffusion, shape
            ``(batch, num_conditions)``.
        unet: Diffusion model predicting noise/sample/velocity.
        conditions_model: Model embedding the covariates into a context vector.
        noise_scheduler: Noise scheduler (DDIM-style) already configured
            with inference timesteps.
        covars: Covariates tensor to condition on, shape ``(batch, covar_dim)``.
        uncond_embeddings: Optional per-timestep unconditional embeddings
            (see above).
        free_guidance_ratio: Classifier-free-guidance strength.
        return_noisy_steps: If True, also return the intermediate
            (noisy) volume vectors at every denoising step.

    Returns:
        The final denoised ROI-volume vector(s) as a numpy array, shape
        ``(batch, num_conditions)``. If ``return_noisy_steps=True``,
        returns ``(volumes, denoising_steps)`` where ``denoising_steps``
        is a list of intermediate numpy arrays, one per timestep.
    """
    device = next(unet.parameters()).device
    initial_noise = initial_noise.to(device)
    covars = covars.to(device)

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    conditioning_emb = conditions_model(covars)

    if uncond_embeddings is None:
        uncond_embeddings_ = torch.zeros_like(conditioning_emb)
    else:
        uncond_embeddings_ = None

    volumens = initial_noise
    if return_noisy_steps:
        denoising_steps = []
        
    def denoising_step(x, model, t, context=None, next_t=None, fgr=1.0):
        """Perform a single reverse-diffusion step, applying classifier-free guidance if configured."""
        # free guidance setup
        using_free_guidance = False
        batch_size = x.shape[0]
        
        if context.shape[0] == x.shape[0] * 2:
            using_free_guidance = True
            x = torch.cat([x] * 2)
            
        timesteps = torch.full((x.shape[0],), fill_value=t, dtype=all_timesteps.dtype, device=device)
        noise_pred = model(x=x,timesteps=timesteps,context=context)
        
        if using_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + fgr * (noise_pred_cond - noise_pred_uncond)
            # noise_pred = noise_pred_cond + free_guidance_ratio * (noise_pred_uncond - noise_pred_cond) # mine
            x = x[:batch_size]

        # if not isinstance(noise_scheduler, RFlowScheduler):
        #     x, _ = noise_scheduler.step(noise_pred, t, x)
        # else:
        #     x, _ = noise_scheduler.step(noise_pred, t, x, next_t)  # type: ignore
            x, _ = noise_scheduler.step(noise_pred, t, x)
        
        return x
    
    # for i, (t, next_t) in enumerate(progress_bar):
    for i, (t, next_t) in enumerate(zip(all_timesteps, all_next_timesteps)):

        if uncond_embeddings_ is None: # means thath we have optimized the uncond embeddings
            context = torch.cat([uncond_embeddings[i].expand(*conditioning_emb.shape).to(device), conditioning_emb], dim=0)
        else: # means that we are reconstructing the image only from the noised latents
            context = torch.cat([uncond_embeddings_, conditioning_emb], dim=0)
        
        volumens = denoising_step(volumens, unet, t, context=context, next_t=next_t, fgr=free_guidance_ratio)

        if return_noisy_steps:
            denoising_steps.append(volumens.cpu().numpy())
    volumens = volumens.cpu().numpy()
    if return_noisy_steps:
        return volumens, denoising_steps
    return volumens



def create_input_and_mask(nb_tissues: int, age: float, sex: int, dx: int, normalizer, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a random initial-noise vector and a covariates tensor for a single condition set.

    Args:
        nb_tissues: Number of ROI-volume conditions (dimensionality of the noise vector).
        age: Raw (unstandardized) age; standardized internally via ``normalizer``.
        sex: Integer-coded sex.
        dx: Integer-coded diagnosis (0, 1, or 2).
        normalizer: Fitted normalizer used to standardize ``age``.
        seed: Random seed for the initial noise sample.

    Returns:
        ``(rand_noise, covars)``: ``rand_noise`` has shape
        ``(1, nb_tissues)``; ``covars`` has shape ``(1, 1 + 1 + 3)``
        (age, sex, one-hot dx).
    """
    age_norm = normalizer.transform_single(age, "age")
    covars_list = [torch.tensor([[age_norm]]), torch.tensor([[sex]]), F.one_hot(torch.tensor([dx]), num_classes=3)]
    covars = torch.cat(covars_list, dim=1).float()

    gen_randn = torch.Generator().manual_seed(seed) 
    rand_noise = torch.randn((1, nb_tissues), generator=gen_randn)
    return rand_noise, covars


def multi_inference_from_noise(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, nb_tissues: int, normalizer, age: float = 55, sex: int = 1, dx: int = 2, num_tests: int = 5, fgr: float = 3.0, verbose: bool = True, seed: int = 42) -> list[np.ndarray]:
    """Generate multiple independent ROI-volume samples for a fixed covariate set.

    Each of the ``num_tests`` samples uses a different (deterministically
    incremented) seed for its initial noise.

    Args:
        unet: Diffusion model.
        conditions_model: Covariates embedding model.
        noise_scheduler: Configured noise scheduler.
        nb_tissues: Number of ROI-volume conditions.
        normalizer: Fitted normalizer for the age covariate.
        age: Raw age to condition on.
        sex: Integer-coded sex to condition on.
        dx: Integer-coded diagnosis to condition on.
        num_tests: Number of independent samples to generate.
        fgr: Classifier-free-guidance ratio.
        verbose: If True, show a progress bar.
        seed: Base seed; incremented by 1 for each successive sample.

    Returns:
        A list of ``num_tests`` generated ROI-volume arrays.
    """
    x_rec_norm_list = []
    if verbose:
        bar = tqdm(total=num_tests) 

    for i in range(num_tests):
        # ====== Inferencia (imputación) ======
        seed = seed + i  # different seed for each test
        rand_noise, covars = create_input_and_mask(nb_tissues=nb_tissues, age=age, sex=sex, dx=dx, normalizer=normalizer, seed=seed)
        x_rec_norm = diffusion_loop(rand_noise, unet, conditions_model, noise_scheduler, covars, free_guidance_ratio=fgr)
        # x_rec_norm = diffusion_loop_with_stochastic_refinement(rand_noise, model["unet"],model["conditions_model"], model["noise_scheduler"], covars, free_guidance_ratio=fgr, num_trials=5, noise_scale=.0001)

        x_rec_norm = x_rec_norm.cpu().numpy()
        x_rec_norm_list.append(x_rec_norm)
        if verbose:
            bar.update(1)

    if verbose:
        bar.close()

    return x_rec_norm_list


def get_closer_covar_rows(age: float, sex: int, dx: int, train_dataset: pd.DataFrame, normalizer, age_std_threshold: float = 0.2) -> pd.DataFrame:
    """Find real dataset rows whose covariates closely match a target (age, sex, dx).

    Args:
        age: Raw target age; standardized internally via ``normalizer``.
        sex: Integer-coded target sex (exact match required).
        dx: Integer-coded target diagnosis (exact match required).
        train_dataset: DataFrame with standardized ``"age"``, integer
            ``"sex"``, and integer ``"dx"`` columns.
        normalizer: Fitted normalizer used to standardize ``age``.
        age_std_threshold: Maximum allowed absolute difference between a
            row's standardized age and the target's, for the row to be
            considered a match.

    Returns:
        Subset of ``train_dataset`` (with an added ``"diff_age"`` column)
        matching ``sex``, ``dx``, and the age threshold.
    """
    age = normalizer.transform_single(age, "age")
    dataset_df = train_dataset.copy()
    dataset_df["diff_age"] = np.abs(dataset_df["age"] - age)

    dataset_df = dataset_df[(dataset_df["diff_age"]<age_std_threshold) & (dataset_df["sex"]==sex) & (dataset_df["dx"]==dx)]
    return dataset_df


def compute_distribution_synthesis(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, conditions_keys_ordered: list[str], train_df: pd.DataFrame, normalizer, age: float, sex: int, dx: int, max_tests: int | None = None, fgr: float = 1.0, save_path: str | None = None, seed: int = 42, age_std_threshold: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Compare generated vs. real ROI-volume distributions for one covariate group.

    Finds real subjects matching the target covariates (widening the age
    threshold if fewer than 5 matches are found), generates the same
    number of synthetic ROI-volume profiles, and optionally saves a
    per-structure boxplot comparison.

    Args:
        unet: Diffusion model.
        conditions_model: Covariates embedding model.
        noise_scheduler: Configured noise scheduler.
        conditions_keys_ordered: Ordered list of ROI-volume condition column names.
        train_df: Training DataFrame (standardized volumes + covariates).
        normalizer: Fitted normalizer for the age covariate.
        age: Raw target age.
        sex: Integer-coded target sex.
        dx: Integer-coded target diagnosis.
        max_tests: Maximum number of samples to generate (capped further
            by the number of matching real subjects found).
        fgr: Classifier-free-guidance ratio.
        save_path: Optional path to save a boxplot comparison figure. If
            ``None``, no figure is created.
        seed: Base seed for generation.
        age_std_threshold: Initial age-matching threshold (doubled once
            if fewer than 5 real matches are found).

    Returns:
        ``(x_real, x_rec)``: real matched ROI-volume values and generated
        ROI-volume values, both shape ``(num_tests, num_structures)``.

    Side Effects:
        If ``save_path`` is given, creates the parent directory (if
        needed) and writes a matplotlib figure to that path.
    """
    # obtain org values to compare
    df_org = get_closer_covar_rows(age, sex, dx, train_df, normalizer, age_std_threshold=age_std_threshold)[conditions_keys_ordered]
    if len(df_org) < 5:
        df_org = get_closer_covar_rows(age, sex, dx, train_df, normalizer, age_std_threshold=age_std_threshold*2)[conditions_keys_ordered]
    num_tests = min(len(df_org), max_tests)
    logger.info(f"age: {age}, sex: {sex}, dx: {dx} -> using num_tests={num_tests}")
            
    nb_tissues = len(conditions_keys_ordered)
    inference_results = multi_inference_from_noise(unet, conditions_model, noise_scheduler, nb_tissues=nb_tissues, normalizer=normalizer, age=age, sex=sex, dx=dx, num_tests=num_tests, fgr=fgr, seed=seed)
    inference_results = np.array(inference_results).squeeze()

    x_real = df_org.values
    x_rec = inference_results
    
    if save_path is not None:
        col_names = df_org.columns.tolist()
        positions = []
        labels = []
        pos = 1
        fig, ax = plt.subplots(figsize=(len(col_names) * 4, 8))
        for i in range(len(col_names)):
            col_name = col_names[i]
            data_gt = df_org.values[:, i]
            data_rec = inference_results[:, i]

            # Dibujar boxplots de GT y REC
            bp = ax.boxplot(
                [data_gt, data_rec],
                positions=[pos, pos + 1],
                widths=0.6,
                patch_artist=True,
            )

            # Dibujar puntos individuales sobre los boxplots
            jitter_gt = np.random.normal(pos, 0.05, size=len(data_gt))
            jitter_rec = np.random.normal(pos + 1, 0.05, size=len(data_rec))

            ax.scatter(jitter_gt, data_gt, color="tab:blue", alpha=0.6, s=20, edgecolor="k", linewidth=0.3, label="_nolegend_", zorder=2)
            ax.scatter(jitter_rec, data_rec, color="tab:orange", alpha=0.6, s=20, edgecolor="k", linewidth=0.3, label="_nolegend_", zorder=2)

            # Guardar etiquetas y posiciones
            labels.extend([f"{col_name} (GT)", f"{col_name} (REC)"])
            positions.extend([pos, pos + 1])

            # Avanzar posición: +3 para dejar separación entre grupos
            pos += 3

        # add a label at the beginning with len(df_org) and num_tests
        ax.text(0.0, 0.0, f"Samples: {df_org.shape[0]}\nTests: {num_tests}", transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # draw horizontal green line at 0
        ax.axhline(y=0, color='g', linestyle='--', linewidth=1)
        # Configuración del eje
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Values (normalized)")
        ax.set_xlabel("Features")
        ax.set_ylim(-3, 3)

        ax.set_title("Boxplots: Ground Truth vs Reconstruction")

        fig.tight_layout()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=75, bbox_inches='tight')
            plt.close(fig)
    
    return x_real, x_rec


def compute_distribution_loss(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, normalizer, step: int, args: argparse.Namespace, output_dir_results: str, output_dir_plots: str) -> tuple[dict, dict]:
    """Evaluate distribution-matching quality (Frechet distance) across a grid of validation covariates.

    Args:
        unet: Diffusion model.
        conditions_model: Covariates embedding model.
        noise_scheduler: Configured noise scheduler.
        normalizer: Fitted normalizer for the age covariate.
        step: Current global training step (used for output filenames).
        args: Training args; must include ``training_dataset_path_name``,
            ``val_covars_list``, ``free_guidance_ratio``, ``val_seed``,
            ``val_age_std_threshold``.
        output_dir_results: Directory to write the aggregate metrics JSON to.
        output_dir_plots: Directory to write per-covariate-group boxplot
            comparisons to.

    Returns:
        ``(results, agg)`` as returned by :func:`evaluate_groups`.

    Side Effects:
        Writes ``{output_dir_results}/distribution_evaluation_step_{step}.json``
        and one boxplot figure per entry in ``args.val_covars_list`` under
        ``output_dir_plots``.
    """

    real_dict = {}
    gen_dict = {}

    train_df = pd.read_csv(args.training_dataset_path_name)
    train_df = train_df[train_df[cfg.COL_SPLIT] == "train"]
    
    for covars in args.val_covars_list:
        plot_name = f"boxplot_age_{covars['age']}_sex_{covars['sex']}_dx_{covars['dx']}.png"
        plot_path_name = os.path.join(output_dir_plots, plot_name)

        x_real, x_rec = compute_distribution_synthesis(unet, 
                                                       conditions_model, 
                                                       noise_scheduler, 
                                                       args.conditions_keys_ordered,
                                                       train_df,
                                                         normalizer,
                                                       age=covars['age'], 
                                                       sex=covars['sex'], 
                                                       dx=covars['dx'], 
                                                       max_tests=100, 
                                                       fgr=args.free_guidance_ratio, 
                                                       save_path=plot_path_name,
                                                       seed=args.val_seed,
                                                       age_std_threshold=args.val_age_std_threshold)

        real_dict[(covars['age'], covars['sex'], covars['dx'])] = x_real
        gen_dict[(covars['age'], covars['sex'], covars['dx'])] = x_rec
    results, agg = evaluate_groups(real_dict, gen_dict, min_n=5, n_bootstrap=200, random_state=0)
    
    # save agg results to a json file
    agg_save_path = os.path.join(output_dir_results, f"distribution_evaluation_step_{step}.json")
    with open(agg_save_path, "w") as f:
        json.dump(agg, f)

    return results, agg



def load_multiple_real_inputs(train_df: pd.DataFrame, conditions_keys_ordered: list[str], nb_samples: int = 100, only_baseline: bool = True, filters: dict | None = None, seed: int = 42) -> tuple[np.ndarray, list[torch.Tensor], list[dict]]:
    """Sample a batch of real subjects' ROI-volume vectors and covariates.

    Args:
        train_df: Either a path (str) or DataFrame accepted by
            :class:`src.utils.load_dataset.LoadDataset`.
        conditions_keys_ordered: Ordered list of ROI-volume condition column names.
        nb_samples: Number of subjects to sample (via
            ``train_df.sample(n=nb_samples, ...)`` -- sampled from the
            *original* ``train_df``, not the ``only_baseline``-filtered
            one; see implementation note in the body).
        only_baseline: If True, restrict the candidate pool (used to
            build ``dataset_df``, though see note above about what is
            actually sampled from) to each subject's baseline session.
        filters: Optional column-based filters applied to ``dataset_df``.
        seed: Random seed for the sampling (``random_state``).

    Returns:
        ``(input_vec, covars_list, meta_data)`` where:
            - ``input_vec`` has shape ``(nb_samples, 1, num_conditions)``.
            - ``covars_list`` is a list of ``nb_samples`` tensors, each
              shape ``(1, 1 + 1 + 3)`` (age, sex, one-hot dx).
            - ``meta_data`` is a list of ``{subject_id, session_id}`` dicts.
    """
    if only_baseline:
        dataset_df = load_dataset.LoadDataset(train_df, sid_column=cfg.COL_SUBJECT_ID, session_column=cfg.COL_SESSION_ID, age_column=cfg.COL_AGE).get_baseline_df()
    else:
        dataset_df = train_df.copy()

    if filters is not None:
        for key, value in filters.items():
            dataset_df = dataset_df[dataset_df[key].isin(value)]
            
    dataset_df = dataset_df.reset_index(drop=True)
    rows = train_df.sample(n=nb_samples, random_state=seed).reset_index(drop=True)
    # rows = dataset_df.iloc[:nb_samples]
    

    # Matriz de condiciones
    input_vec = rows[conditions_keys_ordered].values.astype(np.float32)
    # add extra dimension to be in the shape (nb_samples, 1, nb_conditions)
    input_vec = np.expand_dims(input_vec, axis=1)

    # Covariables: edad, sexo y dx (one-hot)
    covars_list = []
    for _, row in rows.iterrows():
        age = torch.tensor([[row[cfg.COL_AGE]]], dtype=torch.float32)
        sex = torch.tensor([[row[cfg.COL_SEX]]], dtype=torch.float32)
        dx = F.one_hot(torch.tensor([int(row[cfg.COL_DX])]), num_classes=3).float()
        covars_list.append(torch.cat([age, sex, dx], dim=1))
    
    # covars = torch.cat(covars_list, dim=0)  # (nb_samples, n_covars)

    # Metadata
    meta_data = [{cfg.COL_SUBJECT_ID: row[cfg.COL_SUBJECT_ID], cfg.COL_SESSION_ID: row[cfg.COL_SESSION_ID]} for _, row in rows.iterrows()]

    return input_vec, covars_list, meta_data


def invert_latents(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, 
                    input_vec: np.ndarray, covars: torch.Tensor,
                    free_guidance_ratio: float = 7.5, num_inner_steps: int = 10, early_stop_epsilon: float = 1e-8,
                    compute_uncond_embeddings: bool = True) -> dict:
    """Invert a real ROI-volume vector into its noisy-latent + null-text-embedding trajectory.

    Wraps :class:`src.brainst_vol.null_inversion.NullInversion` to recover
    the noise trajectory (via DDIM inversion) and, optionally, refine
    per-step unconditional embeddings (via null-text optimization) so
    that re-running the forward diffusion from these latents accurately
    reconstructs ``input_vec``.

    Args:
        unet: Diffusion model.
        conditions_model: Covariates embedding model.
        noise_scheduler: Configured noise scheduler.
        input_vec: The real ROI-volume vector to invert.
        covars: Covariates tensor describing ``input_vec``'s subject.
        free_guidance_ratio: Classifier-free-guidance strength used
            during inversion.
        num_inner_steps: Number of inner optimization steps per DDIM
            step during null-text optimization.
        early_stop_epsilon: Early-stopping loss threshold for null-text
            optimization.
        compute_uncond_embeddings: If False, skip null-text optimization
            entirely (faster, less accurate reconstruction).

    Returns:
        A dict with keys ``"noisy_latents"`` (the most-noised latent, on
        CPU), ``"ddim_latents"`` (the full inversion trajectory, on CPU),
        and ``"uncond_embeddings"`` (list of per-step embeddings on CPU,
        or ``None`` if ``compute_uncond_embeddings=False``).

    Side Effects:
        Clears the CUDA cache after detaching all returned tensors.
    """
    # -------- Null inversion
    # instantiate null inversion and invert
    null_inversion_class = null_inversion.NullInversion(unet, conditions_model, noise_scheduler, free_guidance_ratio=free_guidance_ratio)
    
    # prepare the conditions tensor
    # conditioning_tensor = conditions_model(covars.to(device)).to(device)
    ddim_latents, uncond_embeddings = null_inversion_class.invert(input_vec, covars, num_inner_steps=num_inner_steps, early_stop_epsilon=early_stop_epsilon, verbose=False, compute_uncond_embeddings=compute_uncond_embeddings)

    # detach all latents to free memory
    ddim_latents = [__lt.detach().cpu() for __lt in ddim_latents]
    if uncond_embeddings is not None:
      uncond_embeddings = [__uemb.detach().cpu() for __uemb in uncond_embeddings]
    noisy_latent = ddim_latents[-1]
    torch.cuda.empty_cache()

    # return noisy_latent, ddim_latents, uncond_embeddings
    return {
        "noisy_latents": noisy_latent,
        "ddim_latents": ddim_latents,
        "uncond_embeddings": uncond_embeddings
    }
    
def compute_reconstruction_loss(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, normalizer, step: int, args: argparse.Namespace, output_dir_results: str) -> dict:
    """Measure reconstruction accuracy via DDIM inversion + re-generation on real subjects.

    For a random batch of real subjects, inverts each subject's true
    ROI-volume vector (without null-text optimization, for speed), then
    regenerates it from the inverted latents and measures the mean
    absolute error against the ground truth.

    Args:
        unet: Diffusion model.
        conditions_model: Covariates embedding model.
        noise_scheduler: Configured noise scheduler.
        normalizer: Fitted normalizer (currently unused in this function
            body beyond being accepted for interface consistency).
        step: Current global training step (used for output filenames).
        args: Training args; must include ``training_dataset_path_name``,
            ``val_nb_reconstruction_samples``, ``free_guidance_ratio``,
            ``conditions_keys_ordered``, and optionally ``val_seed``.
        output_dir_results: Directory to write per-subject reconstruction
            CSVs and the aggregate MAE JSON to.

    Returns:
        ``{"rec_mae": float}``, the mean absolute reconstruction error
        across all sampled subjects and structures.

    Side Effects:
        Writes ``reconstruction_org_rec_step_{step}.csv``,
        ``reconstruction_diff_step_{step}.csv``, and
        ``reconstruction_mae_step_{step}.json`` under ``output_dir_results``.
    """

    # find subjects
    train_df = pd.read_csv(args.training_dataset_path_name)
    train_df = train_df[train_df[cfg.COL_SPLIT] == "train"]

    # load multiple inputs (seed for reproducibility)
    input_vec, covars, meta_data = load_multiple_real_inputs(
        train_df,
        args.conditions_keys_ordered,
        nb_samples=args.val_nb_reconstruction_samples,
        only_baseline=True,
        filters=None,
        seed=getattr(args, "val_seed", None)  # soporta seed opcional
    )

    # apply null inversion and reconstruction
    reconstruction_list = []
    bar = tqdm(total=input_vec.shape[0], desc="Reconstructing samples")
    for i in range(input_vec.shape[0]):
        # batch-safe: mantengo dim 0

        inversion = invert_latents(
            unet,
            conditions_model,
            noise_scheduler,
            input_vec=input_vec[i],
            covars=covars[i],
            free_guidance_ratio=args.free_guidance_ratio,
            num_inner_steps=10,
            early_stop_epsilon=1e-10,
            compute_uncond_embeddings=False
        )

        reconstruction = diffusion_loop(
            inversion["noisy_latents"],
            unet,
            conditions_model,
            noise_scheduler,
            covars[i],
            uncond_embeddings=inversion["uncond_embeddings"],
            free_guidance_ratio=args.free_guidance_ratio
        )

        # reconstruction = reconstruction.cpu().numpy().squeeze()
        reconstruction_list.append(reconstruction)
        bar.update(1)
    bar.close()

    # diff_df
    input_vec = input_vec.squeeze()
    reconstruction_np = np.array(reconstruction_list).squeeze()
    diff_np = input_vec - reconstruction_np
    diff_df = pd.DataFrame(diff_np, columns=args.conditions_keys_ordered)
    diff_df.insert(0, cfg.COL_SESSION_ID, [m[cfg.COL_SESSION_ID] for m in meta_data])
    diff_df.insert(0, cfg.COL_SUBJECT_ID, [m[cfg.COL_SUBJECT_ID] for m in meta_data])

    # org_rec_df
    org_df = pd.DataFrame(input_vec, columns=args.conditions_keys_ordered)
    rec_df = pd.DataFrame(reconstruction_np, columns=args.conditions_keys_ordered)

    org_rec_df = pd.concat(
        [org_df.add_suffix("_org"), rec_df.add_suffix("_rec")],
        axis=1
    )
    org_rec_df.insert(0, cfg.COL_SESSION_ID, [m[cfg.COL_SESSION_ID] for m in meta_data])
    org_rec_df.insert(0, cfg.COL_SUBJECT_ID, [m[cfg.COL_SUBJECT_ID] for m in meta_data])

    # save results
    org_rec_df_name = os.path.join(output_dir_results, f"reconstruction_org_rec_step_{step}.csv")
    diff_df_name = os.path.join(output_dir_results, f"reconstruction_diff_step_{step}.csv")
    org_rec_df.to_csv(org_rec_df_name, index=False)
    diff_df.to_csv(diff_df_name, index=False)

    # mean absolute error
    mae = np.mean(np.abs(np.array(diff_np)))
    
    mae_json = {"reconstruction_mae": float(mae)
    }
    
    # save mae to json
    mae_json_name = os.path.join(output_dir_results, f"reconstruction_mae_step_{step}.json")
    with open(mae_json_name, "w") as f:
        json.dump(mae_json, f)

    return {
        "rec_mae": float(mae)
    }


def validation(
                unet: torch.nn.Module,
                conditions_model: torch.nn.Module,
                noise_scheduler,
                normalizer,
                step: int,
                args: argparse.Namespace,
               ) -> dict:
    """Run full BrainST-vol validation: reconstruction accuracy + distribution matching.

    Combines :func:`compute_reconstruction_loss` (how accurately real
    profiles can be inverted and regenerated) and
    :func:`compute_distribution_loss` (how closely freshly-sampled
    profiles match the real distribution, per covariate group, via
    Frechet distance).

    Args:
        unet: Diffusion model.
        conditions_model: Covariates embedding model.
        noise_scheduler: Configured noise scheduler.
        normalizer: Fitted normalizer for the age covariate.
        step: Current global training step (used for output paths/naming).
        args: Training args; must include (at least) ``output_path``,
            ``val_imgs_dir_name``, plus everything required by
            :func:`compute_reconstruction_loss` and
            :func:`compute_distribution_loss`.

    Returns:
        ``{"frechet_mean": float, "rec_mae": float}``.

    Side Effects:
        Creates ``{args.output_path}/{args.val_imgs_dir_name}/step_{step}/{plots,results}``
        and writes all artifacts produced by the two sub-evaluations
        into those directories.
    """
    logger.info(f"Validation step {step}, fgr: {args.free_guidance_ratio}...")
    output_dir = os.path.join(args.output_path, args.val_imgs_dir_name, f"step_{step}")
    output_dir_plots = os.path.join(output_dir, "plots")
    output_dir_results = os.path.join(output_dir, "results")
    os.makedirs(output_dir_plots, exist_ok=True)
    os.makedirs(output_dir_results, exist_ok=True)
    
    # computing reconstruction loss
    rec_results = compute_reconstruction_loss(
                                        unet,
                                        conditions_model,
                                        noise_scheduler,
                                        normalizer,
                                        step,
                                        args,
                                        output_dir_results=output_dir_results)
    
    
    # compute distributions loss
    dist_results, dist_agg = compute_distribution_loss(
                                        unet,
                                        conditions_model,
                                        noise_scheduler,
                                        normalizer,
                                        step,
                                        args,
                                        output_dir_results=output_dir_results,
                                        output_dir_plots=output_dir_plots)
    

    return  {"frechet_mean": dist_agg["frechet_mean"], "rec_mae": rec_results["rec_mae"]}






def save_model(unet: torch.nn.Module, conditions_model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler, global_step: int, out_model_path: str, ema: EMA | None = None, best: bool = False) -> None:  # MOD: se añade parámetro ema
    """Save a training checkpoint (UNet, conditions model, optimizer, scheduler, EMA).

    Args:
        unet: The diffusion UNet (state dict saved via ``.module`` if
            wrapped in ``DistributedDataParallel``).
        conditions_model: Model embedding the covariates.
        optimizer: Optimizer whose state is checkpointed.
        lr_scheduler: LR scheduler whose state is checkpointed (or ``None``).
        global_step: Current global training step, stored under
            ``"num_train_timesteps"`` and used to name the checkpoint file.
        out_model_path: Directory to write the checkpoint to.
        ema: Optional :class:`EMA` tracker; if given, its shadow weights
            are also saved.
        best: If True, the checkpoint is saved with a ``_best`` suffix,
            and any previously-saved ``*_best.pt`` checkpoint in
            ``out_model_path`` is deleted first.

    Side Effects:
        Writes ``{out_model_path}/model_{global_step}.pt`` (or
        ``..._best.pt``) to disk; deletes any prior best checkpoint when
        ``best=True``; triggers a CUDA cache clear and Python GC.
    """
    # Guardar el modelo
    unet_state_dict = unet.module.state_dict() if torch.distributed.is_initialized() else unet.state_dict()
    checkpoint = {
        "unet_state_dict": unet_state_dict,
        "conditions_model_state_dict": conditions_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_train_timesteps": global_step,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }
    # MOD: Agregar los pesos EMA en el checkpoint
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.shadow

    if best:
        # find the best checkpoint (is the file that ends by _best.pt)
        path_name_old_best_chk = glob.glob(os.path.join(out_model_path, "*_best.pt"))
        if path_name_old_best_chk:
            os.remove(path_name_old_best_chk[0])
        global_step = f"{global_step}_best"

    torch.save(checkpoint, f"{out_model_path}/model_{global_step}.pt")
    logger.info(f"Model saved in {out_model_path}/model_{global_step}.pt")
    
    del checkpoint, unet_state_dict
    torch.cuda.empty_cache()
    gc.collect()


def load_checkpoint(checkpoint_path: str, unet: torch.nn.Module, conditions_model: torch.nn.Module, device: torch.device, train_dataloader_len: int,
                    gradient_accumulation_steps: int, batch_size: int, optimizer: torch.optim.Optimizer | None = None, lr_scheduler=None, ema: EMA | None = None) -> tuple[int, int]:
    """Load a training checkpoint and restore model/optimizer/scheduler/EMA state.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file (loaded to CPU first).
        unet: UNet module to load weights into (moved to ``device`` after loading).
        conditions_model: Conditions-embedding module to load weights into.
        device: Device to move models (and optimizer/EMA buffers) to after loading.
        train_dataloader_len: Length of the training DataLoader, used to compute ``first_epoch``.
        gradient_accumulation_steps: Used to compute ``first_epoch``.
        batch_size: Used to compute ``first_epoch``.
        optimizer: Optional optimizer to restore state into (buffers moved to ``device``).
        lr_scheduler: Optional LR scheduler to restore state into.
        ema: Optional :class:`EMA` tracker to restore shadow weights into (moved to ``device``).

    Returns:
        ``(global_step, first_epoch)`` recovered from the checkpoint.

    Side Effects:
        Mutates ``unet``, ``conditions_model``, ``optimizer``,
        ``lr_scheduler``, and ``ema`` in place; moves all restored tensors
        to ``device``.
    """
    # 1. Load checkpoint on CPU to avoid using VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Load weights into models
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    conditions_model.load_state_dict(checkpoint["conditions_model_state_dict"], strict=False)
    # 3. Move models to GPU
    unet.to(device)
    conditions_model.to(device)

    # 4. EMA (optional)
    if ema is not None and "ema_state_dict" in checkpoint:
        # Move only the EMA tensors to GPU
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_state_dict"].items()}

    # 5. Optimizer and scheduler (optional)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        # For Adam, some states may be on CPU and others on GPU
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer buffers to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint and checkpoint["lr_scheduler_state_dict"] is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # 6. Compute first_epoch and global_step
    global_step = checkpoint["num_train_timesteps"]
    first_epoch = (global_step * gradient_accumulation_steps * batch_size) // train_dataloader_len + 1

    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Resuming training from epoch {first_epoch} and global step {global_step}")

    return global_step, first_epoch



def save_configurations(args_train: argparse.Namespace, networks_config: argparse.Namespace, config_path: str, config_name: str = "model_config.json") -> None:
    """Serialize the training and network configs to a JSON file.

    Args:
        args_train: Training configuration (an ``argparse.Namespace``,
            possibly nested).
        networks_config: Network architecture configuration.
        config_path: Directory to write the config file into.
        config_name: Output filename.

    Side Effects:
        Writes ``{config_path}/{config_name}`` to disk.
    """
    argparse_dict = {
        "args_train": fc.args_to_dict(args_train, deep_conversion=True),
        "networks_config": fc.args_to_dict(networks_config, deep_conversion=True)
    }
    argparse_json = json.dumps(argparse_dict, indent=4)
    with open(os.path.join(config_path, config_name), "w") as outfile:
        outfile.write(argparse_json)
    logger.info(f"Model configurations saved in: {os.path.join(config_path, config_name)}")





class EMA:
    """Exponential moving average of a model's parameters.

    Maintains a "shadow" copy of each trainable parameter, updated as
    ``shadow = decay * shadow + (1 - decay) * param`` on every call to
    :meth:`update`. :meth:`apply_shadow`/:meth:`restore` allow temporarily
    swapping the model's live weights for the EMA weights (e.g. during
    validation) and swapping back afterward.
    """

    def __init__(self, model: torch.nn.Module, decay: float, warm_up_steps: int = 0, warm_up_decay: float = 0.1, optimize_cpu: bool = False):
        """
        Initializes the EMA class to manage the exponential moving average
        of the model parameters.

        Args:
            model (torch.nn.Module): The model whose parameters will be averaged.
            decay (float): Decay rate for the EMA.
            warm_up_steps: Number of initial steps during which ``warm_up_decay`` is used instead of ``decay`` (faster initial tracking).
            warm_up_decay: Decay rate used during the warm-up period.
            optimize_cpu: If True, keep the EMA shadow parameters on CPU (saves GPU memory at the cost of extra host-device transfers).
        """
        self.model = model
        self.decay = decay
        self.warm_up_steps = warm_up_steps
        self.warm_up_decay = warm_up_decay
        self.shadow = {}
        self.backup = {}
        self.optimize_cpu = optimize_cpu

        for name, param in model.named_parameters():
            if param.requires_grad:
                if self.optimize_cpu:
                    self.shadow[name] = param.detach().cpu().clone()
                else:
                    self.shadow[name] = param.detach().clone()

    def update(self, step: int | None = None) -> None:
        """
        Updates the shadow parameters using the exponential moving average.

        Args:
            step: Current training step; if given and less than
                ``self.warm_up_steps``, ``self.warm_up_decay`` is used
                instead of ``self.decay`` for this update.
        """
        decay = self.decay
        if step is not None and self.warm_up_steps > 0 and step < self.warm_up_steps:
            decay = self.warm_up_decay
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.optimize_cpu:
                    new_avg = (1.0 - decay) * param.detach().cpu() + decay * self.shadow[name]
                else:
                    new_avg = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_avg

    def apply_shadow(self) -> None:
        """
        Applies the averaged (EMA) parameters to the model while saving the original ones.

        Side Effects:
            Overwrites ``self.model``'s parameters in place with the EMA
            shadow values; stores the original values in ``self.backup``
            for later restoration via :meth:`restore`.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.optimize_cpu:
                    self.backup[name] = param.detach().clone()  # keep original
                    param.data.copy_(self.shadow[name].to(param.device)) 
                else:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """
        Restores the model's original parameters.

        Side Effects:
            Overwrites ``self.model``'s parameters in place with the
            values saved in ``self.backup`` by :meth:`apply_shadow`, then
            clears ``self.backup``.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def create_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_start_factor: float, warmup_steps: int, max_train_steps: int, eta_min: float):
    """
    Creates a SequentialLR with a Linear warmup followed by CosineAnnealingLR.

    "warmup_start_factor": 1e-2 # Initial learning rate factor (multiplied by base_lr) during warmup
    "warmup_steps": 1000, # Number of steps for warmup
    "eta_min": 1e-6, # Minimum learning rate after warmup

    Args:
        optimizer: Optimizer to schedule.
        warmup_start_factor: Initial LR multiplier at the start of warmup.
        warmup_steps: Number of linear-warmup steps.
        max_train_steps: Total training steps (the cosine phase runs for ``max_train_steps - warmup_steps``).
        eta_min: Minimum LR reached at the end of the cosine phase.

    Returns:
        A ``torch.optim.lr_scheduler.SequentialLR`` combining linear warmup then cosine annealing.
    """

    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Cosine decay scheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_train_steps - warmup_steps,
        eta_min=eta_min
    )

    # Combine them
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler







def train(
    args_train: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run the full BrainST-vol training loop.

    Instantiates models, dataset, optimizer/scheduler, optionally resumes
    from a checkpoint or loads pretrained weights, then iterates over
    epochs/batches performing the diffusion denoising-loss training step
    (with optional Gaussian augmentation noise on the target volumes),
    with gradient accumulation, periodic checkpointing, periodic
    validation (with EMA swap-in/out; see :func:`validation`), and early
    stopping on validation loss plateau.

    Args:
        args_train: Full training configuration (see the ``args_train``
            dict defined at the bottom of this module for every expected
            field).
        device: Device to train on.

    Side Effects:
        This is the main entry point of the script: it creates output
        directories, writes TensorBoard logs, periodically writes model
        checkpoints and validation artifacts to disk, and runs for
        ``args_train.max_train_steps`` optimizer steps (or until early
        stopping triggers).
    """

    # ---- reproducibility
    set_seed(args_train.seed)
    gen_t = torch.Generator().manual_seed(args_train.seed) 
    gen_noise = torch.Generator().manual_seed(args_train.seed)
    gen_dataloader = torch.Generator().manual_seed(args_train.seed)
    gen_free_guidance = torch.Generator().manual_seed(args_train.seed)
    gen_augmentation_noise = torch.Generator().manual_seed(args_train.seed)

    # ---- instantiate models
    networks_config = fc.dict_to_args(cfg.ARCHITECTURE_BRAINST_VOL, deep_conversion=True)
    models_dict = instantiate_models.instantiate_conditioned_models(networks_config, cfg.DEVICE, args_train.val_num_diffusion_steps)

    unet = models_dict["unet"]
    conditions_model = models_dict["conditions_model"]
    noise_scheduler = models_dict["noise_scheduler"]
    
    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        training_dataset_path_name=args_train.training_dataset_path_name,
        conditions_keys_ordered=args_train.conditions_keys_ordered,
        batch_size=args_train.batch_size,
        gen_dataloader=gen_dataloader,
        dataset_filters=args_train.dataset_filters,
        max_timepoints_per_epoch=args_train.max_timepoints_per_epoch,
    )

    # ---- create folders
    os.makedirs(args_train.output_path, exist_ok=True)
    _checkpoint_dir_name =  os.path.join(args_train.output_path, args_train.checkpoints_dir_name)
    _logs_dir_name = os.path.join(args_train.output_path, args_train.logs_dir_name)
    _val_imgs_dir_name = os.path.join(args_train.output_path, args_train.val_imgs_dir_name)
    os.makedirs(_checkpoint_dir_name, exist_ok=True)
    os.makedirs(_logs_dir_name, exist_ok=True)
    os.makedirs(_val_imgs_dir_name, exist_ok=True)

    # ---- save configurations
    save_configurations(args_train, networks_config, args_train.output_path)


    # ---- create tensorboard writer and save configurations
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  # Formato: Año-Mes-Día_Hora-Minuto
    _sum_writter_dir = os.path.join(_logs_dir_name, f"logs_{timestamp}")
    os.makedirs(_sum_writter_dir, exist_ok=True)
    writer = SummaryWriter(_sum_writter_dir)

    # ---- optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(conditions_model.parameters()),
        lr=args_train.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    if args_train.lr_scheduler is not None:
        if args_train.lr_scheduler.name == "PolynomialLR":
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args_train.max_train_steps, power=args_train.lr_scheduler.power)
        elif args_train.lr_scheduler.name == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_train.max_train_steps, eta_min=args_train.lr_scheduler.eta_min)
        elif args_train.lr_scheduler.name == "WarmupCosineLR":
            lr_scheduler = create_warmup_cosine_scheduler(optimizer,
                                                          warmup_start_factor=args_train.lr_scheduler.warmup_start_factor,
                                                          warmup_steps=args_train.lr_scheduler.warmup_steps,
                                                          max_train_steps=args_train.max_train_steps,
                                                          eta_min=args_train.lr_scheduler.eta_min)
    else:
        lr_scheduler = None

    # ---- loss function
    loss_pt = torch.nn.MSELoss()

    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (args_train.max_train_steps * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
    logger.info(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)
    conditions_model.to(device)

    # Initilize ema
    if args_train.use_ema:
        ema = EMA(unet, 
                  decay=args_train.ema_params.decay, 
                  warm_up_steps=args_train.ema_params.warm_up_steps, 
                  warm_up_decay=args_train.ema_params.warm_up_decay,
                  optimize_cpu=False)
    else:
        ema = None

    # priority is to resume from check point
    if args_train.resume_from_checkpoint_path_name is not None:
        global_step, first_epoch = load_checkpoint(
            args_train.resume_from_checkpoint_path_name,
            unet,
            conditions_model,
            device=device,
            train_dataloader_len=len(train_dataloader),
            gradient_accumulation_steps=args_train.gradient_accumulation_steps,
            batch_size=args_train.batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema
        )

    elif args_train.load_pretrained_model_from is not None:
        checkpoint = torch.load(args_train.load_pretrained_model_from, map_location=device_name)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        conditions_model.load_state_dict(checkpoint["conditions_model_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
            logger.info("EMA state loaded from checkpoint")
        logger.info(f"Pretrained model loaded from {args_train.load_pretrained_model_from}")
    
    unet.train()
    conditions_model.train() 

    # ---- memory reduction
    # -------- automatic mixed precision
    if args_train.amp:
        scaler = GradScaler()
    else:
        scaler = None
    gradient_accumulation_count = 0

    # ---- copy normalizer parameters .json to the model folder
    shutil.copy2(args_train.normalizer_params, os.path.join(args_train.output_path, "normalizer_params.json"))
    

    early_stopping = False
    best_val_loss = float("inf")
    loss_val=-1
    patience_counter = 0 


    # ---- training loop
    progress_bar = tqdm(
        range(args_train.max_train_steps),
        desc="Training",
        initial=global_step
    )

    # validation
    normalizer = data_normalization.SavedNormalizerBrainStructures(args_train.normalizer_params)
    early_stopping = False
    best_val_loss = float("inf")
    loss_val=-1
    patience_counter = 0

    for epoch in range(first_epoch, max_epochs):
        for batch in train_dataloader:

            # prepare inputs
            cond_list = [batch[key] for key in args_train.conditions_keys_ordered]  # list of (B, n_conditions, 1)
            volumens = torch.stack(cond_list, dim=1).float().squeeze(-1) # (B, num_conditions)

            if args_train.augmentation_std > 0:
                augmentation_noise = torch.empty_like(volumens).uniform_(-args_train.augmentation_std, args_train.augmentation_std, generator=gen_augmentation_noise)
                volumens += augmentation_noise
            volumens = volumens.to(device) # (B, num_conditions)

            # covars
            covars_list = [batch[key] for key in args_train.covars_list]  # list of (B, n_covars, variable)
            covars = torch.cat(covars_list, dim=1).float().to(device)  # (B, covars_dimension)

            # Forward pass
            with autocast("cuda", enabled=args_train.amp):
                # generate noise and timesteps with dedicate generatos and in the cpu for reproducibility
                noise = torch.randn(volumens.shape, device="cpu", generator=gen_noise).to(device)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(volumens)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (volumens.shape[0],), device="cpu", generator=gen_t).long().to(device)

                noisy_volumens = noise_scheduler.add_noise(original_samples=volumens, noise=noise, timesteps=timesteps)
                covars_emb = conditions_model(covars)

                # free guidance
                free_guidance_prob = torch.rand(1, generator=gen_free_guidance).item()
                if free_guidance_prob < args_train.free_guidance_threshold:
                    covars_emb = torch.zeros_like(covars_emb)

                model_output = unet(noisy_volumens, 
                                  timesteps=timesteps,
                                  context = covars_emb,
                                    )
                

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    # predict noise
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    # predict sample
                    model_gt = volumens
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    # predict velocity
                    model_gt = volumens - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from ",
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                    )

            loss = loss_pt(model_output.float(), model_gt.float())   # Dividir para escalar la pérdida
            loss = loss / args_train.gradient_accumulation_steps  # Dividir la pérdida por los pasos de acumulación de gradientes

            # Acumulación de gradientes
            if args_train.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_accumulation_count += 1  # Contador de pasos acumulados

            # Solo se actualizan los pesos cada `gradient_accumulation_steps` pasos
            if gradient_accumulation_count % args_train.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args_train.amp:
                    scaler.unscale_(optimizer)  # Desescalar antes de clipping
                torch.nn.utils.clip_grad_norm_(
                    list(unet.parameters()) + list(conditions_model.parameters()), max_norm=1.0
                )             
                if args_train.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # update ema
                if args_train.use_ema:
                    ema.update(step=global_step)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                gradient_accumulation_count = 0  # Reiniciar el contador

                # update writter 
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("Time_steps", timesteps[0], global_step)

                # update progress bar 
                progress_bar.update(1)
                # logs_conditions = {key: [round(_v.item(), 2) for _v in batch[key]] for key in args_train.conditions_keys_ordered}

                logs = {"loss": loss.detach().item(), 
                        }
                
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # save the model in intervals
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema)

                # # Validate
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    # Pause the training progress bar
                    progress_bar.set_description("Validating")
                    progress_bar.clear()
                    val_start = time.time()

                    # prepare models for validation
                    unet.eval()
                    conditions_model.eval()

                    if args_train.use_ema:
                        ema.apply_shadow()

                    # run validation
                    try:                    
                        val_results = validation(
                            unet=unet,
                            conditions_model=conditions_model,
                            noise_scheduler=noise_scheduler,
                            normalizer=normalizer,
                            step=global_step,
                            args=args_train,
                        )
                        loss_dist = val_results["frechet_mean"]
                        loss_rec = val_results["rec_mae"]
                        loss_val = loss_dist + loss_rec
                        
                        writer.add_scalar("Loss/val_dist", loss_dist, global_step)
                        writer.add_scalar("Loss/val_rec", loss_rec, global_step)
                        writer.add_scalar("Loss/val", loss_val, global_step)

                        # remove "total_vol" from val_results[1]
                        if loss_val < best_val_loss:
                            best_val_loss = loss_val
                            patience_counter = 0
                            logger.info(f"Validation loss improved at step {global_step}: {loss_val}")
                            save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, best=True)
                        else:
                            patience_counter += 1
                            if patience_counter >= args_train.patience:
                                logger.info(f"Early stopping triggered at step {global_step}")
                                early_stopping = True
                        args_train.initial_val = False

                    except Exception as e:
                        print(f"ERROR DURING VALIDATION STEP {global_step}: {e}")

                    # recover training state
                    unet.train()
                    conditions_model.train()
                    if args_train.use_ema:
                        ema.restore()

                    # Resume training progress bar
                    progress_bar.set_description("Training")
                    progress_bar.refresh()
                    progress_bar.last_print_t -= time.time() - val_start
                
                if global_step >= args_train.max_train_steps or early_stopping:
                    break

        if global_step >= args_train.max_train_steps or early_stopping:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # make  out_model_path dir if it does not exist
    save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema)






args_train = {
    # directories 
    "output_path": os.path.join(cfg.PATH_MODELS_TRAINING, "brainst_vol"),
    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",
        
    # data
    "training_dataset_path_name": os.path.join(cfg.PATH_DATA_TRAINING, "example", "training_data.csv"),
    "normalizer_params": os.path.join(cfg.PATH_DATA_TRAINING, "example", "normalized_params.json"),
    "path_name_ref_img": os.path.join(cfg.PATH_DATA_TRAINING, "images", "preprocessed", "ADNI", "ADNI002S0295", "m000", "sub-ADNI002S0295_ses-m000_run-01_T1w_synthsr.nii.gz"),

    # "conditions_keys_ordered": ["total_vol", "surrounding_csf_vol", "lateral_ventricles_vol"],  # order of the conditions in the conditioning embedding
    "conditions_keys_ordered": cfg.STRUCTURE_NAME_LIST_VOL,
    "covars_list": cfg.COVARS_LIST, # list of covariates to use as conditions
    
    "dataset_filters": None, # {dataset:["ADNI", "AIBL", "CamCAN"]}

    # overfitting
    "augmentation_std": 0.0, #0.01,
    "max_timepoints_per_epoch": 3, # max number of timepoints for a given subject to be used in each epoch. If None, all timepoints are used. This is useful for datasets with many timepoints per subject to avoid overfitting to a specific subject.

    # training configuration
    "max_train_steps": 5000, # number of training steps
    "save_checkpoint_interval": 250, # save the model every n steps

    # ---- memory reduction
    "amp": False,

    # ---- Training stability
    "batch_size": 2048, # 256, #3 
    "gradient_accumulation_steps": 1,
    "use_ema": True,
    "ema_params": {
        "decay": 0.999,
        "warm_up_steps": 1000,
        "warm_up_decay": 0.5,
    },

    # ---- optimizer
    "lr":  1e-4, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5

    # ---- lr_scheduler
    "lr_scheduler": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-2, "warmup_steps": 100, "eta_min": 1e-6},


    # ---- pretrained_model
    "load_pretrained_model_from": None, # not working

    # ---- resume from checkpoint
    "resume_from_checkpoint_path_name": None,

    # ---- reproducibility
    "seed": 42,

    # ---- Free guidance
    "free_guidance_ratio": 1.0,
    "free_guidance_threshold": 0.2,

    # ---- validation
    "val_interval": 500, # 2500
    "val_evaluation_initial_step": 20000,
    "initial_val": True,
    "val_seed": 42,
    "val_num_diffusion_steps": 30, # number of diffusion steps for the validation
    "val_age_std_threshold": 0.25,
    "val_covars_list": [
                        {"age": 55, "sex": 0, "dx": 0}, {"age": 55, "sex": 0, "dx": 1}, {"age": 55, "sex": 0, "dx": 2},
                        {"age": 60, "sex": 0, "dx": 0}, {"age": 60, "sex": 0, "dx": 1}, {"age": 60, "sex": 0, "dx": 2},
                        {"age": 65, "sex": 0, "dx": 0}, {"age": 65, "sex": 0, "dx": 1}, {"age": 65, "sex": 0, "dx": 2},
                        {"age": 70, "sex": 0, "dx": 0}, {"age": 70, "sex": 0, "dx": 1}, {"age": 70, "sex": 0, "dx": 2},
                        {"age": 75, "sex": 0, "dx": 0}, {"age": 75, "sex": 0, "dx": 1}, {"age": 75, "sex": 0, "dx": 2},
                        {"age": 80, "sex": 0, "dx": 0}, {"age": 80, "sex": 0, "dx": 1}, {"age": 80, "sex": 0, "dx": 2},
                        {"age": 85, "sex": 0, "dx": 0}, {"age": 85, "sex": 0, "dx": 1}, {"age": 85, "sex": 0, "dx": 2},
                        {"age": 90, "sex": 0, "dx": 0}, {"age": 90, "sex": 0, "dx": 1}, {"age": 90, "sex": 0, "dx": 2},
                        ],
    "val_nb_reconstruction_samples": 250,
    # ---- early stopping
    "patience": 10,

}


args_train = fc.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
)

# 23668MiB