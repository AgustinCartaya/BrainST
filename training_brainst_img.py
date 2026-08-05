"""Training script for BrainST-img: the ROI-volume-conditioned latent
diffusion model that generates 3D brain image latents.

This script wires together:
    - `PrepareTrainingDataset` / `MaxPerSubjectSampler`: a longitudinal
      dataset loader that caps the number of timepoints sampled per
      subject per epoch (to avoid overfitting on subjects with many
      follow-ups) and optionally loads per-structure attention masks.
    - `diffusion_model_unet_maisi_mask_att`: the conditioned UNet
      backbone (MAISI architecture) with optional cross-attention
      supervision against ground-truth tissue masks.
    - `validation()`: periodically generates images from fixed
      seeds/conditions, decodes them, (optionally) segments them, and
      compares the resulting ROI volumes against the requested targets.
    - `EMA`: exponential moving average of UNet weights, used for more
      stable validation/inference.

Run directly (no CLI): the hard-coded `args_train` dictionary at the
bottom of this file configures the run.
"""

from __future__ import annotations

import argparse
import datetime
import gc

# for validation
import glob
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import time
from functools import partial

import numpy as np
import pandas as pd

# pytorch
import torch

# monai
from monai.networks.schedulers.ddpm import DDPMPredictionType

# images
from PIL import Image

# data loader
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# mine
import configs as cfg
import src.utils.functions as fc
import src.utils.nifti_functions as nfc
import src.utils.util_freesurfer_segmentation as ufs
from src.brainst_img import instantiate_models
from src.brainst_img.networks_declaration import attention_controller
from src.utils import data_normalization, load_dataset, prep_segmentation

logger = logging.getLogger(__name__)

device_name = cfg.DEVICE
device = torch.device(device_name)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs (NumPy, PyTorch CPU/GPU) for reproducibility.

    Also forces cuDNN into deterministic mode. Note this does not seed
    Python's built-in ``random`` module (used elsewhere via
    dedicated calls / ``torch.Generator`` objects instead).

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

    Wraps a :class:`src.utils.load_dataset.LoadDataset`, applies optional
    column-based filters, and exposes :meth:`get_train_data` to build the
    list of instance dicts consumed by :class:`PrepareTrainingDataset`.
    """

    def __init__(self,
                training_dataset_path_name: str,
                conditions_keys_ordered: list[str],
                dataset_filters: dict | None = None,
                att_mask_resolution_list: list[tuple[int, int, int]] | None = None,
                att_mask_structure_mapping: dict | None = None):
        """Load the dataset and latents from the specified paths.

        Args:
          training_dataset_path_name: Path to the training dataset.
          conditions_keys_ordered: List of condition keys in the desired order.
          dataset_filters: Optional filters to apply to the dataset in the form of a dictionary where keys are column names and values are lists of values to filter by.
          att_mask_resolution_list: List of resolutions for the attention masks.
          att_mask_structure_mapping: Mapping of the attention mask structure.
        """
        self.complete_dataset = load_dataset.LoadDataset(training_dataset_path_name, sid_column=cfg.COL_SUBJECT_ID, session_column=cfg.COL_SESSION_ID, age_column=cfg.COL_AGE) 

        self.conditions_keys_ordered = conditions_keys_ordered
        self.att_mask_resolution_list = att_mask_resolution_list
        self.att_mask_structure_mapping = att_mask_structure_mapping

        if dataset_filters is not None:
            for column, values_list in dataset_filters.items():
                self.complete_dataset.df = self.complete_dataset.df[self.complete_dataset.df[column].isin(values_list)]

    def create_tissue_mask_path_name(self, s_tissue_mask_path: str, structure_name: str, resolution_str: str, sid: str, session_id: str) -> str:
        """Resolve the on-disk path of a precomputed tissue attention mask.

        Tries two naming conventions in order: a bare
        ``{structure_name}_{resolution_str}.npy`` file, then a
        subject/session-qualified
        ``{sid}_{session_id}_{structure_name}_{resolution_str}.npy`` file.

        Args:
            s_tissue_mask_path: Directory containing this subject's tissue masks.
            structure_name: Name of the anatomical structure.
            resolution_str: Mask resolution encoded as an underscore-joined string (e.g. ``"48_64_48"``).
            sid: Subject id (used for the fallback naming convention).
            session_id: Session id (used for the fallback naming convention).

        Returns:
            The resolved path to the mask file.

        Raises:
            ValueError: If neither naming convention matches an existing file.
        """
        
        att_mask_name = f"{structure_name}_{resolution_str}.npy"
        if os.path.exists(os.path.join(s_tissue_mask_path, att_mask_name)):
            return os.path.join(s_tissue_mask_path, att_mask_name)
        att_mask_name = f"{sid}_{session_id}_{structure_name}_{resolution_str}.npy"
        if os.path.exists(os.path.join(s_tissue_mask_path, att_mask_name)):
            return os.path.join(s_tissue_mask_path, att_mask_name)

        raise ValueError(f"Attention mask file not found for structure {structure_name} at resolution {resolution_str} for subject {sid} and session {session_id}. Checked paths: {os.path.join(s_tissue_mask_path, att_mask_name)}")


    def create_att_mask_dict(self, row: pd.Series) -> dict:
        """Build the per-resolution attention-mask path/value dict for one dataset row.

        For each requested resolution, builds a dict mapping each
        attention-map "channel" name (per ``self.att_mask_structure_mapping``)
        to either:
            - a scalar weight (int/float, used directly as a constant mask value),
            - a list/tuple of structure names (combined into one path list, merged via max later), or
            - a single structure name resolved to a mask file path.

        Args:
            row: A row of the training dataset DataFrame.

        Returns:
            ``{resolution_str: {channel_name: path_or_paths_or_value}}``.
        """
        att_mask_dict = {}
        for att_res in self.att_mask_resolution_list:
            res_str = "_".join([str(x) for x in att_res])
            _att_mask_res = {}
            for key, value in self.att_mask_structure_mapping.items():
                # if value == 1:
                if isinstance(value, int) or isinstance(value, float):
                    _att_mask_res[key] = value
                elif isinstance(value, list) or isinstance(value, tuple):
                    combined_att_map_list = []
                    for _v in value:
                        att_mask_path_name = self.create_tissue_mask_path_name(row[cfg.COL_PREP_TISSUE_MASKS_PATH], _v, res_str, row[cfg.COL_SUBJECT_ID], row[cfg.COL_SESSION_ID])
                        combined_att_map_list.append(att_mask_path_name)
                    _att_mask_res[key] = combined_att_map_list
                else:
                    att_mask_path_name = self.create_tissue_mask_path_name(row[cfg.COL_PREP_TISSUE_MASKS_PATH], value, res_str, row[cfg.COL_SUBJECT_ID], row[cfg.COL_SESSION_ID])
                    _att_mask_res[key] = att_mask_path_name
            att_mask_dict[res_str] = _att_mask_res
        return att_mask_dict

    def get_train_data(self, only_baseline: bool = False) -> list[dict]:
        """Assemble the list of training instances (one dict per subject/session row).

        Args:
            only_baseline: If True, use only each subject's baseline
                (first) session instead of every available session.

        Returns:
            A list of instance dicts, each with keys ``"id"``,
            ``"latent_path_name"``, one entry per key in
            ``self.conditions_keys_ordered``, and (if attention masks are
            configured) an ``"att_mask"`` entry. Rows whose
            ``cfg.COL_PREP_LATENT_PATH`` is ``NaN`` are skipped.

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
            if not pd.isna(row[cfg.COL_PREP_LATENT_PATH]):
                _instance = {}
                _instance["id"] = row[cfg.COL_SUBJECT_ID]
                _instance["latent_path_name"] = row[cfg.COL_PREP_LATENT_PATH]

                for key in self.conditions_keys_ordered:
                    if key in row:
                        _instance[key] = row[key]
                    else:
                        raise ValueError(f"Condition key {key} not found in the dataset.")

                # attention mask
                if self.att_mask_resolution_list is not None and self.att_mask_structure_mapping is not None:
                    _instance["att_mask"] = self.create_att_mask_dict(row)

                instances.append(_instance)
        return instances




class PrepareTrainingDataset(Dataset):
    """PyTorch ``Dataset`` yielding precomputed latents, conditions, and (optionally) attention masks."""

    def __init__(self, 
                 training_dataset_path_name: str,
                 conditions_keys_ordered: list[str],
                 dataset_filters: dict | None = None,
                att_mask_resolution_list: list[tuple[int, int, int]] | None = None,
                att_mask_structure_mapping: dict | None = None,
                att_mask_weights: dict | None = None
                 ):
        """Load training instances via :class:`LoadPaths` and store dataset config.

        Args:
            training_dataset_path_name: Path to the training CSV.
            conditions_keys_ordered: Ordered list of condition column names.
            dataset_filters: Optional column-based row filters (see :class:`LoadPaths`).
            att_mask_resolution_list: Attention-mask resolutions to load, or ``None`` to disable attention-mask loading.
            att_mask_structure_mapping: Structure-to-channel mapping for attention masks (see :meth:`LoadPaths.create_att_mask_dict`).
            att_mask_weights: Optional per-structure scalar weights applied to flattened attention masks in :meth:`load_att_masks`.
        """

        # load data
        data_loader = LoadPaths(training_dataset_path_name, 
                                conditions_keys_ordered,
                                  dataset_filters=dataset_filters, 
                                  att_mask_resolution_list=att_mask_resolution_list, 
                                  att_mask_structure_mapping=att_mask_structure_mapping)
        
        self.train_data = data_loader.get_train_data(only_baseline=False)
        self.conditions_keys_ordered = conditions_keys_ordered

        logger.info(f"Number of training images: {len(self.train_data)}")

        # number of latent in the folder
        self.num_instances = len(self.train_data) 
        self._length = self.num_instances

        # attention mask resolution list
        self.att_mask_resolution_list = att_mask_resolution_list
        self.att_mask_structure_mapping = att_mask_structure_mapping
        self.att_mask_weights = att_mask_weights

    def load_att_masks(self, instance: dict) -> dict:
        """Load, combine, weight, and softmax-normalize the attention masks for one instance.

        For each resolution: loads (or synthesizes, for scalar channel
        values) each channel's mask, optionally combines multiple source
        structures per channel via element-wise max, fills in a
        "complement" channel (for entries flagged with a negative scalar
        value) as the sum of all other channels, applies optional
        per-structure weights, flattens each channel to a 1D vector, stacks
        channels, and applies a softmax across channels so the result sums
        to 1 per voxel.

        Args:
            instance: A training instance dict with an ``"att_mask"`` key
                as produced by :meth:`LoadPaths.create_att_mask_dict`.

        Returns:
            ``{resolution_str: tensor}`` where each tensor has shape
            ``(num_voxels, num_structures)`` and rows sum to 1 (softmax
            over the structure/channel dimension).
        """
        att_mask_dict = {}
        for att_mask_res, att_mask_res_dict in instance["att_mask"].items():
            att_mask_res_list = []
            _resolution = [int(x) for x in att_mask_res.split("_")]
            complete_mask=None
            for structure_name, att_mask_path_name in att_mask_res_dict.items():
                if isinstance(att_mask_path_name, int) or isinstance(att_mask_path_name, float):
                    if att_mask_path_name < 0:
                        if complete_mask is None:
                            complete_mask = np.zeros(_resolution, dtype=np.float32)
                        att_mask = None
                    else:
                        att_mask = np.zeros(_resolution, dtype=np.float32)
                        att_mask.fill(att_mask_path_name)
                elif isinstance(att_mask_path_name, list) or isinstance(att_mask_path_name, tuple):
                    att_mask = np.zeros(_resolution, dtype=np.float32)
                    for sub_att_map in att_mask_path_name:
                        att_mask = np.maximum(att_mask, np.load(sub_att_map))
                else:
                    att_mask = np.load(att_mask_path_name)

                att_mask_res_list.append(att_mask)

                if complete_mask is not None and att_mask is not None:
                    complete_mask += att_mask

            # add complete mask
            if complete_mask is not None:
                for i in range(len(att_mask_res_list)):
                    if att_mask_res_list[i] is None:
                        att_mask_res_list[i] = complete_mask
                        
            # flat and apply weights
            for i, (structure_name, att_mask_path_name) in enumerate(att_mask_res_dict.items()):
                att_mask_res_list[i] = att_mask_res_list[i].reshape(-1)
                if self.att_mask_weights is not None and structure_name in self.att_mask_weights:
                    att_mask_res_list[i] *= self.att_mask_weights[structure_name]

            att_mask_res_list = np.stack(att_mask_res_list, axis=0).T  # (HxWxD, num_structures)
            att_mask_res_tensor = torch.from_numpy(att_mask_res_list)

            att_mask_res_tensor = att_mask_res_tensor.softmax(dim=-1)
            att_mask_dict[att_mask_res] = att_mask_res_tensor
        return att_mask_dict


    def __len__(self) -> int:
        """Number of training instances."""
        return self._length

    def __getitem__(self, index: int) -> dict:
        """Fetch one training example: latent, id, conditions, and (optionally) attention masks.

        Args:
            index: Index into the dataset (wrapped modulo ``self.num_instances``).

        Returns:
            A dict with keys ``"latent"`` (squeezed latent tensor),
            ``"id"``, one entry per condition key (each a 1-element
            tensor), and (if configured) ``"att_mask"``.
        """
        # dictionary to store the image and the prompt
        example = {}
        # select latent path name from the list
        instance = self.train_data[index % self.num_instances]

        # load latent
        path_name_latent = instance["latent_path_name"]
        instance_latent = np.load(path_name_latent)


        # remove dimensions of size 1
        instance_latent = np.squeeze(instance_latent)

        # apply the transformations and save the image in the dictionary
        example["latent"] = torch.from_numpy(instance_latent)

        # obtain the age of the image
        example["id"] = instance["id"]

        # obtain the conditions
        for key in self.conditions_keys_ordered:
            example[key] = torch.tensor([instance[key]])

        # attention mask
        if "att_mask" in instance:
            example["att_mask"] = self.load_att_masks(instance)
        return example
    
def collate_fn(examples: list[dict], conditions_keys_ordered: list[str]) -> dict:
    """Batch a list of dataset examples into stacked tensors.

    Args:
        examples: List of per-example dicts as produced by
            :meth:`PrepareTrainingDataset.__getitem__`.
        conditions_keys_ordered: Ordered list of condition keys expected
            in every example.

    Returns:
        A dict with ``"id"`` (list), ``"latent"`` (stacked, contiguous,
        float tensor), one stacked tensor per condition key, and (if
        present in the first example) a stacked/contiguous/float
        ``"att_mask"`` dict.

    Raises:
        ValueError: If a condition key is missing from the first example.
    """
    res_dict = {}

    res_dict["id"] = [example["id"] for example in examples]

    latent = torch.stack([example["latent"] for example in examples])
    latent = latent.to(memory_format=torch.contiguous_format).float()
    res_dict["latent"] = latent


    for key in conditions_keys_ordered:
        if key in examples[0]:
            res_dict[key] = torch.stack([example[key] for example in examples])
        else:
            # If the key is not found, raise an error
            raise ValueError(f"Condition key {key} not found in the examples.")

    # attention mask
    if "att_mask" in examples[0]:
        att_mask = {}
        for key in examples[0]["att_mask"].keys():
            att_mask[key] = torch.stack([example["att_mask"][key] for example in examples])
            att_mask[key] = att_mask[key].to(memory_format=torch.contiguous_format).float()
        res_dict["att_mask"] = att_mask
    return res_dict




class MaxPerSubjectSampler(Sampler):
    """Epoch sampler that caps how many timepoints per subject are drawn each epoch.

    Prevents subjects with many longitudinal follow-ups from dominating
    training relative to subjects with few sessions.
    """

    def __init__(self, dataset: PrepareTrainingDataset, max_per_subject: int = 3, shuffle: bool = True, generator: torch.Generator | None = None):
        """
        Args:
            dataset: instancia de PrepareTrainingDataset (con atributo train_data que incluye 'id')
            max_per_subject: número máximo de muestras por sujeto por época
            shuffle: whether to shuffle indices within each subject and across the final epoch order.
            generator: optional ``torch.Generator`` for reproducible shuffling.
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
                    indices = [indices[i] for i in torch.randperm(len(indices), generator=self.generator)]
                else:
                    random.shuffle(indices)

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




def instantiate_dataset(training_dataset_path_name: str, 
                        conditions_keys_ordered: list[str], 
                        batch_size: int, 
                        gen_dataloader: torch.Generator, 
                        dataset_filters: dict | None = None, 
                        max_timepoints_per_epoch: int = 3,
                        att_mask_resolution_list: list[tuple[int, int, int]] | None = None, 
                        att_mask_structure_mapping: dict | None = None, 
                        att_mask_weights: dict | None = None) -> torch.utils.data.DataLoader:
    """Build the training ``DataLoader`` with subject-capped sampling.

    Args:
        training_dataset_path_name: Path to the training CSV.
        conditions_keys_ordered: Ordered list of condition column names.
        batch_size: Batch size for the DataLoader.
        gen_dataloader: RNG used by :class:`MaxPerSubjectSampler` for reproducible shuffling.
        dataset_filters: Optional column-based row filters.
        max_timepoints_per_epoch: Passed to :class:`MaxPerSubjectSampler` as ``max_per_subject``.
        att_mask_resolution_list: Attention-mask resolutions to load, or ``None`` to disable.
        att_mask_structure_mapping: Structure-to-channel mapping for attention masks.
        att_mask_weights: Optional per-structure attention-mask weights.

    Returns:
        A configured ``torch.utils.data.DataLoader`` (8 workers, persistent workers enabled).
    """
    # ---- Data set creation
    train_dataset = PrepareTrainingDataset(
        training_dataset_path_name=training_dataset_path_name,
        conditions_keys_ordered=conditions_keys_ordered,
        dataset_filters=dataset_filters,

        att_mask_resolution_list=att_mask_resolution_list,
        att_mask_structure_mapping=att_mask_structure_mapping,
        att_mask_weights=att_mask_weights,
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


def conditions_to_name(conditions_dict: dict) -> str:
    """
    Convert conditions dictionary to a string name.

    Args:
        conditions_dict: Mapping of condition name to value. Float values
            are formatted with 2 decimal places.

    Returns:
        An underscore-joined ``key_value_key_value...`` string with no
        trailing underscore.
    """
    name = ""
    for key, value in conditions_dict.items():
        if isinstance(value, float):
            value = f"{value:.2f}"
        name += f"{key}_{value}_"
    return name.rstrip('_')


def get_img_attention_maps(att_maps: np.ndarray, prompt_lits: list[str]) -> np.ndarray:
    """Render a side-by-side strip of 2D attention-map slices, one per condition.

    Args:
        att_maps: Attention maps array of shape ``(X, Y, Z, num_prompts)``.
        prompt_lits: Condition/prompt names, one per channel of ``att_maps``
            (used as text labels).

    Returns:
        A single 2D ``uint8`` RGB image, the horizontal concatenation of
        each channel's 3-view slice montage, labeled with its condition name.
    """
    img_list = []
    att_maps = data_normalization.normalize_image(att_maps, strictly_positive=False)
    att_maps = (att_maps * 255).astype(np.uint8)
    for i, p in enumerate(prompt_lits):
        img = att_maps[:, :, :, i]
        img_2D = fc.cat_3_views([img], layer_offset=None, axis=0, img_cropping=0, to_rgb=True)[0]
        img_2D = fc.resize_image(img_2D, value=128)
        img_2D = fc.text_over_image(img_2D, text=p, font_scale=0.3, font_thickness=1, margin_ratio=0.2)
        img_list.append(img_2D)
    # concatenate the images
    complete_img = np.concatenate(img_list, axis=1)
    return complete_img






def find_closest_rows(dataset_df: pd.DataFrame, asked_conditions_list: list[dict]) -> pd.DataFrame:
    """
    dataset_df: DataFrame con tus sujetos
    asked_conditions_list: lista de diccionarios con valores objetivo
    condition_columns: lista de columnas a considerar en la distancia

    Args:
        dataset_df: Candidate rows to search over.
        asked_conditions_list: List of target condition dicts; the columns
            of the first dict determine which columns are used for the
            distance computation.

    Returns:
        A DataFrame with one row per entry in ``asked_conditions_list``:
        the row of ``dataset_df`` whose condition columns are closest
        (Euclidean distance) to that target.
    """
    closest_rows = []
    condition_columns = list(asked_conditions_list[0].keys())

    # Convertir dataset a numpy para eficiencia
    X = dataset_df[condition_columns].to_numpy()

    for cond_dict in asked_conditions_list:
        # convertir dict a array
        target = np.array([cond_dict[col] for col in condition_columns])

        # calcular distancias Euclidianas
        distances = np.linalg.norm(X - target, axis=1)

        # encontrar índice del más cercano
        closest_idx = distances.argmin()
        closest_rows.append(dataset_df.iloc[closest_idx])

    # devolver DataFrame con los más cercanos
    return pd.DataFrame(closest_rows)



def _compute_single_metrics(args_tuple: tuple) -> tuple[list[float], list[float], list[float]]:
    """Compute per-structure MAE/normalized-MAE/volume for a single generated segmentation.

    Worker function intended for use with ``multiprocessing.Pool.imap``
    (see :func:`compute_metrics`).

    Args:
        args_tuple: ``(seg, conditions, data_normalizer)`` where ``seg``
            is a generated-and-segmented volume, ``conditions`` is the
            ``{structure_name: target_value}`` dict used to generate it,
            and ``data_normalizer`` is the fitted volume normalizer.

    Returns:
        ``(mae_norm_list, mae_list, vol_list)``, one entry per structure
        in ``conditions``, in iteration order.
    """
    seg, conditions, data_normalizer = args_tuple
    mae_list = []
    mae_norm_list = []
    vol_list = []
    for s_name, s_value in conditions.items():
        seg_mask = ufs.merge_seg96_to_mask(seg, cfg.STRUCTURE_INDEX_VOL_DICT[s_name])
        icv_vol = np.sum(seg > 0)

        vol = np.sum(seg_mask)
        mult = 1
        if s_name != "total_vol":
            vol /= icv_vol 
            mult = 100
 
        mae_list.append(np.mean(np.abs(vol - data_normalizer.inverse_transform_single(s_value, s_name))*mult))
        mae_norm_list.append(np.mean(np.abs(data_normalizer.transform_single(vol, s_name) - s_value))) 
        vol_list.append(vol)
    return mae_norm_list, mae_list, vol_list


def compute_metrics(seg_list: list[np.ndarray], args: argparse.Namespace, data_normalizer, n_processes: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-seed/condition volume-error metrics for a batch of generated+segmented images.

    Args:
        seg_list: Segmented generated images, ordered as
            ``for seed in args.val_seeds: for condition in args.val_conditions_list: ...``
            (matching the generation order in :func:`validation`).
        args: Training args, must include ``val_seeds``, ``val_conditions_list``, ``conditions_keys_ordered``.
        data_normalizer: Fitted volume normalizer.
        n_processes: Number of worker processes for the metrics computation.

    Returns:
        ``(df_error_norm, df_error, df_vol)`` DataFrames, each with
        columns ``["seed", "condition"] + args.conditions_keys_ordered``
        and one row per (seed, condition) pair.

    Side Effects:
        Spawns a ``multiprocessing.Pool`` of ``n_processes`` workers.
    """
    # Prepara la lista de tareas: cada tarea es (seg, conditions, data_normalizer)
    tasks = []
    c = 0
    for seed in args.val_seeds:
        for i, conditions in enumerate(args.val_conditions_list):
            seg = seg_list[c]   # mismo orden secuencial
            tasks.append((seg, conditions, data_normalizer))
            c += 1

    # Ejecuta en paralelo con progress bar
    results = []
    with mp.Pool(processes=n_processes) as pool:
        for r in tqdm(pool.imap(_compute_single_metrics, tasks), total=len(tasks), desc="Computing metrics for val segs"):
            results.append(r)

    mae_norm_list = [r[0] for r in results]
    mae_list = [r[1] for r in results]
    vol_list = [r[2] for r in results]


    # Reconstruye la tabla en el mismo orden
    error_norm_table = np.zeros((len(mae_norm_list), 2 + len(args.conditions_keys_ordered)))
    error_table = np.zeros((len(mae_list), 2 + len(args.conditions_keys_ordered)))
    vol_table = np.zeros((len(vol_list), 2 + len(args.conditions_keys_ordered)))
    c = 0
    for seed in args.val_seeds:
        for i in range(len(args.val_conditions_list)):
            error_norm_table[c] = [seed, i] + mae_norm_list[c]
            error_table[c] = [seed, i] + mae_list[c]
            vol_table[c] = [seed, i] + vol_list[c]
            c += 1

    df_error_norm = pd.DataFrame(error_norm_table, columns=["seed", "condition"] + args.conditions_keys_ordered)
    df_error = pd.DataFrame(error_table, columns=["seed", "condition"] + args.conditions_keys_ordered)
    df_vol = pd.DataFrame(vol_table, columns=["seed", "condition"] + args.conditions_keys_ordered)
    return df_error_norm, df_error, df_vol


def _segment_single_image(path_name_img: str, output_path_segmentations: str) -> np.ndarray:
    """Segment one generated image with SynthSeg and load the resulting label map.

    Worker function intended for use with ``multiprocessing.Pool.imap``
    (see :func:`segment_created_images`).

    Args:
        path_name_img: Path to the generated image (expected to contain
            ``"img"`` in its filename, replaced with ``"seg"`` for the
            output path).
        output_path_segmentations: Directory to write the segmentation to.

    Returns:
        The loaded segmentation array.

    Side Effects:
        Runs SynthSeg (via
        :func:`src.utils.prep_segmentation.save_synthseg_segmentation`)
        and writes the resulting segmentation file.
    """
    path_name_seg = os.path.join(
        output_path_segmentations,
        os.path.basename(path_name_img).replace("img", "seg"),
    )
    os.makedirs(os.path.dirname(path_name_seg), exist_ok=True)

    # Aplica la segmentación
    prep_segmentation.save_synthseg_segmentation(
        img_path_name=path_name_img,
        out_path_name=path_name_seg,
        verify=True,
        verbose=False,
        robust=True,
        cortical_parcelation=False,
    )

    # Carga el resultado
    seg, _ = nfc.load_nifti(path_name_seg)
    return seg


def segment_created_images(path_name_imgs_list: list[str], args: argparse.Namespace, output_path_segmentations: str, n_processes: int = 8) -> list[np.ndarray]:
    """Segment a batch of generated validation images in parallel.

    Args:
        path_name_imgs_list: Paths to the generated images, ordered as
            ``for seed in args.val_seeds: for condition in args.val_conditions_list: ...``.
        args: Training args, must include ``val_seeds`` and ``val_conditions_list``.
        output_path_segmentations: Directory to write segmentations to.
        n_processes: Number of worker processes.

    Returns:
        List of segmentation arrays, in the same order as the (implied)
        task list.

    Side Effects:
        Spawns a ``multiprocessing.Pool`` of ``n_processes`` workers;
        writes segmentation files under ``output_path_segmentations``.
    """
    # Genera la lista de imágenes que hay que segmentar según seeds y condiciones
    tasks = []
    c = 0
    for seed in args.val_seeds:
        for _ in range(len(args.val_conditions_list)):
            tasks.append(path_name_imgs_list[c])
            c += 1
    # Ejecuta en paralelo con progress bar
    seg_list = []
    _segment = partial(_segment_single_image, output_path_segmentations=output_path_segmentations)

    with mp.Pool(processes=n_processes) as pool:
        for seg in tqdm(pool.imap(_segment, tasks), total=len(tasks), desc="Segmenting val imgs"):
            seg_list.append(seg)

    return seg_list



@torch.no_grad()
def validation(
    unet: torch.nn.Module,
    conditions_model: torch.nn.Module,
    noise_scheduler,
    autoencoder,
    latents_shape: torch.Size,
    step: int,
    args: argparse.Namespace,
    att_controller=None,
    data_normalizer=None,
    evaluate: bool = False,
) -> tuple[dict, dict] | None:
    """Generate validation images and (optionally) evaluate ROI-volume accuracy.

    For each (seed, condition) pair: samples fixed initial noise, runs the
    reverse diffusion process (with classifier-free guidance if
    ``args.free_guidance_ratio > 0``), decodes the resulting latents to an
    image, and saves it. Optionally saves cross-attention maps between the
    UNet and the ROI-volume conditions. If ``evaluate=True``, additionally
    segments each generated image and computes per-structure volume
    accuracy metrics against the requested conditions.

    Args:
        unet: The (currently training) diffusion UNet.
        conditions_model: Model embedding the ROI-volume conditions.
        noise_scheduler: DDIM-style noise scheduler (already configured
            with inference timesteps).
        autoencoder: Latent-space decoder used to turn generated latents
            into images.
        latents_shape: Shape of a single training-batch latent, used to
            size the single-sample generation latents.
        step: Current global training step (used for output paths/naming).
        args: Training args; must include (at least) ``output_path``,
            ``val_imgs_dir_name``, ``val_seeds``, ``val_conditions_list``,
            ``free_guidance_ratio``, ``conditions_keys_ordered``,
            ``ref_aff``.
        att_controller: Optional attention-map recorder; if provided, a
            labeled attention-map image is saved per (seed, condition).
        data_normalizer: Fitted volume normalizer, required when
            ``evaluate=True``.
        evaluate: If True, run full evaluation (segment generated images
            and compute volume-accuracy metrics) in addition to image
            generation. If False, only a quick single-condition
            generation pass is run (for lightweight periodic previews).

    Returns:
        If ``evaluate=True`` and generation completed successfully:
        ``(val_results_norm, val_results)``, each a dict mapping condition
        key -> mean metric value across all (seed, condition) pairs.
        Otherwise ``None``.

    Side Effects:
        Writes generated images, 2D preview montages, (optionally)
        attention-map images, segmentations, and metrics CSVs under
        ``{args.output_path}/{args.val_imgs_dir_name}/step_{step}/``.
    """

    logger.info(f"Validation step {step}, fgr: {args.free_guidance_ratio}...")

    used_conditions = args.val_conditions_list
    if not evaluate:
        logger.info("Just a quick validation with the first conditions in the training set...")
        used_conditions = [args.val_conditions_list[0]]

    # output_path_img = os.path.join(args.output_path, args.val_imgs_dir_name, f"step_{step}", "images")
    output_path_step = os.path.join(args.output_path, args.val_imgs_dir_name, f"step_{step}")
    output_path_att_maps = os.path.join(output_path_step, "att_maps")
    output_path_2D_images = os.path.join(output_path_step, "images2d")
    output_path_images = os.path.join(output_path_step, "images")
    output_path_segmentations = os.path.join(output_path_step, "segmentations")
    output_path_metrics = os.path.join(output_path_step, "metrics")

    for _output_path in [output_path_att_maps, output_path_2D_images, output_path_images, output_path_segmentations, output_path_metrics]:
        os.makedirs(_output_path, exist_ok=True)

    imgs_list = []
    path_name_imgs_list = []
    js_table = np.zeros((len(args.val_seeds)*len(used_conditions), 3))  # seed, condition, js
    total_images = len(args.val_seeds) * len(used_conditions)
    model_ready_for_evaluation = True
    c = 0
    for seed in args.val_seeds:
        seed_img_list = []
        for i, conditions in enumerate(used_conditions):
            # instantiate every time to generate using the same initial noise (using CPU generator)
            _l_shape = [1, latents_shape[-4], latents_shape[-3], latents_shape[-2], latents_shape[-1]]
            gen_randn = torch.Generator().manual_seed(seed) 
            latents = torch.randn(_l_shape, generator=gen_randn).half().to(device)

            # preparing conditioning
            conditions_list = [[conditions[key]] for key in args.conditions_keys_ordered]
            conditioning = torch.tensor([conditions_list]).to(device).float()#.unsqueeze(1).permute(0,2,1)
            conditioning_emb = conditions_model(conditioning)

            if args.free_guidance_ratio > 0:
                # for free guidance, we need to create a null conditioning
                null_conditioning = torch.zeros_like(conditioning_emb)
                conditioning_emb = torch.cat([null_conditioning, conditioning_emb], dim=0)


            all_timesteps = noise_scheduler.timesteps
            all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
            progress_bar = tqdm(
                zip(all_timesteps, all_next_timesteps),
                total=min(len(all_timesteps), len(all_next_timesteps)),
                desc=f"Step {step} generating val imgs {c+1}/{total_images}"
            )
            with torch.no_grad(), torch.amp.autocast("cuda"):
                # synthesize latents
                for t, next_t in progress_bar:
                    if args.free_guidance_ratio > 0:
                        latents = torch.cat([latents] * 2)

                    model_output = unet(
                        x=latents,
                        timesteps=torch.Tensor((t,)).to(device),
                        context=conditioning_emb,
                    )

 
                    if args.free_guidance_ratio > 0:
                        # combine the predictions
                        model_output_pred_uncond, model_output_pred_cond = model_output.chunk(2)
                        model_output = model_output_pred_uncond + args.free_guidance_ratio * (model_output_pred_cond - model_output_pred_uncond)
                        latents = latents[:1]

                        latents, _ = noise_scheduler.step(model_output, t, latents)

                # do the attention maps before the free memory
                if att_controller is not None:
                    att_map_resolution = args.att_mask_resolution_list[0]
                    attention_maps = attention_controller.aggregate_attention(att_controller, att_map_resolution, ["down", "up"], is_cross=True, select=0, nb_prompts=1)
                    img_att_maps = get_img_attention_maps(attention_maps.numpy(), args.conditions_keys_ordered)

                    img_att_maps = Image.fromarray(img_att_maps)
                    name_att_map = f"att_maps_step_{step}_seed_{seed}_cond_{i}"
                    img_att_maps.save(f"{output_path_att_maps}/{name_att_map}.png")

                    att_controller.reset()

                # free memory for the autoencoder
                del model_output
                if args.free_guidance_ratio > 0:
                    del model_output_pred_uncond, model_output_pred_cond
                torch.cuda.empty_cache()
                
                # decode the latents to images
                synthetic_images = autoencoder.decode(latents)
                synthetic_images = torch.clip(synthetic_images, 0.0, 1.0)
                synthetic_images = synthetic_images.squeeze().cpu().numpy().astype(np.float32)

                # decode the latents to images
                path_name_img = os.path.join(output_path_images, f"img_step_{step}_seed_{seed}_cond_{i}.nii.gz")
                path_name_imgs_list.append(path_name_img)
                nfc.save_nifti(synthetic_images, args.ref_aff, path_name_img)
            
            seed_img_list.append(synthetic_images)

            c += 1
            
            if not model_ready_for_evaluation:
                break


        imgs_list.extend(seed_img_list)

        # ---- save 2D images for visualization
        # # obtain 3 layers of the images
        imgs_list_2D = fc.cat_n_views_different_layers(seed_img_list, 
                                                    view_layersoffset_list=[(2, 0), (2, -15), (1, 0), (0, 10)], 
                                                    axis=0, 
                                                    img_cropping=50,
                                                    to_rgb=True)
        # save synthetic images
        complete_img = np.concatenate(imgs_list_2D, axis=1)
        complete_img = Image.fromarray(complete_img)
        complete_img.save(f"{output_path_2D_images}/imgs2D_step_{step}_seed_{seed}.png" )


        if not model_ready_for_evaluation:
            break

    # save images
    if model_ready_for_evaluation and evaluate:

        seg_list = segment_created_images(path_name_imgs_list, args, output_path_segmentations)
        df_error_norm, df_error, df_vol = compute_metrics(seg_list, args, data_normalizer)

        # save validation results
        path_name_val_mae_norm = os.path.join(output_path_metrics, f"mae_norm_step_{step}.csv")
        path_name_val_mae = os.path.join(output_path_metrics, f"mae_step_{step}.csv")
        path_name_val_vols = os.path.join(output_path_metrics, f"vols_step_{step}.csv")

        df_error_norm.to_csv(path_name_val_mae_norm, index=False)
        df_error.to_csv(path_name_val_mae, index=False)
        df_vol.to_csv(path_name_val_vols, index=False)

        val_results_norm = df_error_norm[args.conditions_keys_ordered].mean().to_dict()
        val_results = df_error[args.conditions_keys_ordered].mean().to_dict()
        return val_results_norm, val_results

    return None
    


def save_model(unet: torch.nn.Module, conditions_model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler, global_step: int, out_model_path: str, ema: EMA | None = None, best: bool = False) -> None:  # MOD: se añade parámetro ema
    """Save a training checkpoint (UNet, conditions model, optimizer, scheduler, EMA).

    Args:
        unet: The diffusion UNet (state dict saved via ``.module`` if
            wrapped in ``DistributedDataParallel``).
        conditions_model: Model embedding the ROI-volume conditions.
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
        "optimizer_state_dict": optimizer.state_dict(),
        "num_train_timesteps": global_step,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "conditions_model_state_dict": conditions_model.state_dict(),
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



def save_configurations(args_train: argparse.Namespace, networks_config: argparse.Namespace, config_path: str, config_name: str = "training_config.json") -> None:
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
    """Run the full BrainST-img training loop.

    Instantiates models, dataset, optimizer/scheduler, optionally resumes
    from a checkpoint or loads pretrained weights, then iterates over
    epochs/batches performing the diffusion denoising-loss training step
    (plus an optional attention-map supervision loss), with gradient
    accumulation, periodic checkpointing, periodic validation (with EMA
    swap-in/out), and early stopping on validation loss plateau.

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
    gen_cond_noise = torch.Generator().manual_seed(args_train.seed)

    # ---- instantiate models
    networks_config = fc.dict_to_args(cfg.ARCHITECTURE_BRAINST_IMG, deep_conversion=True)
    num_conditions = len(args_train.conditions_keys_ordered)    
    models_dict = instantiate_models.instantiate_conditioned_models(networks_config, cfg.PATH_AUTOENCODER_CHK, cfg.DEVICE, args_train.val_num_diffusion_steps)
    
    unet = models_dict["unet"]
    conditions_model = models_dict["conditions_model"]
    noise_scheduler = models_dict["noise_scheduler"]
    autoencoder = models_dict["autoencoder"]
    
    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        training_dataset_path_name=args_train.training_dataset_path_name,
        conditions_keys_ordered=args_train.conditions_keys_ordered,
        batch_size=args_train.batch_size,
        gen_dataloader=gen_dataloader,
        dataset_filters=args_train.dataset_filters,
        max_timepoints_per_epoch=args_train.max_timepoints_per_epoch,

        att_mask_resolution_list=args_train.att_mask_resolution_list,
        att_mask_structure_mapping=fc.args_to_dict(args_train.att_mask_structure_mapping),
        att_mask_weights=fc.args_to_dict(args_train.att_mask_weights) if args_train.att_mask_weights is not None else None,
    )

    # ---- create folders
    os.makedirs(args_train.output_path, exist_ok=True)
    _checkpoint_dir_name =  os.path.join(args_train.output_path, args_train.checkpoints_dir_name)
    _logs_dir_name = os.path.join(args_train.output_path, args_train.logs_dir_name)
    _val_imgs_dir_name = os.path.join(args_train.output_path, args_train.val_imgs_dir_name)
    os.makedirs(_checkpoint_dir_name, exist_ok=True)
    os.makedirs(_logs_dir_name, exist_ok=True)
    os.makedirs(_val_imgs_dir_name, exist_ok=True)

    # ---- obtain the val conditions
    train_df = pd.read_csv(args_train.training_dataset_path_name)
    train_df = train_df[train_df[cfg.COL_SPLIT] == "train"]
    val_conditions_list_df = find_closest_rows(train_df, args_train.val_expected_conditions_list)
    args_train.val_conditions_list = val_conditions_list_df[args_train.conditions_keys_ordered].to_dict(orient="records")
    args_train.val_conditions_metadata_list =  val_conditions_list_df[[cfg.COL_SUBJECT_ID, cfg.COL_SESSION_ID, cfg.COL_AGE, cfg.COL_SPLIT, cfg.COL_DX]].to_dict(orient="records")

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

    # ---- early stopping
    _, ref_aff = nfc.load_nifti(args_train.path_name_ref_img)
    data_normalizer = data_normalization.ZScoreStandardizerBrainStructures(args_train.conditions_keys_ordered)
    data_normalizer.load_params(args_train.normalizer_params)
    args_train.ref_aff = ref_aff

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

    # Attention controller
    att_controller = None
    if ((args_train.att_mask_resolution_list is not None) or
        (args_train.save_val_att_masks)):
        resolutions_list = [np.prod(res) for res in args_train.att_mask_resolution_list]
        att_controller = attention_controller.AttentionStore(resolutions_list = resolutions_list)
        attention_controller.register_attention_control(unet, att_controller, register_self=False)

    for epoch in range(first_epoch, max_epochs):
        for batch in train_dataloader:

            # prepare inputs
            latents = batch["latent"].to(device)

            cond_list = [batch[key] for key in args_train.conditions_keys_ordered]  # list of (B, n_conditions, 1)
            conditioning = torch.stack(cond_list, dim=1).float() # (B, num_conditions, 1)

            if args_train.condition_noise_std > 0:
                cond_noise = torch.empty_like(conditioning).uniform_(-args_train.condition_noise_std, args_train.condition_noise_std, generator=gen_cond_noise)
                conditioning += cond_noise
            conditioning = conditioning.to(device).float() # (B, num_conditions, 1)

            # Forward pass
            with autocast("cuda", enabled=args_train.amp):
                # generate noise and timesteps with dedicate generatos and in the cpu for reproducibility
                noise = torch.randn(latents.shape, device="cpu", generator=gen_noise).to(device)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device="cpu", generator=gen_t).long().to(device)
                noisy_latent = noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)

                conditioning_emb = conditions_model(conditioning)

                # free guidance
                free_guidance_prob = torch.rand(1, generator=gen_free_guidance).item()
                if free_guidance_prob < args_train.free_guidance_threshold:
                    conditioning_emb = torch.zeros_like(conditioning_emb)

                model_output = unet(noisy_latent, 
                                  timesteps=timesteps,
                                  context = conditioning_emb,
                                    )
                

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    # predict noise
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    # predict sample
                    model_gt = latents
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    # predict velocity
                    model_gt = latents - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from ",
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                    )

                loss_noise = loss_pt(model_output.float(), model_gt.float())   # Dividir para escalar la pérdida

                # attention mask loss
                loss_att_maps = 0
                if (free_guidance_prob >= args_train.free_guidance_threshold) and (att_controller is not None) and ('att_mask' in batch):

                    for att_res in args_train.att_mask_resolution_list:
                        att_res_prod = np.prod(att_res)
                        att_maps_res_list = []
                        # obtain all the attention maps for the current resolution
                        for key_place_in_unet in att_controller.step_store:
                            if att_res_prod in att_controller.step_store[key_place_in_unet]:
                                att_maps_res_list.extend(att_controller.step_store[key_place_in_unet][att_res_prod])
                        
                        if len(att_maps_res_list) > 0:
                            att_maps_mean = torch.mean(torch.stack(att_maps_res_list, dim=0), dim=0)
                            att_res_name = "_".join([str(x) for x in att_res])
                            att_maps_gt = batch['att_mask'][att_res_name].to(device).unsqueeze(1) # add the place for the heads
                            att_maps_gt = att_maps_gt.expand(-1, att_maps_mean.shape[1], -1, -1) # duplicate the heads

                            loss_att_maps += loss_pt(att_maps_mean.float(), att_maps_gt.float())
                        else:
                            raise ValueError(f"No attention maps found for resolution {att_res}")

            if att_controller is not None:
                att_controller.reset()  # reset attention controller for each batch

            loss = loss_noise + args_train.loss_weights.att_mask * loss_att_maps
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
                if loss_att_maps > 0:
                    writer.add_scalar("Loss/train_noise", loss_noise.item(), global_step)
                    writer.add_scalar("Loss/train_att_maps", loss_att_maps.item(), global_step)
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("Time_steps", timesteps[0], global_step)

                # update progress bar 
                progress_bar.update(1)

                logs = {"loss": loss.detach().item(), 
                        "id": [f'{__id[:4]}' for __id in batch["id"]], 
                        }
                
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # save the model in intervals
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema)

                # Generar imágenes en intervalos
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    # Pause the training progress bar
                    progress_bar.set_description("Validating")
                    progress_bar.clear()
                    val_start = time.time()

                    # prepare models for validation
                    unet.eval()
                    conditions_model.eval()

                    if att_controller is not None:
                        att_controller.eval()
                        att_controller.reset()

                    if args_train.use_ema:
                        ema.apply_shadow()

                    # run validation
                    try:
                        val_results = validation(
                            unet=unet,
                            conditions_model=conditions_model,
                            noise_scheduler=noise_scheduler,
                            autoencoder=autoencoder,
                            latents_shape=latents.shape,
                            step=global_step,
                            args=args_train,
                            att_controller=att_controller,
                            data_normalizer=data_normalizer,
                            evaluate=global_step >= args_train.val_evaluation_initial_step,
                        )

                        if not args_train.initial_val and val_results is not None:
                            loss_val = np.mean(list(val_results[0].values()))
                            writer.add_scalar("Loss/val", loss_val, global_step)

                            # remove "total_vol" from val_results[1]
                            val_results[1].pop("total_vol", None)
                            writer.add_scalar("Loss/val_not_norm", np.mean(list(val_results[1].values())), global_step)
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
                        logger.error(f"ERROR DURING VALIDATION STEP {global_step}: {e}")

                    # recover training state
                    unet.train()
                    conditions_model.train()
                    if att_controller is not None:
                        att_controller.train()
                        att_controller.reset()

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
    "output_path": os.path.join(cfg.PATH_MODELS_TRAINING, "brainst_img"),
    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",
        
    # data
    "training_dataset_path_name": os.path.join(cfg.PATH_DATA_TRAINING, "example", "training_data.csv"),
    "normalizer_params": os.path.join(cfg.PATH_DATA_TRAINING, "example", "normalized_params.json"),
    "path_name_ref_img": os.path.join(cfg.PATH_DATA_TRAINING, "images", "preprocessed", "ADNI", "ADNI002S0295", "m000", "sub-ADNI002S0295_ses-m000_run-01_T1w_synthsr.nii.gz"),


    "conditions_keys_ordered": cfg.STRUCTURE_NAME_LIST_VOL,
    "dataset_filters": None, # add any filter in the from of a dictionary, for example {"age": [20, 25, 30]} to filter the dataset to only include subjects with age 20, 25, or 30. If None, no filtering is applied.

    # overfitting
    "condition_noise_std": 0.01, # added noise to the conditions during training for regularization and to avoid overfitting. 
    "max_timepoints_per_epoch": 3, # max number of timepoints for a given subject to be used in each epoch. If None, all timepoints are used. This is useful for datasets with many timepoints per subject to avoid overfitting to a specific subject.

    # training configuration
    "max_train_steps": 500000, # number of training steps
    "save_checkpoint_interval": 50000, # save the model every n steps

    # ---- memory reduction
    "amp": True, 

    # ---- Training stability
    "batch_size": 3,
    "gradient_accumulation_steps": 4,
    "use_ema": True,
    "ema_params": {
        "decay": 0.999,
        "warm_up_steps": 2000,
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

    # ---- loss
    "loss_weights": {
        "noise": 1.0, # default
        "att_mask": 0.5,  # weight for the attention mask loss
    },


    # ---- Free guidance
    "free_guidance_ratio": 3.0,
    "free_guidance_threshold": 0.2,

    # ---- attention mask loss
    "att_mask_resolution_list":[(48,64,48), (24,32,24),(12,16,12), (6,8,6)],  # resolution of the attention mask to use in the loss
    "att_mask_structure_mapping": {
        "total_vol": "total", 
        "surrounding_csf_vol": "surrounding_csf", 
        "cortical_gm_vol": "cortical_gm",
        "cerebral_wm_vol": "cerebral_wm",
        "lateral_ventricles_vol": "lateral_ventricles", # inferior lateral ventricles already included in the mask
        # "lateral_ventricles_vol": ("lateral_ventricles", "inferior_lateral_ventricles"),
        "third_ventricle_vol": "third_ventricle",
        "fourth_ventricle_vol": "fourth_ventricle",
        "thalamus_vol": "thalamus",
        "hippocampus_vol": "hippocampus",
        "amygdala_vol": "amygdala",
        "putamen_vol": "putamen",
        "pallidum_vol": "pallidum",
        "caudate_vol": "caudate",
        "accumbens_area_vol": "accumbens",
        "ventral_dc_vol": "ventral_dc",
        "cerebellum_gm_vol": "cerebellum_gm",
        "cerebellum_wm_vol": "cerebellum_wm",
        "brainstem_vol": "brainstem",
    },


    "att_mask_weights": {
        "total_vol": 1.0,
        "surrounding_csf_vol": 4.5,
        "cortical_gm_vol": 4.0,
        "cerebral_wm_vol": 4.0,
        "lateral_ventricles_vol": 4.0,
        "third_ventricle_vol": 4.0,
        "fourth_ventricle_vol": 4.0,
        "thalamus_vol": 4.0,
        "hippocampus_vol": 4.0,
        "amygdala_vol": 4.0,
        "putamen_vol": 4.0,
        "pallidum_vol": 4.0,
        "caudate_vol": 4.0,
        "accumbens_area_vol": 4.0,
        "ventral_dc_vol": 4.0,
        "cerebellum_gm_vol": 4.0,
        "cerebellum_wm_vol": 4.0,
        "brainstem_vol": 4.0,
    },


    # ---- validation
    "val_interval": 2500, # validate every n steps
    "val_evaluation_initial_step": 20000, # start evaluating the validation metrics after this step
    "initial_val": True, # if true, it will run a validation at the beginning of the training (step 1)
    "val_seeds": [42,10], # seeds for the validation random number generator to generate the noise for the latents
    "val_num_diffusion_steps": 30, # number of diffusion steps for the validation
    "val_expected_conditions_list" : [ # desired conditions for the validation set. The closest rows in the dataset will be used for validation.
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # young
        {"total_vol": -3.0, "surrounding_csf_vol": 3.0, "cortical_gm_vol": -3.0, "cerebral_wm_vol": -3.0, "lateral_ventricles_vol": 3.0, "fourth_ventricle_vol": 3.0, "cerebellum_gm_vol": -3.0, "cerebellum_wm_vol": -3.0, "brainstem_vol": -3.0}, # old 
        
        {"total_vol": -3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": 3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": -3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": -3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": 3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": 3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": -3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": -3.0, "brainstem_vol": 3.0}, # old
        {"total_vol": 3.0, "surrounding_csf_vol": -3.0, "cortical_gm_vol": 3.0, "cerebral_wm_vol": 3.0, "lateral_ventricles_vol": -3.0, "fourth_ventricle_vol": -3.0, "cerebellum_gm_vol": 3.0, "cerebellum_wm_vol": 3.0, "brainstem_vol": -3.0}, # old
    ],
    "save_val_att_masks": True,  # if True, it will save the attention masks during validation (This takes more time)

    # ---- early stopping
    "patience": 100,

}

args_train = fc.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
)

# 23668MiB