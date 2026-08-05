"""Core generation pipelines for the BrainST framework.

This module wires together two diffusion models:

- **BrainST-img**: a diffusion model over latent brain-image
  representations, conditioned on an ROI-volume profile. It turns a
  volumetric profile into an actual 3D image.
- **BrainST-vol**: a diffusion model over ROI (region-of-interest) brain
  volumes, conditioned on covariates (age / sex / diagnosis). It predicts
  a *volumetric profile* of the brain (one value per ROI).

Three pipelines are exposed, mirroring the ``--generation_type`` values in
``main_generation.py``:

- :func:`brainst_synthesis`      -- generate a brand-new brain image.
- :func:`brainst_transformation` -- morph an existing image toward a
  target ROI-volume profile / segmentation.
- :func:`brainst_longitudinal`   -- age/progress an existing image toward
  a target covariate set (age, diagnosis), using BrainST-vol to first
  predict how the ROI volumes themselves would change.

Naming conventions used throughout this file:
    - ``*_dict``            : a ``{roi_name: value}`` or ``{covariate: value}`` mapping.
    - ``*_vec`` / ``*_arr``  : a plain numpy array (no ROI/covariate names attached).
    - ``initial_*``          : describes the *source* / starting-point brain.
    - ``target_*``           : describes the desired / destination brain.
    - ``noisy_latents``      : the (partially) noised latent tensor fed into a
      diffusion loop -- used for both the volume and image diffusion models.
    - ``reconstructed_*``    : the denoised output of a diffusion loop.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import configs as cfg
import src.brainst_img.instantiate_models as instantiate_brainst_img
import src.brainst_vol.instantiate_models as instantiate_brainst_vol
import src.brainst_vol.null_inversion as null_inversion_volumes
import src.utils.functions as fc
from src.brainst_img import generate_image, null_inversion, utils_generation
from src.brainst_vol import generate_volumes
from src.utils import data_normalization, prep_volumes

logger = logging.getLogger(__name__)


def get_volumes_from_segmentation(seg: np.ndarray, normalizer) -> dict:
    """Compute a normalized ROI-volumes dict directly from a segmentation map.

    Pipeline: raw voxel counts per ROI -> percentage of intracranial
    volume (ICV) -> standardized (z-scored) scale used by the models.

    Args:
        seg: Segmentation label map (one integer label per anatomical structure).
        normalizer: Fitted normalizer used for the final standardization step.

    Returns:
        ``{roi_name: standardized_volume}`` for every ROI in
        ``cfg.STRUCTURE_NAME_LIST_VOL``.
    """
    roi_volumes_dict = prep_volumes.get_volumes(seg, cfg.STRUCTURE_INDEX_VOL_DICT)
    roi_volumes_dict = data_normalization.normalize_by_icv(
        pd.DataFrame([roi_volumes_dict]),
        structure_names=cfg.STRUCTURE_NAME_LIST_VOL,
        icv_column="total_vol",
        percentage=False,
    ).iloc[0].to_dict()
    roi_volumes_dict = normalizer.transform(pd.DataFrame([roi_volumes_dict])).iloc[0].to_dict()

    return roi_volumes_dict


def instantiate_brainst_img_model(diffusion_steps: int = 50) -> dict:
    """Instantiate the BrainST-img model bundle from ``configs.py``.

    Args:
        diffusion_steps: Number of denoising steps to configure the noise
            scheduler with.

    Returns:
        Model bundle containing (at least) ``unet``, ``conditions_model``,
        ``noise_scheduler``, and ``autoencoder``.
    """
    networks_config = fc.dict_to_args(cfg.ARCHITECTURE_BRAINST_IMG, deep_conversion=True)
    brainst_img = instantiate_brainst_img.instantiate_model_and_load(
        networks_config,
        cfg.PATH_BRAINST_IMG_CHK,
        cfg.PATH_AUTOENCODER_CHK,
        device=cfg.DEVICE,
        dm_num_inference_steps=diffusion_steps,
    )
    return brainst_img


def instantiate_brainst_vol_model(diffusion_steps: int = 50) -> dict:
    """Instantiate the BrainST-vol model bundle from ``configs.py``.

    Args:
        diffusion_steps: Number of denoising steps to configure the noise
            scheduler with.

    Returns:
        Model bundle containing (at least) ``unet``, ``conditions_model``,
        and ``noise_scheduler``.
    """
    networks_config = fc.dict_to_args(cfg.ARCHITECTURE_BRAINST_VOL, deep_conversion=True)
    brainst_vol = instantiate_brainst_vol.instantiate_model_and_load(
        networks_config, cfg.PATH_BRAINST_VOL_CHK, device=cfg.DEVICE, dm_num_inference_steps=diffusion_steps
    )
    return brainst_vol


# =====================================================
# SYNTHESIS
# =====================================================

def brainst_vol_synthesis(
    brainst_vol: dict,
    target_covariates_dict: dict,
    seed: int = 2,
    free_guidance_ratio: float = 2.0,
) -> dict:
    """Sample a brand-new ROI-volumes profile from the BrainST-vol diffusion model.

    Args:
        brainst_vol: Instantiated BrainST-vol model bundle, as returned by
            :func:`instantiate_brainst_vol_model` (``unet``,
            ``conditions_model``, ``noise_scheduler``).
        target_covariates_dict: ``{"age": ..., "sex": ..., "dx": ...}``
            covariates to condition on (age already standardized,
            sex/dx already integer-coded).
        seed: Random seed controlling the initial noise sample.
        free_guidance_ratio: Classifier-free-guidance strength.

    Returns:
        ``{roi_name: predicted_volume}`` sampled ROI-volumes profile.
    """
    device = next(brainst_vol["unet"].parameters()).device

    # 1. Sample the initial Gaussian noise the diffusion process starts from.
    noisy_latents = utils_generation.gen_random_latents(cfg.NB_STRUCTURES, seed=seed, device=device).unsqueeze(0)

    # 2. Run the reverse diffusion process to denoise into an ROI-volumes vector.
    reconstructed_volumes = generate_volumes.diffusion_loop(
        initial_noise=noisy_latents,
        unet=brainst_vol["unet"],
        conditions_model=brainst_vol["conditions_model"],
        noise_scheduler=brainst_vol["noise_scheduler"],
        covars_list_dict=[target_covariates_dict],
        covars_keys_ordered=cfg.COVARS_LIST,
        free_guidance_ratio=free_guidance_ratio,
        return_noisy_steps=False,
    )

    # 3. Convert the raw prediction vector (batch of 1) into a named ROI-volumes dict.
    predicted_roi_volumes_vec = reconstructed_volumes[0]
    target_roi_volumes_dict = {
        roi_name: float(predicted_roi_volumes_vec[i]) for i, roi_name in enumerate(cfg.STRUCTURE_NAME_LIST_VOL)
    }

    return target_roi_volumes_dict


def brainst_img_synthesis(
    brainst_img: dict,
    target_roi_volumes_dict: dict,
    seed: int = 2,
    free_guidance_ratio: float = 2.0,
) -> np.ndarray:
    """Sample a brand-new brain image from the BrainST-img diffusion model.

    Args:
        brainst_img: Instantiated BrainST-img model bundle, as returned by
            :func:`instantiate_brainst_img_model` (``unet``,
            ``conditions_model``, ``noise_scheduler``, ``autoencoder``).
        target_roi_volumes_dict: ``{roi_name: standardized_volume}``
            profile to condition on.
        seed: Random seed controlling the initial noise sample.
        free_guidance_ratio: Classifier-free-guidance strength.

    Returns:
        The reconstructed (decoded) image, as ``float32``.
    """
    device = next(brainst_img["unet"].parameters()).device

    # 1. Sample the initial Gaussian noise latents the diffusion process starts from.
    noisy_latents = utils_generation.gen_random_latents(cfg.SHAPE_LATENT, seed=seed, device=device).unsqueeze(0)

    # 2. Run the reverse diffusion process and decode straight to image space.
    reconstructed_latents = generate_image.diffusion_loop(
        noisy_latents=noisy_latents,
        unet=brainst_img["unet"],
        conditions_model=brainst_img["conditions_model"],
        noise_scheduler=brainst_img["noise_scheduler"],
        autoencoder=brainst_img["autoencoder"],
        conditions_list=[target_roi_volumes_dict],
        conditions_keys_ordered=cfg.STRUCTURE_NAME_LIST_VOL,
        uncond_embeddings=None,
        free_guidance_ratio=free_guidance_ratio,
        decode_img=True,
        decode_first=True,
        decode_complete=True,
    )

    # 3. Extract the decoded image (batch of 1).
    reconstructed_img = reconstructed_latents["images"][0].astype(np.float32)
    return reconstructed_img


def brainst_synthesis(
    target_roi_volumes_dict: dict | None = None,
    target_covariates_dict: dict | None = None,
    target_seg: np.ndarray | None = None,
    seed: int = 2,
    diffusion_steps: int = 50,
    normalizer=None,
) -> np.ndarray:
    """End-to-end synthesis pipeline: produce a brand-new brain image.

    Exactly one conditioning source is needed to obtain the target
    ROI-volumes profile (checked in priority order):

    1. ``target_roi_volumes_dict`` -- used directly.
    2. ``target_covariates_dict``  -- BrainST-vol samples a volumes
       profile conditioned on these covariates.
    3. ``target_seg``              -- ROI volumes are computed directly
       from the given segmentation.

    Args:
        target_roi_volumes_dict: Explicit target ROI-volumes profile
            (standardized scale).
        target_covariates_dict: ``{"age", "sex", "dx"}`` covariates to
            condition BrainST-vol on.
        target_seg: Segmentation map to derive ROI volumes from (requires
            ``normalizer``).
        seed: Random seed for both diffusion models.
        diffusion_steps: Number of denoising steps for both diffusion
            models.
        normalizer: Fitted normalizer, required only when deriving
            volumes from ``target_seg``.

    Returns:
        The synthesized brain image.
    """
    # 1. Resolve the target ROI-volumes profile if not given directly.
    if target_roi_volumes_dict is None:
        if target_covariates_dict is not None:
            brainst_vol = instantiate_brainst_vol_model(diffusion_steps=diffusion_steps)
            target_roi_volumes_dict = brainst_vol_synthesis(
                brainst_vol, target_covariates_dict, seed=seed, free_guidance_ratio=cfg.BRAINST_VOL_FREE_GUIDANCE_RATIO
            )
        else:
            target_roi_volumes_dict = get_volumes_from_segmentation(target_seg, normalizer)

        logger.info("Target ROI volumes dict: %s", target_roi_volumes_dict)

    # 2. Generate the image conditioned on the resolved ROI-volumes profile.
    brainst_img = instantiate_brainst_img_model(diffusion_steps=diffusion_steps)
    reconstructed_img = brainst_img_synthesis(
        brainst_img, target_roi_volumes_dict, seed=seed, free_guidance_ratio=cfg.BRAINST_IMG_FREE_GUIDANCE_RATIO
    )

    return reconstructed_img


# =====================================================
# TRANSFORMATION
# =====================================================

def brainst_img_transformation(
    brainst_img: dict,
    img: np.ndarray,
    initial_roi_volumes_dict: dict,
    target_roi_volumes_dict: dict,
    uncond_embeddings_path_name: str | None = None,
    latents_output_path_name: str | None = None,
) -> np.ndarray:
    """Transform a real brain image toward a target ROI-volumes profile.

    Uses null-text inversion + guided diffusion. The source image is
    first encoded and "inverted" (i.e. the noise trajectory that would
    have produced it, conditioned on its *own* ROI-volumes profile, is
    recovered). The reverse diffusion is then run from that noise, but
    conditioned on the *target* ROI-volumes profile, which morphs the
    image toward the target while preserving as much of the original
    subject's structure/identity as possible.

    Args:
        brainst_img: Instantiated BrainST-img model bundle.
        img: Source (preprocessed) brain image to transform.
        initial_roi_volumes_dict: ROI-volumes profile describing ``img``
            as-is.
        target_roi_volumes_dict: Desired ROI-volumes profile for the
            output image.
        uncond_embeddings_path_name: Optional path to cache/reuse the
            null-text (unconditional) embeddings computed during
            inversion.
        latents_output_path_name: Optional path to cache/reuse the
            inverted noisy latents.

    Returns:
        The transformed (decoded) image, as ``float32``.

    Side Effects:
        If ``latents_output_path_name``/``uncond_embeddings_path_name``
        are given and do not yet exist, the inversion results are written
        to disk for reuse on subsequent calls.
    """
    # 1. Encode the source image into latent space.
    initial_latents = brainst_img["autoencoder"].encode(img).cpu().numpy()

    # 2. Invert: recover the noisy latents + null-text embeddings that
    #    would reconstruct `img` under its own (initial) ROI-volumes profile.
    inversion_result = null_inversion.create_save_load_null_inversion_results(
        brainst_img["unet"],
        brainst_img["conditions_model"],
        brainst_img["noise_scheduler"],
        initial_latents,
        initial_roi_volumes_dict,
        cfg.STRUCTURE_NAME_LIST_VOL,
        free_guidance_ratio=cfg.BRAINST_IMG_FREE_GUIDANCE_RATIO,
        compute_uncond_embeddings=True,
        num_inner_steps=2,
        early_stop_epsilon=1e-8,
        verbose=False,
        latents_output_path_name=latents_output_path_name,
        uncond_embeddings_path_name=uncond_embeddings_path_name,
    )

    # 3. Run the reverse diffusion process from the inverted (noisiest)
    #    latents, now conditioned on the target ROI-volumes profile.
    reconstructed_latents = generate_image.diffusion_loop(
        noisy_latents=inversion_result["noisy_latents"],
        unet=brainst_img["unet"],
        conditions_model=brainst_img["conditions_model"],
        noise_scheduler=brainst_img["noise_scheduler"],
        autoencoder=brainst_img["autoencoder"],
        conditions_list=[target_roi_volumes_dict],
        conditions_keys_ordered=cfg.STRUCTURE_NAME_LIST_VOL,
        uncond_embeddings=inversion_result["uncond_embeddings"],
        free_guidance_ratio=cfg.BRAINST_IMG_FREE_GUIDANCE_RATIO,
        decode_img=True,
        decode_first=True,
        decode_complete=True,
    )

    # 4. Extract the decoded image (batch of 1).
    reconstructed_img = reconstructed_latents["images"][0].astype(np.float32)
    return reconstructed_img


def brainst_vol_transformation(
    brainst_vol: dict,
    initial_roi_volumes_dict: dict,
    initial_covariates_dict: dict,
    target_covariates_dict: dict,
    free_guidance_ratio: float = 1.0,
    compute_uncond_embeddings: bool = False,
) -> dict:
    """Transform an ROI-volumes profile from an initial to a target covariate state.

    Mirrors :func:`brainst_img_transformation` but operates on ROI-volume
    vectors instead of images: invert the initial profile under its own
    covariates, then re-run the diffusion conditioned on the target
    covariates.

    Args:
        brainst_vol: Instantiated BrainST-vol model bundle.
        initial_roi_volumes_dict: ROI-volumes profile describing the
            subject's current state.
        initial_covariates_dict: ``{"age", "sex", "dx"}`` covariates
            describing the current state.
        target_covariates_dict: ``{"age", "sex", "dx"}`` covariates
            describing the desired state.
        free_guidance_ratio: Classifier-free-guidance strength used for
            both the inversion and the subsequent reverse diffusion.
        compute_uncond_embeddings: Whether to compute/return null-text
            (unconditional) embeddings during inversion.

    Returns:
        ``{roi_name: predicted_volume}`` profile for the target
        covariates.
    """
    # 1. Flatten the initial ROI-volumes dict into an ordered vector.
    initial_roi_volumes_vec = np.expand_dims(
        [initial_roi_volumes_dict[key] for key in cfg.STRUCTURE_NAME_LIST_VOL], axis=0
    )

    # 2. Invert: recover the noisy latents (+ optional null-text embeddings)
    #    that would reconstruct the initial ROI-volumes vector under the
    #    initial covariates.
    inversion_result = null_inversion_volumes.invert_latents(
        brainst_vol["unet"],
        brainst_vol["conditions_model"],
        brainst_vol["noise_scheduler"],
        input_vec=initial_roi_volumes_vec,
        covars_list_dict=[initial_covariates_dict],
        covars_keys_ordered=cfg.COVARS_LIST,
        free_guidance_ratio=free_guidance_ratio,
        num_inner_steps=4,
        early_stop_epsilon=1e-10,
        compute_uncond_embeddings=compute_uncond_embeddings,
    )

    # 3. Run the reverse diffusion process from the inverted latents, now
    #    conditioned on the target covariates.
    reconstructed_volumes = generate_volumes.diffusion_loop(
        inversion_result["noisy_latents"],
        brainst_vol["unet"],
        brainst_vol["conditions_model"],
        brainst_vol["noise_scheduler"],
        covars_list_dict=[target_covariates_dict],
        covars_keys_ordered=cfg.COVARS_LIST,
        uncond_embeddings=inversion_result["uncond_embeddings"],
        free_guidance_ratio=free_guidance_ratio,
        return_noisy_steps=False,
    )

    # 4. Convert the raw prediction vector (batch of 1) into a named ROI-volumes dict.
    predicted_roi_volumes_vec = reconstructed_volumes[0]
    predicted_roi_volumes_dict = {
        roi_name: float(predicted_roi_volumes_vec[i]) for i, roi_name in enumerate(cfg.STRUCTURE_NAME_LIST_VOL)
    }
    return predicted_roi_volumes_dict


def brainst_transformation(
    img: np.ndarray,
    seg: np.ndarray,
    normalizer,
    target_roi_volumes_dict: dict | None = None,
    target_seg: np.ndarray | None = None,
    diffusion_steps: int = 50,
    uncond_embeddings_path_name: str | None = None,
    latents_output_path_name: str | None = None,
) -> np.ndarray:
    """End-to-end transformation pipeline: morph a real image toward a target ROI/segmentation.

    Args:
        img: Source (preprocessed) brain image to transform.
        seg: Segmentation of ``img``, used to compute its current
            ROI-volumes profile.
        normalizer: Fitted normalizer used when deriving ROI volumes from
            a segmentation.
        target_roi_volumes_dict: Explicit target ROI-volumes profile. If
            omitted, it is derived from ``target_seg`` instead.
        target_seg: Target segmentation to derive ROI volumes from, used
            only when ``target_roi_volumes_dict`` is not given.
        diffusion_steps: Number of denoising steps for the BrainST-img
            model.
        uncond_embeddings_path_name: Optional path to cache/reuse the
            null-text embeddings.
        latents_output_path_name: Optional path to cache/reuse the
            inverted noisy latents.

    Returns:
        The transformed brain image.
    """
    # 1. Compute the source image's current ROI-volumes profile.
    initial_roi_volumes_dict = get_volumes_from_segmentation(seg, normalizer)
    logger.info("Initial ROI volumes dict: %s", initial_roi_volumes_dict)

    # 2. Resolve the target ROI-volumes profile if not given directly.
    if target_roi_volumes_dict is None:
        target_roi_volumes_dict = get_volumes_from_segmentation(target_seg, normalizer)
        logger.info("Target ROI volumes dict: %s", target_roi_volumes_dict)

    # 3. Transform the image via inversion + guided diffusion.
    brainst_img = instantiate_brainst_img_model(diffusion_steps=diffusion_steps)
    reconstructed_img = brainst_img_transformation(
        brainst_img,
        img,
        initial_roi_volumes_dict,
        target_roi_volumes_dict,
        uncond_embeddings_path_name=uncond_embeddings_path_name,
        latents_output_path_name=latents_output_path_name,
    )

    return reconstructed_img


def brainst_longitudinal(
    img: np.ndarray,
    seg: np.ndarray,
    normalizer,
    initial_covariates_dict: dict | None = None,
    target_covariates_dict: dict | None = None,
    diffusion_steps: int = 50,
    uncond_embeddings_path_name: str | None = None,
    latents_output_path_name: str | None = None,
) -> np.ndarray:
    """End-to-end longitudinal-prediction pipeline.

    Simulates how a real brain image would look at a different point
    along the aging / disease-progression trajectory. Unlike
    :func:`brainst_transformation` (which needs an explicit target
    ROI-volumes profile or segmentation), this pipeline *derives* the
    target ROI-volumes profile itself, by running BrainST-vol from the
    subject's initial covariates to their target covariates.

    Args:
        img: Source (preprocessed) brain image to transform.
        seg: Segmentation of ``img``, used to compute its current
            ROI-volumes profile.
        normalizer: Fitted normalizer used when deriving ROI volumes from
            a segmentation.
        initial_covariates_dict: ``{"age", "sex", "dx"}`` covariates
            describing the subject's current state.
        target_covariates_dict: ``{"age", "sex", "dx"}`` covariates
            describing the desired (future) state.
        diffusion_steps: Number of denoising steps for both diffusion
            models.
        uncond_embeddings_path_name: Optional path to cache/reuse the
            null-text embeddings.
        latents_output_path_name: Optional path to cache/reuse the
            inverted noisy latents.

    Returns:
        The longitudinally-predicted brain image.
    """
    # 1. Compute the source image's current ROI-volumes profile.
    initial_roi_volumes_dict = get_volumes_from_segmentation(seg, normalizer)
    logger.info("Initial ROI volumes dict: %s", initial_roi_volumes_dict)

    # 2. Predict the target ROI-volumes profile by transforming the
    #    initial profile from the initial covariates to the target covariates.
    brainst_vol = instantiate_brainst_vol_model(diffusion_steps=diffusion_steps)
    target_roi_volumes_dict = brainst_vol_transformation(
        brainst_vol, initial_roi_volumes_dict, initial_covariates_dict, target_covariates_dict
    )
    logger.info("Target ROI volumes dict: %s", target_roi_volumes_dict)

    # 3. Transform the image toward the predicted target ROI-volumes profile.
    brainst_img = instantiate_brainst_img_model(diffusion_steps=diffusion_steps)
    reconstructed_img = brainst_img_transformation(
        brainst_img,
        img,
        initial_roi_volumes_dict,
        target_roi_volumes_dict,
        uncond_embeddings_path_name=uncond_embeddings_path_name,
        latents_output_path_name=latents_output_path_name,
    )

    return reconstructed_img