"""Multi-resolution tissue attention-mask generation for BrainST-img training.

For each anatomical structure of interest, builds a soft, multi-scale
attention mask from a segmentation: extract a binary structure mask,
optionally dilate/erode/blur it, then resample it down to each of the
diffusion model's internal latent resolutions. These masks supervise the
UNet's cross-attention maps during training (see
``training_brainst_img.py``'s attention-mask loss).

"""

import os

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    gaussian_filter,
)
from skimage.morphology import ball, closing, dilation, erosion, opening

import src.utils.functions as fc
import src.utils.nifti_functions as nfc
import src.utils.util_freesurfer_segmentation as ufs
from src.utils import data_normalization


def apply_morphological_operation_sequence(img: np.ndarray, op_list: list[str], selem_radius: int = 2, binary: bool = True, until_convergence: bool = False, max_iterations: int = 50) -> np.ndarray:
    """Apply a sequence of morphological operations to an image, optionally until convergence.

    Args:
        img: Input image (typically a binary or soft mask).
        op_list: List of operations to apply, in order. Supported:
            ``"dilation"``, ``"erosion"``, ``"opening"``, ``"closing"``.
        selem_radius: Radius of the (ball-shaped) structuring element.
        binary: If True, use binary morphological operations
            (``scipy.ndimage``); if False, use grayscale morphology
            (``skimage.morphology``).
        until_convergence: If True, repeat the (de-duplicated) operation
            sequence until the result stops changing between iterations.
        max_iterations: Safety cap on iterations when
            ``until_convergence=True``.

    Returns:
        The processed image.

    Raises:
        ValueError: If an unsupported operation name appears in ``op_list``.
    """
    selem = ball(radius=selem_radius)
    processed = img.copy()

    def apply_once(image):
        """Apply every operation in ``op_list`` once, in order."""
        for op in op_list:
            if op == "dilation":
                image = binary_dilation(image, structure=selem) if binary else dilation(image, selem)
            elif op == "erosion":
                image = binary_erosion(image, structure=selem) if binary else erosion(image, selem)
            elif op == "opening":
                image = binary_opening(image, structure=selem) if binary else opening(image, selem)
            elif op == "closing":
                image = binary_closing(image, structure=selem) if binary else closing(image, selem)
            else:
                raise ValueError(f"Operation '{op}' not supported.")
        return image

    if until_convergence:
        # just take no repeated operations
        op_list = list(dict.fromkeys(op_list))  # Remove duplicates while preserving order
        for i in range(max_iterations):
            prev = processed.copy()
            processed = apply_once(processed)
            if np.array_equal(prev, processed):
                break
        else:
            print(f"Warning: Did not converge after {max_iterations} iterations.")
    else:
        processed = apply_once(processed)

    return processed

def prepare_masks(mask: np.ndarray,
                  size_list: list[tuple[int, int, int]] = [(192,256,192), (12,16,12), (6,8,6)], 
                  nb_dilations: int = 0, 
                  nb_erosions: int = 0,
                  sigma: float = 0, 
                  resize_order: int = 3,
                  markov_resize: bool = True,
                  selem_radius: int = 2,
                  scale_factor: np.ndarray | float | None = None,
                  normalize: bool = False) -> list[np.ndarray]:
    """Build a multi-resolution pyramid of a (optionally morphed/scaled) soft mask.

    Args:
        mask: Input binary (or soft) mask.
        size_list: Target spatial sizes to produce, one output per entry.
        nb_dilations: Number of binary dilation passes applied first.
        nb_erosions: Number of binary erosion passes applied after dilation.
        sigma: Gaussian blur sigma applied after morphology (0 = no blur).
        resize_order: Interpolation order used for resizing
            (``fc.resize3d``).
        markov_resize: If True, each resolution is resized from the
            *previous* (already-resized) resolution rather than always
            from the original ``mask`` -- cheaper and produces a smoother
            pyramid, at the cost of compounding interpolation error.
        selem_radius: Structuring-element radius for the morphology ops.
        scale_factor: Optional per-voxel (array) or scalar weight applied
            multiplicatively to every resolution's mask (no clipping
            applied after scaling, to preserve attention-reweighting values).
        normalize: If True, apply ``data_normalization.normalize_image`` to each
            resolution's mask after resizing

    Returns:
        A list of masks, one per entry in ``size_list``, each clipped to
        ``[0, 1]`` after resizing (before any ``scale_factor`` is applied).

    Raises:
        ValueError: If ``scale_factor`` is a numpy array whose shape does
            not match ``mask.shape``.
    """

    mask = apply_morphological_operation_sequence(mask, ["dilation"]*nb_dilations, selem_radius=selem_radius)
    mask = apply_morphological_operation_sequence(mask, ["erosion"]*nb_erosions, selem_radius=selem_radius)

    if sigma > 0:
        mask = gaussian_filter(mask, sigma=sigma)

    scale_factor_mask = None
    if scale_factor is not None:
        # verify if scale factor is a numpy array with the same shape as mask
        if isinstance(scale_factor, np.ndarray):
            if scale_factor.shape == mask.shape:
                scale_factor_mask = scale_factor
            else:
                raise ValueError(f"scale_factor shape {scale_factor.shape} does not match mask shape {mask.shape}")
        # verify if scale factor is a float
        elif isinstance(scale_factor, float) or isinstance(scale_factor, int):
            scale_factor = float(scale_factor)
            scale_factor_mask = np.zeros_like(mask)
            scale_factor_mask.fill(scale_factor)

    mask = np.clip(mask, 0, 1)
    __mask_list = []
    for i in range(len(size_list)):
        if np.prod(size_list[i]) == np.prod(mask.shape):
            __mask_list.append(mask)
        else:
            if markov_resize and len(__mask_list) > 0:
                __mask_list.append(np.clip(fc.resize3d(__mask_list[-1], size_list[i], order=resize_order), 0, 1))
            else:
                __mask_list.append(np.clip(fc.resize3d(mask, size_list[i], order=resize_order), 0, 1))
        if normalize:
            __mask_list[-1] = data_normalization.normalize_image(__mask_list[-1])

        if scale_factor_mask is not None:
            # resize scale factor to the same shape as __mask_list[-1]
            if scale_factor_mask.shape != __mask_list[-1].shape:
                scale_factor_mask = fc.resize3d(scale_factor_mask, __mask_list[-1].shape, order=0)
            __mask_list[-1] = __mask_list[-1] * scale_factor_mask # no clipping to add attention reweighting

    return __mask_list




def get_tissue_masks(seg: np.ndarray, mask_index_dict: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Extract one binary structure mask per entry in a label-index mapping.

    Args:
        seg: 96-class segmentation.
        mask_index_dict: ``{structure_name: [label_id, ...]}``.

    Returns:
        ``{structure_name: binary_mask}`` (``float32``), one entry per
        key in ``mask_index_dict``.
    """
    mask_tissue_dict = {}

    for tissue_name, tissue_index in mask_index_dict.items():
        mask_tissue = ufs.merge_seg96_to_mask(seg, tissue_index).astype(np.float32)
        mask_tissue_dict[tissue_name] = mask_tissue

    return mask_tissue_dict
    


def create_tissue_masks(seg_path_name: str, tissue_masks_path: str, structure_mask_index_dict: dict[str, list[int]], mask_resolution_list: list[int], verify: bool = False) -> None:
    """Compute and save multi-resolution attention masks for every structure of a subject.

    Args:
        seg_path_name: Path to the subject's preprocessed segmentation.
        tissue_masks_path: Output directory for the ``.npy`` mask files.
        structure_mask_index_dict: ``{structure_name: [label_id, ...]}``.
        mask_resolution_list: Downsampling factors relative to the
            model's latent resolution (``latent_shape // r`` for each
            ``r`` in this list; the latent shape itself is derived as
            ``segmentation.shape // 4``).
        verify: If True and ``tissue_masks_path`` already contains at
            least ``len(structure_mask_index_dict) * len(mask_resolution_list)``
            files, skip mask creation entirely.

    Side Effects:
        Creates ``tissue_masks_path`` if needed and writes one
        ``{tissue_name}_{shape}.npy`` file per (structure, resolution)
        pair.
    """
    nb_masks = len(structure_mask_index_dict) * len(mask_resolution_list)
    if verify and os.path.exists(tissue_masks_path) and len(os.listdir(tissue_masks_path)) >= nb_masks:
        print(f"Skipping mask creation for {tissue_masks_path}, All tissue masks already exist.")
        return

    os.makedirs(tissue_masks_path, exist_ok=True)
    
    seg, _ = nfc.load_nifti(seg_path_name)

    latent_shape = np.array(seg.shape) // 4
    size_list = [latent_shape // r for r in mask_resolution_list]

    mask_tissue_dict = get_tissue_masks(seg, structure_mask_index_dict)

    for tissue_name, mask_tissue in mask_tissue_dict.items():
        mask_list = prepare_masks(
            mask=mask_tissue,
            size_list=size_list,
            nb_dilations=1,
            nb_erosions=0,
            sigma=0,
            resize_order=3,
            markov_resize=True,
            selem_radius=2,
            normalize=True,
        )
        for mask in mask_list:
            mask_shape = mask.shape
            mask_shape_text = "_".join(map(str, mask_shape))
            name_mask = f"{tissue_name}_{mask_shape_text}.npy"
            path_name_mask = os.path.join(tissue_masks_path, name_mask)
            np.save(path_name_mask, mask.astype(np.float32))