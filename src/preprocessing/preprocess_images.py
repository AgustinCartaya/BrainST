"""End-to-end preprocessing pipeline for a single T1-weighted brain MRI.

Pipeline stages (controlled via the `steps` argument):
    1. super_resolution — SynthSR upsampling/denoising.
    2. register         — rigid registration to MNI space via USLR.
    3. segment_registered — SynthSeg segmentation of the registered image.
    4. resize           — center-crop/pad to a fixed training shape.
    5. normalize        — percentile-based intensity normalization to [0, 1].
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

import src.utils.functions as fc
import src.utils.nifti_functions as nfc
from src.utils import data_normalization, prep_segmentation, prep_synthsr
from src.utils.USLR_lite import uslr_registration

DEFAULT_PREPROCESSING_STEPS = [
    "super_resolution",
    "register",
    "segment_registered",
    "resize",
    "normalize",
]


def preprocess_image(
    input_path_name: str,
    output_path: str,
    shape_prep_img: tuple[int, int, int] = (192, 256, 192),
    steps: Sequence[str] = DEFAULT_PREPROCESSING_STEPS,
    seg_raw_path_name: str | None = None,
    verify: bool = False,
    verbose: bool = False,
    keep_intermediate_files: bool = True,
) -> tuple[str, str]:
    """Run the full BrainST preprocessing pipeline on a single raw MRI.

    Args:
        input_path_name: Path to the raw (or already super-resolved)
            input T1w image.
        output_path: Directory where all intermediate and final outputs
            are written.
        shape_prep_img: Target ``(X, Y, Z)`` shape for the ``resize`` step.
        steps: Subset/ordering of
            ``["super_resolution", "register", "segment_registered",
            "resize", "normalize"]`` to run. Stages not listed are
            skipped, but note that ``segment_registered`` is implicitly
            forced on if no valid ``seg_raw_path_name`` segmentation is
            available at that point in the pipeline (see implementation).
        seg_raw_path_name: Optional path to a pre-computed raw
            segmentation of ``input_path_name`` (skips running SynthSeg on
            the raw image if provided).
        verify: If True, skip re-running a stage when its expected output
            file already exists on disk.
        verbose: If True, pass through verbose logging to sub-steps that
            support it (registration).
        keep_intermediate_files: If False, delete intermediate files
            (super-res image, raw segmentation, MNI-registered image, MNI
            segmentation, affine matrix) after the final preprocessed
            image/segmentation are saved.

    Returns:
        A tuple ``(preprocessed_img_path, preprocessed_seg_path)`` — the
        paths to the final preprocessed image and segmentation.
    """
    img_name = fc.get_img_name(input_path_name)

    preprocessed_img_path = os.path.join(output_path, f"{img_name}_preprocessed.nii.gz")
    preprocessed_seg_path = os.path.join(output_path, f"{img_name}_preprocessed_seg.nii.gz")

    if verify and os.path.exists(preprocessed_img_path) and os.path.exists(preprocessed_seg_path):
        print(f"Skipping preprocessing for {preprocessed_img_path}, preprocessed data already exists.")
        return preprocessed_img_path, preprocessed_seg_path

    # `_current_*` track the "latest" image/segmentation as they move
    # through each stage of the pipeline (super-res -> registered -> ...).
    _current_img_path_name = input_path_name
    _current_seg_path_name = seg_raw_path_name
    _intermediate_files = []

    if "super_resolution" in steps:
        # SynthSR super-resolves the raw image; note: it casts output to UINT8 some times.
        _img_mni_synthsr_path_name = os.path.join(output_path, f"{img_name}_synthsr.nii.gz")
        prep_synthsr.save_synthSR(_current_img_path_name, _img_mni_synthsr_path_name, verify=True, verbose=False)
        _current_img_path_name = _img_mni_synthsr_path_name
        _intermediate_files.append(_img_mni_synthsr_path_name)

    if "register" in steps:
        if seg_raw_path_name is not None:
            _seg_raw_path_name = seg_raw_path_name
        else:
            # Need a raw-space segmentation to drive the USLR affine
            # registration (it aligns anatomical landmarks, not intensities).
            _seg_raw_path_name = os.path.join(output_path, f"{img_name}_seg_raw.nii.gz")
            prep_segmentation.save_synthseg_segmentation(_current_img_path_name, _seg_raw_path_name, verify=True)
            _intermediate_files.append(_seg_raw_path_name)

        _img_mni_path_name = os.path.join(output_path, f"{img_name}_mni.nii.gz")
        _seg_mni_path_name = None  # segmentation in MNI space is produced later, by "segment_registered"
        _affine_matrix_path_name = os.path.join(output_path, f"{img_name}_mni_affine.npy")

        uslr_registration.uslr_mni_registration(
            moving_img_path_name=_current_img_path_name,
            moving_seg_path_name=_seg_raw_path_name,
            out_img_path_name=_img_mni_path_name,
            out_seg_path_name=_seg_mni_path_name,
            out_affine_matrix_path_name=_affine_matrix_path_name,
            verify=verify,
            verbose=verbose,
        )

        _current_img_path_name = _img_mni_path_name

        _intermediate_files.append(_img_mni_path_name)
        _intermediate_files.append(_affine_matrix_path_name)

    # Segment the (now MNI-registered) image directly, rather than
    # resampling the raw-space segmentation, for better alignment with
    # the registered image's voxel grid.
    if "segment_registered" in steps or _current_seg_path_name is None or not os.path.exists(_current_seg_path_name):
        _seg_mni_path_name = os.path.join(output_path, f"{img_name}_seg_mni.nii.gz")
        prep_segmentation.save_synthseg_segmentation(_current_img_path_name, _seg_mni_path_name, verify=True)
        _current_seg_path_name = _seg_mni_path_name
        _intermediate_files.append(_seg_mni_path_name)

    image, image_affine = nfc.load_nifti(_current_img_path_name)
    segmentation, segmentation_affine = nfc.load_nifti(_current_seg_path_name, is_label=True)

    if "resize" in steps:
        image, _offset, new_image_affine = fc.resize_center_crop_pad(image, shape_prep_img, image_affine)
        segmentation, _, new_segmentation_affine = fc.resize_center_crop_pad(
            segmentation, shape_prep_img, segmentation_affine
        )

    if "normalize" in steps:
        image = data_normalization.normalize_image(image, percentile=(0, 100), strictly_positive=True)

    nfc.save_nifti(image, new_image_affine, preprocessed_img_path)
    nfc.save_nifti(segmentation, new_segmentation_affine, preprocessed_seg_path)

    if not keep_intermediate_files:
        for intermediate_file in _intermediate_files:
            if os.path.exists(intermediate_file):
                os.remove(intermediate_file)

    return preprocessed_img_path, preprocessed_seg_path


def postprocess_image(image: np.ndarray, original_shape: tuple[int, int, int]) -> np.ndarray:
    """Undo the `resize` preprocessing step by cropping/padding back.

    Args:
        image: Preprocessed (fixed-shape) image array.
        original_shape: The pre-preprocessing ``(X, Y, Z)`` shape to
            restore.

    Returns:
        The image resized (center-cropped/padded) back to
        ``original_shape``.
    """
    image, _offset = fc.resize_center_crop_pad(image, original_shape, None)
    return image