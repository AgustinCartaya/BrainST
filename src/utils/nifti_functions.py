"""Loading and saving of NIfTI (.nii / .nii.gz) images and segmentations.

Kept intentionally close to nibabel's primitives; the only non-trivial
behavior here is the explicit handling of scl_slope/scl_inter so that
saved intensity images are not re-scaled on the next load.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

# Fallback affine used only when no affine/header metadata is available
# (e.g., synthetic volumes with no source NIfTI to inherit geometry from).
# Encodes an axis-aligned, 1mm-isotropic space with an origin offset
# consistent with the shape used throughout preprocessing (192, 256, 192).
DEFAULT_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -98.0],
        [0.0, 1.0, 0.0, -134.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def load_nifti(
    path_name: str,
    transpose: bool = False,
    is_label: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, nib.Nifti1Header]]:
    """Load a NIfTI file as a numpy array plus its affine/header metadata.

    Args:
        path_name: Path to the ``.nii``/``.nii.gz`` file.
        transpose: If True, transpose axes from ``(x, y, z)`` to
            ``(y, x, z)``. Off by default; kept for backward compatibility
            with older preprocessing paths that expected this layout.
        is_label: If True, the file is treated as a segmentation/label
            map: the raw integer array is read via ``dataobj`` (no
            scl_slope/scl_inter scaling applied, dtype preserved). If
            False, the file is treated as an intensity image: scaling is
            applied and the result is cast to ``float32``.

    Returns:
        A tuple ``(image_data, (affine, header))``. The ``(affine,
        header)`` pair should be passed back to :func:`save_nifti` to
        preserve the original image geometry and header metadata.
    """
    nifti_image = nib.load(path_name)

    if is_label:
        # Preserve integer labels exactly; do not apply intensity scaling.
        image_data = np.asanyarray(nifti_image.dataobj)
    else:
        # Apply scl_slope / scl_inter (as recorded in the header) and
        # return float32, matching the precision used by the rest of the
        # preprocessing/training pipeline.
        image_data = nifti_image.get_fdata(dtype=np.float32)

    if transpose:
        image_data = np.transpose(image_data, (1, 0, 2))

    return image_data, (nifti_image.affine, nifti_image.header)


def save_nifti(
    image: np.ndarray,
    affine: tuple[np.ndarray, nib.Nifti1Header] | None = None,
    output_path_name: str | None = None,
) -> None:
    """Save a numpy array as a NIfTI file.

    Args:
        image: Array to save.
        affine: Either:
            - a ``(affine_matrix, header)`` tuple, as returned by
              :func:`load_nifti`, used to preserve the source image's
              affine and header metadata (the header's dtype and
              scl_slope/scl_inter are overwritten to match ``image``); or
            - ``None``, in which case :data:`DEFAULT_AFFINE` is used and
              no other header metadata is carried over.
        output_path_name: Destination path for the ``.nii``/``.nii.gz``
            file.

    Raises:
        ValueError: If ``output_path_name`` is not provided, or if
            ``affine`` is neither ``None`` nor a 2-element
            ``(matrix, header)`` tuple.
    """
    if output_path_name is None:
        raise ValueError("output_path_name must be provided.")

    if affine is not None:
        # Reuse the source image's affine + header metadata (typically a
        # round trip from load_nifti): update only what must change to
        # reflect the array actually being written.
        original_affine, header = affine
        header = header.copy()

        header.set_data_dtype(image.dtype)

        # `image` already holds the real (unscaled) values, so disable
        # NIfTI-level rescaling on load to avoid double-scaling downstream.
        header["scl_slope"] = 1
        header["scl_inter"] = 0

        nifti_image = nib.Nifti1Image(image, affine=original_affine, header=header)
    else:
        # No source metadata available: fall back to the default affine.
        nifti_image = nib.Nifti1Image(image, affine=DEFAULT_AFFINE)
        nifti_image.set_data_dtype(image.dtype)
        nifti_image.header["scl_slope"] = 1
        nifti_image.header["scl_inter"] = 0

    nib.save(nifti_image, output_path_name)