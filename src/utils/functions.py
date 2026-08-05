"""General-purpose helpers: path/name utilities, argparse<->dict
conversion, geometric resizing, and 2D visualization helpers for
multi-view slices of 3D volumes.
"""

from __future__ import annotations

import argparse
import math as math
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.ndimage import zoom


def get_time_stamp() -> str:
    """Return the current local time as a sortable ``YYYYMMDD_HHMMSS`` string."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def get_img_name(input_path_name: str, is_path: bool = True) -> str:
    """Extract a bare image name by stripping the ``.nii``/``.nii.gz`` suffix.

    Args:
        input_path_name: Either a filesystem path to a NIfTI file
            (``is_path=True``) or an already-bare filename/string
            (``is_path=False``).
        is_path: If True, first strip the directory component via
            ``os.path.basename`` before removing the NIfTI suffix.

    Returns:
        The name with any trailing ``.nii.gz`` or ``.nii`` extension removed.
    """
    if is_path:
        return os.path.basename(input_path_name).replace(".nii.gz", "").replace(".nii", "")
    return input_path_name.replace(".nii.gz", "").replace(".nii", "")


def dict_to_args(dict_to_convert: dict[str, Any], deep_conversion: bool = False) -> argparse.Namespace:
    """Convert a (optionally nested) dictionary into an ``argparse.Namespace``.

    This allows dot-notation access (``args.key``) to configuration
    dictionaries, which is used throughout the training/generation scripts
    to turn plain config dicts into attribute-style argument objects.

    Args:
        dict_to_convert: Dictionary to convert.
        deep_conversion: If True, nested dictionaries are recursively
            converted into nested ``argparse.Namespace`` objects as well.
            If False, nested dicts are kept as plain dicts.

    Returns:
        An ``argparse.Namespace`` with one attribute per top-level key.

    Raises:
        ValueError: If ``dict_to_convert`` is not a ``dict``.
    """
    if not isinstance(dict_to_convert, dict):
        raise ValueError("Argument must be a dictionary.")

    args = argparse.Namespace()
    if not deep_conversion:
        for key, value in dict_to_convert.items():
            setattr(args, key, value)
    else:
        for key, value in dict_to_convert.items():
            if isinstance(value, dict):
                setattr(args, key, dict_to_args(value, deep_conversion=True))
            else:
                setattr(args, key, value)
    return args


def args_to_dict(args: argparse.Namespace, deep_conversion: bool = False) -> dict[str, Any]:
    """Convert an ``argparse.Namespace`` back into a plain dictionary.

    Inverse of :func:`dict_to_args`.

    Args:
        args: Namespace to convert.
        deep_conversion: If True, nested ``argparse.Namespace`` values (and
            those found inside lists) are recursively converted back to
            plain dicts as well.

    Returns:
        A dictionary with one entry per attribute of ``args``.

    Raises:
        ValueError: If ``args`` is not an ``argparse.Namespace``.
    """
    if not isinstance(args, argparse.Namespace):
        raise ValueError("Argument must be an argparse.Namespace object.")
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, argparse.Namespace) and deep_conversion:
            result[key] = args_to_dict(value, deep_conversion=True)
        elif isinstance(value, list):
            result[key] = [
                args_to_dict(item, deep_conversion=True) if isinstance(item, argparse.Namespace) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def update_affine(original_affine: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    """Translate an affine matrix to account for a voxel-space crop/pad offset.

    Args:
        original_affine: Original 4x4 affine matrix.
        offset: Voxel-space displacement ``(dx, dy, dz)`` applied when
            cropping/padding an image (as returned by
            :func:`resize_center_crop_pad`).

    Returns:
        A new 4x4 affine matrix whose translation column has been adjusted
        so that world-space coordinates still line up with the resized
        image.
    """
    dx, dy, dz = offset
    # Convert the voxel-space offset into world-space translation using the
    # affine's linear (rotation/scale) part, then subtract it from the
    # existing translation column.
    translation = original_affine[:3, :3] @ np.array([dx, dy, dz])
    new_affine = original_affine.copy()
    new_affine[:3, 3] -= translation
    return new_affine


def resize3d(image, new_shape=None, scale_percent=None, order=3):
    """"Resize a 3D volume to a new shape or scale percentage.
    
    Args:
        image: 3D array of shape ``(x, y, z)``.
        new_shape: Target shape ``(nx, ny, nz)``.
        scale_percent: Scale percentage for each dimension.
        order: Interpolation order.

    Returns:
        Resized 3D array.
    Raises:
        ValueError: If neither new_shape nor scale_percent is provided.
    """

    if new_shape is None and scale_percent is None:
        raise ValueError("Either new_shape or scale_percent must be provided")
    if new_shape is not None:
        scale_factors = [new_dim / old_dim for new_dim, old_dim in zip(new_shape, image.shape)]
    else:
        scale_factors =  [scale_percent for _ in image.shape]
    return zoom(image, zoom=scale_factors, order=order)


def resize_center_crop_pad(
    image: np.ndarray,
    new_shape: tuple[int, int, int],
    affine: np.ndarray | tuple[np.ndarray, Any] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int]] | tuple[np.ndarray, tuple[int, int, int], Any]:
    """Center-crop and/or zero-pad a 3D volume to a target shape.

    For each axis independently: if the current size exceeds the target,
    the volume is center-cropped; if it is smaller, the volume is
    center-padded with zeros. This is the inverse-compatible operation
    used both to bring raw images to the fixed training shape and to map
    generated images back to their original shape.

    Args:
        image: 3D array of shape ``(x, y, z)``.
        new_shape: Target shape ``(nx, ny, nz)``.
        affine: Optional affine metadata to adjust consistently with the
            crop/pad offset. Three modes are supported:
                - ``None``: no affine adjustment; returns ``(image, offset)``.
                - ``(affine_matrix, header)`` tuple (as returned by
                  :func:`src.utils.nifti_functions.load_nifti`): returns
                  ``(image, offset, (new_affine_matrix, header))``.
                - a plain 4x4 ``np.ndarray``: returns
                  ``(image, offset, new_affine_matrix)``.

    Returns:
        A 2-tuple ``(new_image, offset)`` when ``affine is None``, otherwise
        a 3-tuple ``(new_image, offset, updated_affine)`` where
        ``updated_affine``'s shape depends on the ``affine`` input as
        described above. ``offset`` is the ``(dx, dy, dz)`` voxel
        displacement applied to each axis (positive when padding, negative
        when cropping), suitable for passing to :func:`update_affine`.
    """
    size_x, size_y, size_z = image.shape
    new_size_x, new_size_y, new_size_z = new_shape

    new_image = np.zeros((new_size_x, new_size_y, new_size_z), dtype=image.dtype)

    def get_slices(old_size: int, new_size: int) -> tuple[slice, slice]:
        """Compute the (source, destination) slices for one axis.

        If ``old_size > new_size`` the source is center-cropped; otherwise
        the destination is center-padded. Exactly one of the two slices
        will start at a non-zero offset.
        """
        if old_size > new_size:
            start = (old_size - new_size) // 2
            return slice(start, start + new_size), slice(0, new_size)
        start = (new_size - old_size) // 2
        return slice(0, old_size), slice(start, start + old_size)

    x_slice_old, x_slice_new = get_slices(size_x, new_size_x)
    y_slice_old, y_slice_new = get_slices(size_y, new_size_y)
    z_slice_old, z_slice_new = get_slices(size_z, new_size_z)

    new_image[x_slice_new, y_slice_new, z_slice_new] = image[x_slice_old, y_slice_old, z_slice_old]
    offset = (
        x_slice_new.start - x_slice_old.start,
        y_slice_new.start - y_slice_old.start,
        z_slice_new.start - z_slice_old.start,
    )

    if affine is None:
        return new_image, offset

    if isinstance(affine, tuple):
        new_affine_matrix = update_affine(affine[0], offset)
        return new_image, offset, (new_affine_matrix, affine[1])

    new_affine_matrix = update_affine(affine, offset)
    return new_image, offset, new_affine_matrix


# ---------------------------------------------------------------------------
# Visualization helpers (2D previews of 3D volumes)
# ---------------------------------------------------------------------------

def gray_to_rgb(image: np.ndarray, to_uint8: bool = True, normalize: bool = True) -> np.ndarray:
    """Convert a single-channel (grayscale) image to a 3-channel RGB image.

    Args:
        image: Grayscale array of any shape.
        normalize: If True, min-max normalize to ``[0, 1]`` before
            replication (a constant image maps to all-zeros).
        to_uint8: If True, scale to ``[0, 255]`` and cast to ``uint8``
            after normalization.

    Returns:
        Array with an extra trailing channel dimension of size 3, formed
        by replicating the (optionally normalized) grayscale values.
    """
    if normalize:
        image_min, image_max = np.min(image), np.max(image)
        if image_max > image_min:
            normalized = (image - image_min) / (image_max - image_min)
        else:
            normalized = np.zeros_like(image)
    else:
        normalized = image.copy()

    if to_uint8:
        normalized = (255 * normalized).astype(np.uint8)

    return np.stack([normalized] * 3, axis=-1)


def resize_image(
    img: np.ndarray,
    value: int | None = 512,
    mode: str = "w",
    force_other: int | None = None,
    interpolation: int | None = None,
    is_segmentation: bool = False,
) -> np.ndarray | None:
    """Resize a 2D image to a target width or height, preserving aspect ratio.

    Args:
        img: 2D (or 2D + channels) image to resize. If ``None``, returns
            ``None`` unchanged.
        value: Target size along the axis selected by ``mode``. If
            ``None``, returns a copy of ``img`` unchanged.
        mode: ``'w'`` to resize by width, ``'h'`` to resize by height; the
            other dimension is scaled proportionally (unless overridden by
            ``force_other``).
        force_other: If given, overrides the proportionally-computed size
            of the non-``mode`` axis with this fixed value.
        interpolation: OpenCV interpolation flag (e.g. ``cv2.INTER_LINEAR``).
            Ignored if ``is_segmentation=True`` (see below).
        is_segmentation: If True, forces nearest-neighbor interpolation
            (``cv2.INTER_NEAREST``), which is required to avoid introducing
            spurious label values when resizing integer segmentation maps.

    Returns:
        The resized image, or ``None``/a copy of ``img`` per the early-exit
        cases described above.

    Raises:
        ValueError: If ``mode`` is not ``'w'`` or ``'h'``.
    """
    if img is None:
        return None
    if value is None:
        return img.copy()

    if is_segmentation:
        # Nearest-neighbor avoids interpolated (non-existent) label values.
        interpolation = cv.INTER_NEAREST

    height, width = img.shape[:2]

    if mode == "w":
        scale_factor = value / width
        new_width = value
        new_height = int(height * scale_factor)
    elif mode == "h":
        scale_factor = value / height
        new_height = value
        new_width = int(width * scale_factor)
    else:
        raise ValueError("El modo debe ser 'w' (ancho) o 'h' (altura).")

    if force_other is not None:
        if mode == "w":
            new_height = force_other
        else:
            new_width = force_other

    if interpolation is not None:
        return cv.resize(img, (new_width, new_height), interpolation=interpolation)
    return cv.resize(img, (new_width, new_height))


def text_over_image(
    image: np.ndarray,
    text: str,
    text_color: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 1.0,
    font_thickness: int = 2,
    margin_ratio: float = 0.1,
) -> np.ndarray:
    """Render a text label in a white strip added above an image.

    Long text is automatically truncated with an ellipsis to fit the
    image width.

    Args:
        image: 2D grayscale or 3-channel image to annotate.
        text: Text to render.
        text_color: BGR color tuple for the text (OpenCV convention).
        font_scale: OpenCV font scale.
        font_thickness: OpenCV font stroke thickness.
        margin_ratio: Extra vertical margin above/below the text, as a
            fraction of the text height.

    Returns:
        A new image, taller than the input by the label strip height,
        with ``text`` centered in a white band above the original content.
    """
    if image.ndim == 2:
        image = image[..., None]
    height, width, channels = image.shape
    font = cv.FONT_HERSHEY_SIMPLEX

    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size
    top_margin = int(text_height * (1 + margin_ratio))

    ellipsis = "..."
    max_width = width - 20  # small horizontal margin
    if text_width > max_width:

        def crop_text_to_width(candidate_text: str) -> str:
            """Binary-search-free left-truncation until the text (+ellipsis) fits."""
            for end_index in range(len(candidate_text), 0, -1):
                cropped = candidate_text[:end_index] + ellipsis
                cropped_width = cv.getTextSize(cropped, font, font_scale, font_thickness)[0][0]
                if cropped_width <= max_width:
                    return cropped
            return ellipsis

        text = crop_text_to_width(text)
        text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size

    labeled_image = np.ones((height + top_margin, width, channels), dtype=np.uint8) * 255
    labeled_image[top_margin:] = image

    text_x = (width - text_width) // 2
    text_y = (top_margin + text_height) // 2
    cv.putText(labeled_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv.LINE_AA)

    return labeled_image


def cat_n_views_different_layers_(
    imgs3D_list: Sequence[np.ndarray],
    view_layersoffset_list: Sequence[tuple[int, int]],
    axis: int = 0,
    img_cropping: int = 50,
    to_rgb: bool = False,
) -> list[np.ndarray]:
    """Extract and concatenate 2D slices from multiple views of several 3D volumes.

    Each input volume is first resampled to an isotropic cube (so slices
    from different anatomical planes have comparable pixel spacing), then
    a 2D slice is extracted for each requested ``(view_axis, layer_offset)``
    pair and the slices for a given volume are concatenated side by side.

    Args:
        imgs3D_list: List of 3D (grayscale) or 4D (multi-channel) volumes.
            All volumes must share the same spatial shape.
        view_layersoffset_list: List of ``(view_axis, layer_offset)`` pairs.
            ``view_axis`` selects which spatial axis is sliced (0, 1, or 2);
            ``layer_offset`` shifts the slice from the volume's central
            layer along that axis.
        axis: Axis along which the per-view slices are concatenated (for a
            single volume).
        img_cropping: Number of voxels subtracted from the largest spatial
            dimension when computing the isotropic resample size (used to
            trim empty border regions).
        to_rgb: If True, convert each slice to RGB before concatenation
            (needed when overlaying color annotations later).

    Returns:
        A list with one 2D image per input volume, each being the
        concatenation of its requested view slices along ``axis``.

    Raises:
        ValueError: If a volume's shape is neither 3D nor a valid 4D
            ``(x, y, z, channels)`` array with 1 or 3 channels.
    """
    # Resample every volume to the same isotropic cube size so that slices
    # from different views (axial/coronal/sagittal) have matching scale.
    isotropic_size = max(imgs3D_list[0].shape[:3]) - img_cropping
    isotropic_volumes = []

    for volume in imgs3D_list:
        if volume.ndim == 3:
            volume = gray_to_rgb(volume, to_uint8=True, normalize=True) if to_rgb else volume[..., None]
        elif volume.ndim != 4 or volume.shape[-1] not in (1, 3):
            raise ValueError(f"Formato no soportado: {volume.shape}")

        num_channels = volume.shape[-1]
        resized_channels = [
            resize_center_crop_pad(volume[..., channel], [isotropic_size] * 3)[0] for channel in range(num_channels)
        ]
        isotropic_volumes.append(np.stack(resized_channels, axis=-1))

    # Central layer index per requested view (same for every volume, since
    # they now share the same isotropic shape).
    central_layer = [isotropic_size // 2] * len(view_layersoffset_list)
    slices_2d = []

    for volume in isotropic_volumes:
        view_slices = []
        for view_index, (view_axis, layer_offset) in enumerate(view_layersoffset_list):
            slice_index = [slice(None)] * volume.ndim
            slice_index[view_axis] = central_layer[view_index] + layer_offset
            # flipud + transpose orients the slice for conventional radiological display.
            view_slice = np.flipud(volume[tuple(slice_index)].transpose(1, 0, 2))
            view_slices.append(view_slice)
        slices_2d.append(np.concatenate(view_slices, axis=axis))

    if not to_rgb:
        slices_2d = [image[..., 0] if image.shape[-1] == 1 else image for image in slices_2d]
    return slices_2d


def cat_n_views_different_layers(
    imgs3D_list: Sequence[np.ndarray],
    view_layersoffset_list: Sequence[tuple[int, int]],
    axis: int = 0,
    img_cropping: int = 50,
    to_rgb: bool = False,
) -> list[np.ndarray]:
    """Extract and concatenate 2D slices from multiple views of several 3D volumes.

    Each input volume is first resampled to an isotropic cube (so slices
    from different anatomical planes have comparable pixel spacing), then
    a 2D slice is extracted for each requested ``(view_axis, layer_offset)``
    pair and the slices for a given volume are concatenated side by side.

    Args:
        imgs3D_list: List of 3D (grayscale) or 4D (multi-channel) volumes.
            All volumes must share the same spatial shape.
        view_layersoffset_list: List of ``(view_axis, layer_offset)`` pairs.
            ``view_axis`` selects which spatial axis is sliced (0, 1, or 2);
            ``layer_offset`` shifts the slice from the volume's central
            layer along that axis.
        axis: Axis along which the per-view slices are concatenated (for a
            single volume).
        img_cropping: Number of voxels subtracted from the largest spatial
            dimension when computing the isotropic resample size (used to
            trim empty border regions).
        to_rgb: If True, convert each slice to RGB before concatenation
            (needed when overlaying color annotations later).

    Returns:
        A list with one 2D image per input volume, each being the
        concatenation of its requested view slices along ``axis``.

    Raises:
        ValueError: If a volume's shape is neither 3D nor a valid 4D
            ``(x, y, z, channels)`` array with 1 or 3 channels.
    """

    # Resample every volume to the same isotropic cube size so that slices
    # from different views (axial/coronal/sagittal) have matching scale.
    isotropic_size = max(imgs3D_list[0].shape[:3]) - img_cropping
    isotropic_volumes = []

    for volume in imgs3D_list:
        if volume.ndim == 3:
            volume = gray_to_rgb(volume, to_uint8=True, normalize=True) if to_rgb else volume[..., None]
        elif volume.ndim != 4 or volume.shape[-1] not in (1, 3):
            raise ValueError(f"Formato no soportado: {volume.shape}")

        num_channels = volume.shape[-1]
        resized_channels = [
            resize_center_crop_pad(volume[..., channel], [isotropic_size] * 3)[0] for channel in range(num_channels)
        ]
        isotropic_volumes.append(np.stack(resized_channels, axis=-1))

    # Central layer index per requested view (same for every volume, since
    # they now share the same isotropic shape).
    central_layer = [isotropic_size // 2] * len(view_layersoffset_list)
    slices_2d = []

    for i, volume in enumerate(isotropic_volumes):
        view_slices = []
        for view_index, (view_axis, layer_offset) in enumerate(view_layersoffset_list):
            slice_index = [slice(None)] * volume.ndim
            slice_index[view_axis] = central_layer[view_index] + layer_offset
            # flipud + transpose orients the slice for conventional radiological display.
            view_slice = np.flipud(volume[tuple(slice_index)].transpose(1, 0, 2))

            # # Center crop to the original size in the dimension that is not used for concatenation.
            crop_axis = axis  # the spatial axis of view_slice that is *not* used for concatenation
            _slice_index = [slice(None)] * imgs3D_list[i].ndim
            _slice_index[view_axis] = 0
            original_size = np.flipud(imgs3D_list[i][tuple(_slice_index)].T).shape
            current_size = view_slice.shape[crop_axis]
            if original_size[crop_axis] < isotropic_size:
                start = (current_size - original_size[crop_axis]) // 2
                end = start + original_size[crop_axis]
                if crop_axis == 0:
                    view_slice = view_slice[start:end, :, :]
                else:
                    view_slice = view_slice[:, start:end, :]

            view_slices.append(view_slice)
        slices_2d.append(np.concatenate(view_slices, axis=axis))

    if not to_rgb:
        slices_2d = [image[..., 0] if image.shape[-1] == 1 else image for image in slices_2d]
    return slices_2d




# def cat_n_views_different_layers(
#     imgs3D_list: Sequence[np.ndarray],
#     view_layersoffset_list: Sequence[tuple[int, int]],
#     axis: int = 0,
#     img_cropping: int | list[int] | tuple[int, int, int] | None = 50,
#     to_rgb: bool = False,
# ) -> list[np.ndarray]:
#     """Extract and concatenate 2D slices from multiple views of several 3D volumes.

#     Each input volume is first resampled to an isotropic cube (so slices
#     from different anatomical planes have comparable pixel spacing), then
#     a 2D slice is extracted for each requested ``(view_axis, layer_offset)``
#     pair and the slices for a given volume are concatenated side by side.

#     Args:
#         imgs3D_list: List of 3D (grayscale) or 4D (multi-channel) volumes.
#             All volumes must share the same spatial shape.
#         view_layersoffset_list: List of ``(view_axis, layer_offset)`` pairs.
#             ``view_axis`` selects which spatial axis is sliced (0, 1, or 2);
#             ``layer_offset`` shifts the slice from the volume's central
#             layer along that axis.
#         axis: Axis along which the per-view slices are concatenated (for a
#             single volume).
#         img_cropping: Number of voxels subtracted from the largest spatial
#             dimension when computing the isotropic resample size (used to
#             trim empty border regions).
#         to_rgb: If True, convert each slice to RGB before concatenation
#             (needed when overlaying color annotations later).

#     Returns:
#         A list with one 2D image per input volume, each being the
#         concatenation of its requested view slices along ``axis``.

#     Raises:
#         ValueError: If a volume's shape is neither 3D nor a valid 4D
#             ``(x, y, z, channels)`` array with 1 or 3 channels.
#     """

#     used_img_list = imgs3D_list
#     if img_cropping is not None:
#         if isinstance(img_cropping, int):
#             img_cropping = [img_cropping] * 3
#         # img_cropping = [max(0, img.shape[i] - img_cropping[i]) for i, img in enumerate(imgs3D_list[0].shape[:3])]
#         img_cropping_list = []
#         for i, img in enumerate(imgs3D_list):
#             img_cropping_list.append([max(0, img.shape[j] - img_cropping[j]) for j in range(3)])

#         used_img_list = [resize_center_crop_pad(img, img_cropping_list[i])[0] for i, img in enumerate(imgs3D_list)]

#     # Resample every volume to the same isotropic cube size so that slices
#     # from different views (axial/coronal/sagittal) have matching scale.
#     isotropic_size = max(max(img.shape[:3]) for img in used_img_list)
#     isotropic_volumes = []


#     for volume in used_img_list:
#         if volume.ndim == 3:
#             volume = gray_to_rgb(volume, to_uint8=True, normalize=True) if to_rgb else volume[..., None]
#         elif volume.ndim != 4 or volume.shape[-1] not in (1, 3):
#             raise ValueError(f"Formato no soportado: {volume.shape}")

#         num_channels = volume.shape[-1]
#         resized_channels = [
#             resize_center_crop_pad(volume[..., channel], [isotropic_size] * 3)[0] for channel in range(num_channels)
#         ]
#         isotropic_volumes.append(np.stack(resized_channels, axis=-1))

#     # Central layer index per requested view (same for every volume, since
#     # they now share the same isotropic shape).
#     central_layer = [isotropic_size // 2] * len(view_layersoffset_list)
#     slices_2d = []

#     for i, volume in enumerate(isotropic_volumes):
#         view_slices = []
#         for view_index, (view_axis, layer_offset) in enumerate(view_layersoffset_list):
#             slice_index = [slice(None)] * volume.ndim
#             slice_index[view_axis] = central_layer[view_index] + layer_offset
#             # flipud + transpose orients the slice for conventional radiological display.
#             view_slice = np.flipud(volume[tuple(slice_index)].transpose(1, 0, 2))

#             # # Center crop to the original size in the dimension that is not used for concatenation.
#             crop_axis = axis  # the spatial axis of view_slice that is *not* used for concatenation
#             _slice_index = [slice(None)] * used_img_list[i].ndim
#             _slice_index[view_axis] = 0
#             original_size = np.flipud(used_img_list[i][tuple(_slice_index)].T).shape
#             current_size = view_slice.shape[crop_axis]
#             if original_size[crop_axis] < isotropic_size:
#                 start = (current_size - original_size[crop_axis]) // 2
#                 end = start + original_size[crop_axis]
#                 if crop_axis == 0:
#                     view_slice = view_slice[start:end, :, :]
#                 else:
#                     view_slice = view_slice[:, start:end, :]

#             view_slices.append(view_slice)
#         slices_2d.append(np.concatenate(view_slices, axis=axis))

#     if not to_rgb:
#         slices_2d = [image[..., 0] if image.shape[-1] == 1 else image for image in slices_2d]
#     return slices_2d




def cat_3_views(
    imgs3D_list: Sequence[np.ndarray],
    axis: int = 0,
    layer_offset: Sequence[int] | None = None,
    img_cropping: int = 50,
    to_rgb: bool = False,
) -> list[np.ndarray]:
    """Convenience wrapper extracting the 3 canonical anatomical views (axial/coronal/sagittal).

    Equivalent to :func:`cat_n_views_different_layers` with
    ``view_layersoffset_list = [(2, layer_offset[0]), (1, layer_offset[1]), (0, layer_offset[2])]``.

    Args:
        imgs3D_list: List of 3D/4D volumes (see
            :func:`cat_n_views_different_layers`).
        axis: Axis along which the 3 view slices are concatenated.
        layer_offset: Per-view offsets from each volume's central layer,
            ordered ``[axis-2 offset, axis-1 offset, axis-0 offset]``.
            Defaults to ``[0, 0, 0]`` (exact central slice for each view).
        img_cropping: See :func:`cat_n_views_different_layers`.
        to_rgb: See :func:`cat_n_views_different_layers`.

    Returns:
        One concatenated 3-view 2D image per input volume.
    """
    if layer_offset is None:
        layer_offset = [0, 0, 0]
    return cat_n_views_different_layers(
        imgs3D_list,
        view_layersoffset_list=[(2, layer_offset[0]), (1, layer_offset[1]), (0, layer_offset[2])],
        axis=axis,
        img_cropping=img_cropping,
        to_rgb=to_rgb,
    )



def imgshow(
    img: np.ndarray,
    name: str = "",
    scale: float = 1,
    subplot=None,
    bgr2rgb: bool = False,
    cmap: str = "gray",
    range_values: list = [None, None],
) -> None:
    """Display a single 2D image with matplotlib.

    Args:
        img: 2D (grayscale) or 3D (color) image array to display.
        name: Title shown above the image.
        scale: Scale factor applied to the image's pixel dimensions when
            computing the displayed extent.
        subplot: Optional matplotlib ``Axes`` to draw into. If ``None``, a
            new figure is created.
        bgr2rgb: If True, converts ``img`` from OpenCV's BGR channel order
            to RGB before displaying.
        cmap: Colormap used for grayscale images.
        range_values: ``[vmin, vmax]`` passed to ``imshow``, or the string
            ``"auto"`` to use the image's own min/max.
    """
    width, height = get_resize_dimensions(img, scale)
    if range_values == "auto":
        range_values = [np.min(img), np.max(img)]

    if bgr2rgb:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if subplot is None:
        plt.figure()
        plt.imshow(img, extent=[0, width, 0, height], cmap=cmap, vmin=range_values[0], vmax=range_values[1])
        plt.title(name)
        plt.axis("off")
    else:
        subplot.imshow(img, extent=[0, width, 0, height], cmap=cmap, vmin=range_values[0], vmax=range_values[1])
        subplot.set_title(name)
        subplot.axis("off")


def imgshow_3D(
    img3D: np.ndarray,
    name: str = "",
    current_plane: int | None = None,
    cmap: str = "gray",
    moving_plane: int = 2,
    transpose: bool = True,
    flip_ax: int = 0,
    show: bool = True,
    range_values: list = [None, None],
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Display an interactive single-volume 3D viewer with a slice slider.

    Renders one slice at a time from a 3D (or 4D RGB) volume, with a
    matplotlib ``Slider`` widget to scroll through the ``moving_plane``
    axis.

    Args:
        img3D: 3D (grayscale) or 4D (RGB, last axis size 3) volume.
        name: Title shown above the image.
        current_plane: Initial slice index along ``moving_plane``. Defaults
            to the volume's central slice.
        cmap: Colormap used for grayscale volumes.
        moving_plane: Axis index that the slider scrolls through.
        transpose: If True, transpose the first two axes of each slice
            before display (matches conventional radiological orientation).
        flip_ax: Axis to flip each 2D slice along, or a negative value to
            disable flipping.
        show: If True, calls ``plt.show()`` at the end.
        range_values: ``[vmin, vmax]`` passed to ``imshow``, or the string
            ``"auto"`` to use the volume's own min/max.
        figsize: Matplotlib figure size, in inches.

    Note:
        This function appends the created ``Slider`` to a module-level
        name ``sliders`` that is referenced but never defined in this
        module (unlike :func:`imgshow_3D_list`, which takes ``sliders`` as
        a parameter). Calling this function will raise ``NameError``
        unless a global ``sliders`` list has been injected into this
        module's namespace by external code. This is a pre-existing issue,
        left unchanged pending review (see refactor notes).
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)

    if range_values == "auto":
        range_values = [np.min(img3D), np.max(img3D)]

    # Show the initial image
    current_plane = current_plane if current_plane is not None else img3D.shape[moving_plane] // 2
    img = ax.imshow(
        get_3Dimg_plane(img3D, current_plane, moving_plane, transpose=transpose, flip_ax=flip_ax),
        cmap=cmap,
        vmin=range_values[0],
        vmax=range_values[1],
    )

    # Create a slider to change the displayed plane.
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, f"layer {moving_plane}", 0, img3D.shape[moving_plane] - 1, valinit=current_plane, valstep=1)

    sliders.append(slider)  # NOTE: `sliders` is undefined in this module; see docstring.

    def update(val):
        """Redraw the displayed slice when the slider value changes."""
        current_plane = int(slider.val)
        img.set_data(get_3Dimg_plane(img3D, current_plane, moving_plane, transpose=transpose, flip_ax=flip_ax))
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    if show:
        plt.show()


def create_index_square_table_size_n(nb_elements: int) -> np.ndarray:
    """Arrange ``nb_elements`` indices into a roughly square 2D grid.

    Args:
        nb_elements: Number of elements to place in the grid.

    Returns:
        A 2D object array of shape ``(rows, columns)`` where
        ``rows * columns >= nb_elements``, containing indices
        ``0..nb_elements-1`` in row-major order followed by ``None``
        padding for any unused cells.
    """
    numbers = list(range(nb_elements))
    columns = int(np.ceil(np.sqrt(nb_elements)))
    rows = int(np.ceil(nb_elements / columns))

    # Array with extra cells filled with None (unused grid positions).
    padded_array = np.full(rows * columns, None)
    padded_array[:nb_elements] = numbers

    return padded_array.reshape(rows, columns)


def create_index_rectangular_table_size_n(
    nb_elements: int,
    rows: int | None = None,
    cols: int | None = None,
) -> np.ndarray | None:
    """Arrange ``nb_elements`` indices into a grid with a fixed row or column count.

    Exactly one of ``rows``/``cols`` must be provided; the other dimension
    is computed automatically to fit ``nb_elements``.

    Args:
        nb_elements: Number of elements to place in the grid.
        rows: Fixed number of rows (mutually exclusive with ``cols``).
        cols: Fixed number of columns (mutually exclusive with ``rows``).

    Returns:
        A 2D object array of shape ``(rows, cols)`` with indices
        ``0..nb_elements-1`` in row-major order, ``None``-padded for
        unused cells; or ``None`` if the arguments are invalid (both/
        neither of ``rows``/``cols`` given), in which case a message is
        printed instead of raising.
    """
    if rows is not None and rows < 0:
        rows = None
    if cols is not None and cols < 0:
        cols = None
    if rows is not None and cols is not None:
        print("Only specify either rows or cols, not both.")
        return None
    if rows is None and cols is None:
        print("Specify either rows or cols.")
        return None
    if rows:
        cols = (nb_elements + rows - 1) // rows
    else:
        rows = (nb_elements + cols - 1) // cols

    padded_array = np.full(rows * cols, None)
    padded_array[:nb_elements] = list(range(nb_elements))
    return padded_array.reshape(rows, cols)


def concat_2Dimages_into_rectangular_table(
    images: list[np.ndarray],
    custom_disposition: np.ndarray | tuple[int | None, int | None] | None = None,
) -> np.ndarray:
    """Tile a list of same-shaped 2D/3D images into a single rectangular mosaic.

    Args:
        images: List of 2D (grayscale) or 3D (multi-channel) images, all
            sharing the same shape.
        custom_disposition: Controls the grid layout:
            - ``None``: arrange into a roughly square grid (see
              :func:`create_index_square_table_size_n`).
            - ``(rows, cols)`` tuple (one of which may be ``None``):
              arrange into a grid with a fixed row or column count (see
              :func:`create_index_rectangular_table_size_n`).
            - A pre-built index array (as returned by either helper above):
              used directly as the tile layout.

    Returns:
        A single 2D/3D ``float32`` array containing all input images tiled
        according to the resolved layout. Grid cells with no assigned
        image remain zero-filled.
    """
    if custom_disposition is None:
        table_layout = create_index_square_table_size_n(len(images))
    elif type(custom_disposition) == tuple:
        table_layout = create_index_rectangular_table_size_n(len(images), rows=custom_disposition[0], cols=custom_disposition[1])
    else:
        table_layout = custom_disposition

    image_shape = images[0].shape
    if len(image_shape) == 2:
        full_image = np.zeros((image_shape[0] * table_layout.shape[0], image_shape[1] * table_layout.shape[1]), dtype=np.float32)
    else:
        full_image = np.zeros(
            (image_shape[0] * table_layout.shape[0], image_shape[1] * table_layout.shape[1], image_shape[2]),
            dtype=np.float32,
        )

    for row in range(table_layout.shape[0]):
        for col in range(table_layout.shape[1]):
            if table_layout[row, col] is not None:
                full_image[
                    row * image_shape[0] : (row + 1) * image_shape[0],
                    col * image_shape[1] : (col + 1) * image_shape[1],
                ] = images[table_layout[row, col]]
    return full_image


def get_3Dimg_list_plane(
    img3D_list: list[np.ndarray],
    layer: int = 100,
    moving_plane: int = 2,
    transpose: bool = True,
    flip_ax: int = 0,
    subtitle_space: int | None = 20,
) -> list[np.ndarray]:
    """Extract the same 2D slice from a list of 3D volumes.

    Args:
        img3D_list: List of 3D (or higher-dimensional) volumes, all
            sharing the same shape along ``moving_plane``.
        layer: Slice index along ``moving_plane`` to extract.
        moving_plane: Axis index to slice along.
        transpose: If True, transpose the first two axes of each slice
            (matches conventional radiological orientation).
        flip_ax: Axis to flip each slice along, or a negative value to
            disable flipping.
        subtitle_space: If not ``None``, pads ``subtitle_space`` rows of
            zeros above each slice to leave room for a text label.

    Returns:
        A list of 2D (or higher-dimensional) slices, one per input volume.
    """
    img2D_list = []
    for img3D in img3D_list:
        slice_tuple = [slice(None)] * img3D.ndim
        slice_tuple[moving_plane] = layer
        img2D = img3D[tuple(slice_tuple)]
        if transpose:
            img2D = np.transpose(img2D, (1, 0) + tuple(range(2, img2D.ndim)))
        if flip_ax >= 0:
            img2D = np.flip(img2D, axis=flip_ax)
        if subtitle_space is not None:
            if img2D.ndim == 2:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0)), mode="constant", constant_values=0)
            else:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)
        img2D_list.append(img2D)
    return img2D_list


def imgshow_3D_list(
    img3D_list: list[np.ndarray],
    name: str = "",
    current_plane: int | None = None,
    cmap: str = "gray",
    moving_plane: int = 2,
    transpose: bool = True,
    flip_ax: int = 0,
    show: bool = True,
    range_values: list = [None, None],
    subimg_titles: list[str] | None = None,
    subtitle_space: int = 20,
    subtitle_help: tuple[int, int] = (20, 10),
    figsize: tuple[float, float] = (10, 8),
    custom_disposition=None,
    sliders: list = [],
    save_dict: dict | None = None,
) -> None:
    """Display an interactive tiled grid of multiple 3D volumes with a shared slice slider.

    Args:
        img3D_list: List of 3D (grayscale) or 4D (RGB) volumes, all
            sharing the same shape.
        name: Title shown above the grid.
        current_plane: Initial slice index along ``moving_plane``. Defaults
            to the first volume's central slice.
        cmap: Colormap used for grayscale volumes.
        moving_plane: Axis index that the shared slider scrolls through.
        transpose: If True, transpose each slice's first two axes.
        flip_ax: Axis to flip each slice along, or a negative value to
            disable flipping.
        show: If True, calls ``plt.show()`` at the end.
        range_values: ``[vmin, vmax]`` passed to ``imshow``.
        subimg_titles: Optional per-volume text labels drawn over each tile.
        subtitle_space: Vertical padding (pixels) reserved above each tile
            for ``subimg_titles``.
        subtitle_help: ``(x, y)`` pixel offset applied when positioning
            each ``subimg_titles`` label.
        figsize: Matplotlib figure size, in inches.
        custom_disposition: Grid layout override; see
            :func:`concat_2Dimages_into_rectangular_table`.
        sliders: List that the created ``Slider`` widget is appended to,
            so the caller can keep a reference alive (matplotlib widgets
            are garbage-collected otherwise, which silently disables
            their callbacks).
        save_dict: If given, a dict with keys ``"path_name"`` and ``"dpi"``
            used to additionally save the figure to disk.

    Side Effects:
        Appends the created ``Slider`` to the ``sliders`` list argument.
        If ``save_dict`` is given, writes an image file to
        ``save_dict["path_name"]``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15)

    # Show the initial mosaic.
    current_plane = img3D_list[0].shape[moving_plane] // 2 if current_plane is None else current_plane
    img2D_list = get_3Dimg_list_plane(img3D_list, current_plane, moving_plane, transpose=transpose, flip_ax=flip_ax, subtitle_space=subtitle_space)
    full_image = concat_2Dimages_into_rectangular_table(img2D_list, custom_disposition=custom_disposition).astype(img3D_list[0].dtype)
    img = ax.imshow(full_image, cmap=cmap, vmin=range_values[0], vmax=range_values[1])

    # Create a slider to change the displayed plane.
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, f"layer {moving_plane}", 0, img3D_list[0].shape[moving_plane] - 1, valinit=current_plane, valstep=1)

    sliders.append(slider)

    def update(val):
        """Redraw the mosaic when the slider value changes."""
        current_plane = int(slider.val)
        img2D_list = get_3Dimg_list_plane(img3D_list, current_plane, moving_plane, transpose=transpose, flip_ax=flip_ax, subtitle_space=subtitle_space)
        full_image = concat_2Dimages_into_rectangular_table(img2D_list, custom_disposition=custom_disposition).astype(img3D_list[0].dtype)
        img.set_data(full_image)
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    ax.axis("off")

    plt.rcParams["font.family"] = "monospace"
    if subimg_titles is not None:
        if custom_disposition is None:
            custom_disposition = create_index_square_table_size_n(len(subimg_titles))
        elif type(custom_disposition) == tuple:
            custom_disposition = create_index_rectangular_table_size_n(len(subimg_titles), rows=custom_disposition[0], cols=custom_disposition[1])
        fontsize = 12
        font_size_inch = fontsize / 72
        font_size_pixels = int(np.ceil(font_size_inch * fig.dpi))

        img_size = img2D_list[0].shape
        for row in range(custom_disposition.shape[0]):
            for col in range(custom_disposition.shape[1]):
                if custom_disposition[row, col] is not None:
                    init_center_x = img_size[1] // 2 - ((len(subimg_titles[custom_disposition[row, col]])) * font_size_pixels) // 2
                    pos_x = init_center_x + col * img_size[1] + subtitle_help[0]
                    pos_y = font_size_pixels + row * img_size[0] + subtitle_help[1]
                    ax.text(
                        pos_x,
                        pos_y,
                        subimg_titles[custom_disposition[row, col]],
                        color="white",
                        fontdict={"family": "monospace", "weight": "normal", "size": fontsize},
                    )
    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches="tight")

    if show:
        plt.colorbar(img)
        plt.show()


def imgshow_list(
    img_list: list[np.ndarray],
    name: str = "",
    cmap: str = "gray",
    show: bool = True,
    range_values: list = [None, None],
    subimg_titles: list[str] | None = None,
    subtitle_space: int = 20,
    figsize: tuple[float, float] = (10, 8),
    custom_disposition=None,
    save_dict: dict | None = None,
    show_colorbar: bool = False,
) -> None:
    """Display a static tiled grid of 2D (optionally RGB) images.

    Unlike :func:`imgshow_3D_list`, this displays plain 2D images (no
    volume slicing / slider), making it suitable for previewing a fixed
    set of already-extracted slices side by side.

    Args:
        img_list: List of 2D (grayscale) or 3D (RGB) images.
        name: Title shown above the grid.
        cmap: Colormap used for grayscale images.
        show: If True, calls ``plt.show()`` at the end.
        range_values: ``[vmin, vmax]`` passed to ``imshow``.
        subimg_titles: Optional per-image text labels drawn above each tile.
        subtitle_space: Vertical padding (pixels) reserved above each tile
            for ``subimg_titles``.
        figsize: Matplotlib figure size, in inches.
        custom_disposition: Grid layout override; see
            :func:`concat_2Dimages_into_rectangular_table`.
        save_dict: If given, a dict with keys ``"path_name"`` and ``"dpi"``
            used to additionally save the figure to disk.
        show_colorbar: If True, draws a colorbar next to the mosaic.

    Side Effects:
        If ``save_dict`` is given, writes an image file to
        ``save_dict["path_name"]``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15)

    # Pad each image with blank space for the subtitle, if requested.
    img_list_padded = []
    for img in img_list:
        img2D = img.copy()
        if subtitle_space is not None:
            if img2D.ndim == 2:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0)), mode="constant", constant_values=0)
            else:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)
        img_list_padded.append(img2D)

    full_image = concat_2Dimages_into_rectangular_table(img_list_padded, custom_disposition=custom_disposition).astype(img_list_padded[0].dtype)
    img = ax.imshow(full_image, cmap=cmap, vmin=range_values[0], vmax=range_values[1])

    ax.set_title(name)
    ax.axis("off")

    plt.rcParams["font.family"] = "monospace"
    if subimg_titles is not None:
        if custom_disposition is None:
            custom_disposition = create_index_square_table_size_n(len(subimg_titles))
        elif type(custom_disposition) == tuple:
            custom_disposition = create_index_rectangular_table_size_n(len(subimg_titles), rows=custom_disposition[0], cols=custom_disposition[1])
        fontsize = 12
        font_size_inch = fontsize / 72
        font_size_pixels = int(np.ceil(font_size_inch * fig.dpi))

        img_size = img_list_padded[0].shape

        img_height, img_width = img_size[:2]
        for row in range(custom_disposition.shape[0]):
            for col in range(custom_disposition.shape[1]):
                idx = custom_disposition[row, col]
                if idx is not None:
                    x_center = col * img_width + img_width / 2
                    y_pos = row * img_height + subtitle_space / 2  # centered above the image
                    ax.text(
                        x_center,
                        y_pos,
                        subimg_titles[idx],
                        color="white",
                        ha="center",
                        va="center",
                        fontdict={"family": "monospace", "weight": "normal", "size": fontsize},
                    )

    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches="tight")

    if show:
        if show_colorbar:
            plt.colorbar(img)
        plt.show()