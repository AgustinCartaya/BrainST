"""Compute region-of-interest (ROI) volumes from a brain segmentation map."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def merge_segmentation_to_mask(segmentation: np.ndarray, label_values: Sequence[int]) -> np.ndarray:
    """Build a binary mask that selects a set of segmentation labels.

    Args:
        segmentation: Integer label map (any shape).
        label_values: Label values to include in the resulting mask.

    Returns:
        Binary array (0/1), same shape as ``segmentation``, where voxels
        whose label is in ``label_values`` are set to 1.
    """
    mask = np.isin(segmentation, label_values)
    return np.where(mask, 1, 0)


def get_volumes(
    segmentation: np.ndarray,
    structure_name_to_labels: dict[str, Sequence[int] | None],
) -> dict[str, int]:
    """Compute voxel-count volumes for a set of anatomical structures.

    Args:
        segmentation: Integer label map.
        structure_name_to_labels: Mapping from structure name to either:
            - a sequence of label values to merge into that structure, or
            - ``None``, meaning "all non-background voxels"
              (equivalent to ``segmentation > 0``), typically used for a
              "total"/intracranial-volume entry.

    Returns:
        Dictionary mapping each structure name to its volume in voxels
        (not mm^3 — multiply by voxel volume if physical units are needed).
    """
    volumes: dict[str, int] = {}
    for structure_name, label_values in structure_name_to_labels.items():
        if label_values is None:
            structure_mask = np.where(segmentation > 0, 1, 0)
        else:
            structure_mask = merge_segmentation_to_mask(segmentation, label_values)
        volumes[structure_name] = np.sum(structure_mask)
    return volumes