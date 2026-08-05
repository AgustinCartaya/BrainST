"""Wrapper around FreeSurfer's ``mri_synthseg`` command-line tool."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def save_synthseg_segmentation(
    img_path_name: str,
    out_path_name: str,
    verify: bool = False,
    verbose: bool = False,
    robust: bool = True,
    cortical_parcelation: bool = True,
    path_name_vol_csv: str | None = None,
    path_name_resampled_img: str | None = None,
    path_name_post_prob_img: str | None = None,
    nthreads: int | None = 4,
) -> bool:
    """Run ``mri_synthseg`` to segment a brain MRI.

    The number of output structure labels depends on ``robust`` and
    ``cortical_parcelation``: 31 labels (robust + cortical parcellation,
    the default), 28 labels (cortical parcellation without ``--robust``),
    or 23 labels (neither flag).

    Args:
        img_path_name: Path to the input image.
        out_path_name: Path to write the output segmentation to.
        verify: If True and ``out_path_name`` already exists, skip
            running SynthSeg entirely.
        verbose: If True, log a confirmation message on success.
        robust: If True, pass ``--robust`` (slower but more robust to
            image artifacts).
        cortical_parcelation: If True, pass ``--parc`` to additionally
            parcellate the cortex.
        path_name_vol_csv: Optional path to also write a per-structure
            volume CSV (``--vol``).
        path_name_resampled_img: Optional path to also write the
            resampled input image (``--resample``).
        path_name_post_prob_img: Optional path to also write posterior
            probability maps (``--post``).
        nthreads: Number of CPU threads to use. If ``None``, the
            ``--threads`` flag is omitted (tool default is used).

    Returns:
        ``True`` if the segmentation was skipped because ``out_path_name``
        already existed (and ``verify=True``), or if ``mri_synthseg``
        completed successfully; ``False`` if the subprocess failed.

    Side Effects:
        Runs ``mri_synthseg`` as a subprocess; writes ``out_path_name``
        (and any of the optional outputs requested) to disk.

    Example:
        >>> save_synthseg_segmentation(
        ...     img_path_name="raw_t1w.nii.gz",
        ...     out_path_name="raw_t1w_seg.nii.gz",
        ...     verify=True,
        ...     verbose=True,
        ...     robust=True,
        ...     cortical_parcelation=True,
        ... )
    """
    if verify and os.path.exists(out_path_name):
        logger.info("synthseg already done for: %s\nfile in: %s", img_path_name, out_path_name)
        return True

    command = ["mri_synthseg", "--i", img_path_name, "--o", out_path_name, "--cpu"]

    if robust:
        command.append("--robust")
    if cortical_parcelation:
        command.append("--parc")
    if path_name_vol_csv is not None:
        command.extend(["--vol", path_name_vol_csv])
    if path_name_resampled_img is not None:
        command.extend(["--resample", path_name_resampled_img])
    if path_name_post_prob_img is not None:
        command.extend(["--post", path_name_post_prob_img])
    if nthreads is not None:
        command.extend(["--threads", str(nthreads)])

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            logger.info("synthseg done saved in: %s", out_path_name)
        return True
    logger.error("mri_synthseg failed for %s: %s", img_path_name, result.stderr)
    return False