"""Wrapper around FreeSurfer's ``mri_synthsr`` command-line tool."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def save_synthSR(
    img_path_name: str,
    out_path_name: str,
    verify: bool = False,
    verbose: bool = False,
    nthreads: int | None = 4,
) -> bool | None:
    """Run ``mri_synthsr`` to super-resolve/denoise a brain MRI to 1mm MNI-like space.

    Note:
        SynthSR's output is cast to ``UINT8`` intensities (a known
        characteristic of the tool itself, not this wrapper).

    Args:
        img_path_name: Path to the input image.
        out_path_name: Path to write the super-resolved output to.
        verify: If True and ``out_path_name`` already exists, skip
            running SynthSR entirely.
        verbose: If True, log a confirmation message on success.
        nthreads: Number of CPU threads to use.

    Returns:
        ``True`` if skipped because the output already existed, or if the
        subprocess completed successfully; ``False`` if the subprocess
        failed.

    Side Effects:
        Runs ``mri_synthsr`` as a subprocess; writes ``out_path_name``.

    Example:
        >>> save_synthSR(
        ...     img_path_name="sub-01_T1w.nii.gz",
        ...     out_path_name="sub-01_T1w_synthsr.nii.gz",
        ...     verify=True,
        ... )
    """
    if verify and os.path.exists(out_path_name):
        logger.info("SynthSR already done for: %s\nfile in: %s", img_path_name, out_path_name)
        return True

    command = ["mri_synthsr", "--i", img_path_name, "--o", out_path_name, "--threads", str(nthreads)]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            logger.info("SynthSR done saved in: %s", out_path_name)
        return True
    logger.error("mri_synthsr failed for %s: %s", img_path_name, result.stderr)
    return False