"""Batch preprocessing pipeline for BrainST training data.

Given a CSV of raw subject/session MRI paths, this script runs the full
training-data preparation pipeline:

    1. Preprocess each raw image (super-resolution, MNI registration,
       segmentation, resize, intensity normalization).
    2. Encode each preprocessed image into an autoencoder latent.
    3. Create multi-resolution tissue attention masks per structure.
    4. Verify all expected outputs exist and assemble a preprocessed-data CSV.
    5. Compute per-subject brain-structure volumes from the segmentations.
    6. Fit ICV + z-score normalization parameters and produce the final
       training CSV (with standardized volumes and integer-coded
       categorical covariates).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

import configs as cfg
import src.utils.functions as fc
import src.utils.nifti_functions as nfc
from src.brainst_img.autoencoder_declaration import AutoencoderPrediction
from src.preprocessing import create_att_masks, preprocess_images
from src.utils import data_normalization, prep_volumes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def create_preprocess_training_path(row: pd.Series, base_output_path: str) -> str:
    """Build (and create) the per-subject/session output directory.

    Args:
        row: A row from the input CSV, expected to contain
            ``cfg.COL_SUBJECT_ID``, ``cfg.COL_SESSION_ID``, and
            ``cfg.COL_DATASET``.
        base_output_path: Root directory for all preprocessed outputs.

    Returns:
        Path ``{base_output_path}/{dataset}/{subject_id}/{session_id}``.

    Side Effects:
        Creates the directory (and any missing parents) if it does not
        already exist.
    """
    subject_id = row[cfg.COL_SUBJECT_ID]
    session_id = row[cfg.COL_SESSION_ID]
    dataset = row[cfg.COL_DATASET]

    output_path = os.path.join(base_output_path, dataset, subject_id, session_id)
    os.makedirs(output_path, exist_ok=True)
    return output_path


# ---------------------------------------------------------------------------
# Step 1: Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_training_images(
    row: pd.Series,
    output_path: str,
    shape_prep_img: tuple[int, int, int],
    steps: list[str],
    verify: bool = True,
    verbose: bool = False,
) -> None:
    """Preprocess a single subject/session's raw image (one row of the input CSV).

    Args:
        row: A row from the input CSV.
        output_path: Root output directory (a per-subject/session
            subdirectory is created under it).
        shape_prep_img: Target shape for the preprocessed image.
        steps: Preprocessing stages to run (see
            :func:`src.preprocessing.preprocess_images.preprocess_image`).
        verify: If True, skip stages whose outputs already exist.
        verbose: If True, enable verbose logging in sub-steps that
            support it.

    Side Effects:
        Writes preprocessed image/segmentation files under
        ``output_path``.
    """
    input_path_name = row[cfg.COL_RAW_IMG_PATH]
    output_path = create_preprocess_training_path(row, output_path)
    if cfg.COL_RAW_SEG_PATH in row and pd.notna(row[cfg.COL_RAW_SEG_PATH]):
        seg_raw_path_name = row[cfg.COL_RAW_SEG_PATH]
    else:
        seg_raw_path_name = None

    preprocess_images.preprocess_image(
        input_path_name, output_path, shape_prep_img, steps, seg_raw_path_name, verify=verify, verbose=verbose
    )


def multi_preprocess_training_images(
    data_df: pd.DataFrame,
    output_path: str,
    shape_prep_img: tuple[int, int, int],
    steps: list[str],
    num_workers: int = 2,
    verify: bool = True,
    verbose: bool = False,
) -> None:
    """Preprocess every row of ``data_df`` in parallel.

    Args:
        data_df: Input CSV loaded as a DataFrame; one row per
            subject/session.
        output_path: Root output directory.
        shape_prep_img: Target shape for preprocessed images.
        steps: Preprocessing stages to run.
        num_workers: Number of worker processes
            (``mp.cpu_count()`` if falsy, e.g. 0 or None).
        verify: If True, skip stages whose outputs already exist.
        verbose: If True, enable verbose logging in sub-steps.

    Side Effects:
        Writes preprocessed files for every row; shows a progress bar.
    """
    rows = [row for _, row in data_df.iterrows()]
    process_fn = partial(
        preprocess_training_images,
        output_path=output_path,
        shape_prep_img=shape_prep_img,
        steps=steps,
        verify=verify,
        verbose=verbose,
    )
    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_fn, rows), total=len(rows)))


# ---------------------------------------------------------------------------
# Step 2: Latent creation
# ---------------------------------------------------------------------------

def compute_training_image_latents(data_df: pd.DataFrame, output_path: str) -> None:
    """Encode every preprocessed image into an autoencoder latent and save it as ``.npy``.

    Args:
        data_df: Input CSV loaded as a DataFrame.
        output_path: Root output directory (must match the one used in
            :func:`multi_preprocess_training_images`).

    Side Effects:
        Loads the autoencoder checkpoint onto ``cfg.DEVICE`` and writes
        one ``{img_name}_latent.npy`` file per row (skipping rows whose
        latent already exists).
    """
    autoencoder = AutoencoderPrediction(cfg.PATH_AUTOENCODER_CHK, cfg.DEVICE, half=True)
    for _, row in data_df.iterrows():
        preprocessed_img_path = create_preprocess_training_path(row, output_path)
        img_name = fc.get_img_name(row[cfg.COL_RAW_IMG_PATH])
        latent_path_name = os.path.join(preprocessed_img_path, f"{img_name}_latent.npy")

        if os.path.exists(latent_path_name):
            logger.info("Skipping latent computation for %s, already exists.", latent_path_name)
        else:
            preprocessed_img_path_name = os.path.join(preprocessed_img_path, f"{img_name}_preprocessed.nii.gz")
            img, _affine = nfc.load_nifti(preprocessed_img_path_name)
            latent = autoencoder.encode(img).squeeze().cpu().numpy()
            np.save(latent_path_name, latent)


# ---------------------------------------------------------------------------
# Step 3: Tissue mask creation
# ---------------------------------------------------------------------------

def create_tissue_masks(row: pd.Series, base_output_path: str, shape_prep_img: tuple[int, int, int], verify: bool = True) -> None:
    """Create multi-resolution tissue attention masks for one subject/session.

    Args:
        row: A row from the input CSV.
        base_output_path: Root output directory.
        shape_prep_img: Preprocessed image shape (currently unused inside
            this function body but kept for interface consistency with
            :func:`multi_create_tissue_masks`'s partial application).
        verify: If True, skip mask creation when the expected outputs
            already exist.

    Side Effects:
        Writes tissue mask ``.npy`` files under
        ``{subject_output_path}/tissue_masks``.
    """
    subject_output_path = create_preprocess_training_path(row, base_output_path)
    tissue_masks_path = os.path.join(subject_output_path, "tissue_masks")
    img_name = fc.get_img_name(row[cfg.COL_RAW_IMG_PATH])

    seg_path_name = os.path.join(subject_output_path, f"{img_name}_preprocessed_seg.nii.gz")
    create_att_masks.create_tissue_masks(
        seg_path_name, tissue_masks_path, cfg.STRUCTURE_MASKS_INDEX_DICT, cfg.MASK_RESOLUTION_LIST, verify=verify
    )


def multi_create_tissue_masks(
    data_df: pd.DataFrame,
    base_output_path: str,
    shape_prep_img: tuple[int, int, int],
    num_workers: int = 2,
    verify: bool = True,
) -> None:
    """Create tissue masks for every row of ``data_df`` in parallel.

    Args:
        data_df: Input CSV loaded as a DataFrame.
        base_output_path: Root output directory.
        shape_prep_img: Preprocessed image shape.
        num_workers: Number of worker processes
            (``mp.cpu_count()`` if falsy).
        verify: If True, skip rows whose tissue masks already exist.

    Side Effects:
        Writes tissue mask files for every row; shows a progress bar.
    """
    rows = [row for _, row in data_df.iterrows()]
    process_fn = partial(create_tissue_masks, base_output_path=base_output_path, shape_prep_img=shape_prep_img)
    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_fn, rows), total=len(rows)))


# ---------------------------------------------------------------------------
# Step 4: Training CSV creation and checking
# ---------------------------------------------------------------------------

def check_subject_preprocessed_data(
    row: pd.Series, base_output_path: str
) -> tuple[bool, str | None, str | None, str | None, str | None]:
    """Check whether all expected preprocessed outputs exist for one subject/session.

    Args:
        row: A row from the input CSV.
        base_output_path: Root output directory.

    Returns:
        A tuple ``(all_ok, preprocessed_img_path, preprocessed_seg_path,
        latent_path_name, tissue_masks_path_name)``. Any path whose
        corresponding output is missing is returned as ``None``, and
        ``all_ok`` is ``False`` if any output is missing.
    """
    subject_output_path = create_preprocess_training_path(row, base_output_path)
    img_name = fc.get_img_name(row[cfg.COL_RAW_IMG_PATH])

    all_ok = True

    preprocessed_img_path = os.path.join(subject_output_path, f"{img_name}_preprocessed.nii.gz")
    if not os.path.exists(preprocessed_img_path):
        all_ok = False
        preprocessed_img_path = None

    preprocessed_seg_path = os.path.join(subject_output_path, f"{img_name}_preprocessed_seg.nii.gz")
    if not os.path.exists(preprocessed_seg_path):
        all_ok = False
        preprocessed_seg_path = None

    latent_path_name = os.path.join(subject_output_path, f"{img_name}_latent.npy")
    if not os.path.exists(latent_path_name):
        all_ok = False
        latent_path_name = None

    tissue_masks_path_name = os.path.join(subject_output_path, "tissue_masks")
    if not os.path.exists(tissue_masks_path_name) or len(os.listdir(tissue_masks_path_name)) < cfg.NB_STRUCTURES * cfg.NB_MASK_RESOLUTIONS:
        all_ok = False
        tissue_masks_path_name = None

    return all_ok, preprocessed_img_path, preprocessed_seg_path, latent_path_name, tissue_masks_path_name


def create_training_csv(data_df: pd.DataFrame, output_preprocessed_path: str, output_csv_path_name: str) -> None:
    """Verify preprocessing outputs for every row and write an augmented preprocessed-data CSV.

    Args:
        data_df: Input CSV loaded as a DataFrame.
        output_preprocessed_path: Root output directory to check outputs
            under.
        output_csv_path_name: Destination path for the resulting CSV.

    Side Effects:
        Writes ``output_csv_path_name``, containing every original column
        from ``data_df`` plus ``cfg.COL_PREP_OK``, ``cfg.COL_PREP_IMG_PATH``,
        ``cfg.COL_PREP_SEG_PATH``, ``cfg.COL_PREP_LATENT_PATH``, and
        ``cfg.COL_PREP_TISSUE_MASKS_PATH``, in the same row order as
        ``data_df``.
    """
    # Check all rows in parallel, preserving input order via `imap` (not `imap_unordered`).
    rows = [row for _, row in data_df.iterrows()]
    process_fn = partial(check_subject_preprocessed_data, base_output_path=output_preprocessed_path)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_fn, rows), total=len(rows)))

    new_data = []
    for row, result in zip(rows, results):
        all_ok, preprocessed_img_path, preprocessed_seg_path, latent_path_name, tissue_masks_path_name = result
        new_row = row.to_dict()
        new_row[cfg.COL_PREP_OK] = all_ok
        new_row[cfg.COL_PREP_IMG_PATH] = preprocessed_img_path
        new_row[cfg.COL_PREP_SEG_PATH] = preprocessed_seg_path
        new_row[cfg.COL_PREP_LATENT_PATH] = latent_path_name
        new_row[cfg.COL_PREP_TISSUE_MASKS_PATH] = tissue_masks_path_name
        new_data.append(new_row)

    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_csv(output_csv_path_name, index=False)


def compute_brain_statistics(row: dict, structure_name_index_dict: dict) -> list:
    """Compute per-structure voxel-count volumes for one subject/session.

    Args:
        row: A row (as a dict) from the preprocessed-data CSV, expected
            to contain ``cfg.COL_SUBJECT_ID``, ``cfg.COL_SESSION_ID``, and
            ``cfg.COL_PREP_SEG_PATH``.
        structure_name_index_dict: Mapping from structure name to label
            values (see :func:`src.utils.prep_volumes.get_volumes`).

    Returns:
        A list ``[subject_id, session_id, *structure_volumes]``. If the
        segmentation is missing/unavailable or an error occurs while
        loading it, structure volumes are meant to be filled with ``-1``
        placeholders.
    """
    row_res = [row[cfg.COL_SUBJECT_ID], row[cfg.COL_SESSION_ID]]
    try:
        if not pd.isna(row[cfg.COL_PREP_SEG_PATH]) and os.path.exists(row[cfg.COL_PREP_SEG_PATH]):
            seg, _affine = nfc.load_nifti(row[cfg.COL_PREP_SEG_PATH])
            structure_volumes = prep_volumes.get_volumes(seg, structure_name_index_dict)
            row_res += list(structure_volumes.values())
        else:
            row_res += [-1] * len(structure_name_index_dict)
    except Exception as error:
        logger.error("Error processing %s - %s", row[cfg.COL_SUBJECT_ID], error)
        row_res += [-1] * len(structure_name_index_dict)
    return row_res


def multi_compute_brain_statistics(
    preprocessed_data_df: pd.DataFrame,
    structures_dict: dict,
    output_csv_path_name: str,
    num_workers: int = 2,
) -> None:
    """Compute brain-structure volumes for every subject/session in parallel.

    Args:
        preprocessed_data_df: Preprocessed-data DataFrame (output of
            :func:`create_training_csv`).
        structures_dict: Mapping from structure name to label values.
        output_csv_path_name: Destination path for the resulting
            brain-statistics CSV.
        num_workers: Unused directly here; the pool size is hardcoded to
            16 (see implementation) rather than this parameter — kept for
            interface consistency with the other ``multi_*`` functions.

    Side Effects:
        Writes ``output_csv_path_name`` with columns
        ``[subject_id, session_id, *cfg.STRUCTURE_NAME_LIST_VOL]``.
    """
    rows = preprocessed_data_df.to_dict(orient="records")

    compute_fn = partial(compute_brain_statistics, structure_name_index_dict=structures_dict)
    with mp.Pool(16) as pool:
        results = list(tqdm(pool.imap(compute_fn, rows), total=len(rows), desc="processing subjects"))

    columns = [cfg.COL_SUBJECT_ID, cfg.COL_SESSION_ID] + cfg.STRUCTURE_NAME_LIST_VOL
    data_frame = pd.DataFrame(results, columns=columns)
    os.makedirs(os.path.dirname(output_csv_path_name), exist_ok=True)
    data_frame.to_csv(output_csv_path_name, index=False)


def compute_normalization_params(
    preprocessed_data_df: pd.DataFrame,
    brain_statistics_df: pd.DataFrame,
    output_normalization_params_path_name: str,
    output_normalized_df_path_name: str,
) -> None:
    """Fit ICV + z-score normalization and produce the final training CSV.

    Args:
        preprocessed_data_df: Preprocessed-data DataFrame (paths + raw
            covariates).
        brain_statistics_df: Per-structure voxel-count volumes (output of
            :func:`multi_compute_brain_statistics`).
        output_normalization_params_path_name: Destination path for the
            fitted standardizer parameters (JSON).
        output_normalized_df_path_name: Destination path for the final
            training CSV (standardized volumes + integer-coded
            categoricals).

    Side Effects:
        Writes both output files. Categorical columns (``cfg.COL_DX``,
        ``cfg.COL_SEX``) are mapped to integers in-place if not already
        numeric.
    """
    # normalize brain statistics by ICV
    brain_statistics_normalized_icv_df = data_normalization.normalize_by_icv(
        brain_statistics_df, cfg.STRUCTURE_NAME_LIST_VOL, icv_column="total_vol", percentage=False
    )

    # combine preprocessed data and normalized brain statistics
    complete_df = pd.merge(
        preprocessed_data_df, brain_statistics_normalized_icv_df, on=[cfg.COL_SUBJECT_ID, cfg.COL_SESSION_ID], how="inner"
    )

    # create a standarizer and fit it to the training data (if available) or the entire dataset
    standarizer = data_normalization.ZScoreStandardizerBrainStructures(
        [cfg.COL_AGE] + cfg.STRUCTURE_NAME_LIST_VOL, robust=False
    )

    if cfg.COL_SPLIT in complete_df.columns and "train" in complete_df[cfg.COL_SPLIT].values:
        train_df = complete_df[complete_df[cfg.COL_SPLIT] == "train"]
    else:
        logger.warning(
            "%s column not found or no 'train' split in the dataframe. "
            "Using the entire dataframe for fitting the standardizer.",
            cfg.COL_SPLIT,
        )
        train_df = complete_df

    standarizer.fit(train_df)
    complete_df = standarizer.transform(complete_df)

    # map categorical columns to integers (skip if already numeric)
    if complete_df[cfg.COL_DX].dtype == object:
        complete_df[cfg.COL_DX] = complete_df[cfg.COL_DX].map(cfg.DX_MAPPING).astype(int)
    if complete_df[cfg.COL_SEX].dtype == object:
        complete_df[cfg.COL_SEX] = complete_df[cfg.COL_SEX].map(cfg.SEX_MAPPING).astype(int)

    os.makedirs(os.path.dirname(output_normalization_params_path_name), exist_ok=True)
    os.makedirs(os.path.dirname(output_normalized_df_path_name), exist_ok=True)
    standarizer.save_params(output_normalization_params_path_name)
    complete_df.to_csv(output_normalized_df_path_name, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Build and parse the command-line interface for this script.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess training data for BrainST.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help=(
            "Path to the input CSV file containing raw image paths.\n"
            f"The CSV should have columns: '{cfg.COL_SUBJECT_ID}', '{cfg.COL_SESSION_ID}', '{cfg.COL_AGE}', "
            f"'{cfg.COL_SEX}', {cfg.COL_DX}, '{cfg.COL_RAW_IMG_PATH}', and optionally '{cfg.COL_RAW_SEG_PATH}'."
        ),
    )
    parser.add_argument(
        "--prep_output_path", type=str, required=True, help="Path to the output directory where preprocessed data will be saved."
    )
    parser.add_argument("--threads", type=int, required=True, help="Number of threads to use for preprocessing.")

    args = parser.parse_args()

    return args


def create_test_args() -> argparse.Namespace:
    """Build a hard-coded ``argparse.Namespace`` for local/IDE debugging.

    This is what ``if __name__ == "__main__":`` actually uses (instead of
    :func:`parse_args`) to drive the pipeline.

    Returns:
        A hard-coded ``argparse.Namespace`` with ``input_csv``,
        ``prep_output_path``, and ``threads``.
    """

    csv_path = os.path.join(cfg.PATH_DATA_TRAINING, "example", "raw_data.csv")
    output_path = os.path.join(cfg.PATH_DATA_TRAINING, "images", "preprocessed")
    num_threads = 6

    args = argparse.Namespace(
        input_csv=csv_path,
        prep_output_path=output_path,
        threads=num_threads,
    )
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # args = create_test_args()
    args = parse_args()

    data_df = pd.read_csv(args.input_csv)
    prep_output_path = args.prep_output_path
    num_workers = args.threads

    logger.info("Starting preprocessing of training data with %d threads...", num_workers)

    logger.info("---- Step 1: Preprocessing training images...")
    multi_preprocess_training_images(
        data_df, output_path=prep_output_path, shape_prep_img=cfg.SHAPE_PREP_IMG, steps=cfg.PREPROCESSING_IMAGE_STEPS,
        num_workers=num_workers,
    )

    logger.info("---- Step 2: Computing training image latents...")
    compute_training_image_latents(data_df, output_path=prep_output_path)

    logger.info("---- Step 3: Creating tissue masks...")
    multi_create_tissue_masks(
        data_df, base_output_path=prep_output_path, shape_prep_img=cfg.SHAPE_PREP_IMG, num_workers=num_workers
    )

    logger.info("---- Step 4: Verifying preprocessed data....")
    preprocessed_data_df_path_name = os.path.join(prep_output_path, "preprocessed_data.csv")
    create_training_csv(data_df, output_preprocessed_path=prep_output_path, output_csv_path_name=preprocessed_data_df_path_name)

    logger.info("---- Step 5: Computing brain statistics...")
    preprocessed_data_df = pd.read_csv(preprocessed_data_df_path_name)
    brain_statistics_df_path_name = os.path.join(prep_output_path, "brain_statistics.csv")
    multi_compute_brain_statistics(
        preprocessed_data_df, structures_dict=cfg.STRUCTURE_INDEX_VOL_DICT, output_csv_path_name=brain_statistics_df_path_name,
        num_workers=num_workers,
    )

    logger.info("---- Step 6: Computing normalization parameters and creating training data...")
    brain_statistics_df = pd.read_csv(brain_statistics_df_path_name)
    output_normalization_params_path_name = os.path.join(prep_output_path, "normalization_params.json")
    output_normalized_df_path_name = os.path.join(prep_output_path, "training_data.csv")
    compute_normalization_params(
        preprocessed_data_df, brain_statistics_df, output_normalization_params_path_name, output_normalized_df_path_name
    )

    nb_success = preprocessed_data_df[cfg.COL_PREP_OK].sum()
    nb_total = len(preprocessed_data_df)
    logger.info("Preprocessing complete! %d/%d subjects successfully preprocessed.", nb_success, nb_total)
    logger.info("---------------- Summary of outputs --------")
    logger.info("Preprocessed data saved to:\t\t%s", prep_output_path)
    logger.info("Preprocessed csv saved to:\t\t%s", preprocessed_data_df_path_name)
    logger.info("Brain stats csv saved to:\t\t%s", brain_statistics_df_path_name)
    logger.info("Training csv saved to:\t\t\t%s", output_normalized_df_path_name)
    logger.info("Normalization params json saved to:\t%s", output_normalization_params_path_name)