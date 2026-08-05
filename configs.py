"""Global configuration module for the BrainST project.

This module centralizes every path, naming convention, and hyperparameter
used across the codebase. It is imported everywhere as ``cfg`` (e.g.
``import configs as cfg``), so nothing defined here should be renamed
without updating every call site in the project.

Sections:
    - Base paths (data, models, temp directories).
    - Input CSV column names (raw dataset + preprocessing-derived columns).
    - Preprocessing constants (target shapes, pipeline steps).
    - Covariate encodings (diagnosis, sex).
    - ROI/structure definitions used to build the ROI-volume conditioning
      vector shared by BrainST-vol and BrainST-img.
    - Model architecture/checkpoint paths.
    - Training output paths.
    - Inference defaults (diffusion steps, classifier-free-guidance ratios).
"""

import json
import os

import src.utils.util_freesurfer_segmentation as ufs

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = "cuda:0"


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
PATH_BASE = os.path.dirname(os.path.abspath(__file__))
# PATH_TESTS = os.path.join(PATH_BASE, "tests")
PATH_DATA = os.path.join(PATH_BASE, "data")
PATH_DATA_TRAINING = os.path.join(PATH_DATA, "training")
PATH_DATA_GENERATION = os.path.join(PATH_DATA, "generation")
PATH_TEMP = os.path.join(PATH_BASE, "__temp__")

os.makedirs(PATH_TEMP, exist_ok=True)
os.makedirs(PATH_DATA_TRAINING, exist_ok=True)
os.makedirs(PATH_DATA_GENERATION, exist_ok=True)
# os.makedirs(PATH_TESTS, exist_ok=True)

# CONFIGS_PATH = os.path.join(PATH_BASE, "configs")

# MNI_IMG_PATH = os.path.join(PATH_BASE, "src/preprocessing/USLR/data/atlas/mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz")
# MNI_SEG_PATH = os.path.join(PATH_BASE, "src/preprocessing/USLR/data/atlas/mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz")


# ---------------------------------------------------------------------------
# Input CSV columns (raw dataset, provided by the user)
# ---------------------------------------------------------------------------
COL_SUBJECT_ID = "subject_id"
COL_SESSION_ID = "session_id"
COL_DATASET = "dataset"

COL_AGE = "age"
COL_SEX = "sex"
COL_DX = "dx"
COL_RAW_IMG_PATH = "t1w_raw_img_path"

# Optional columns
COL_RAW_SEG_PATH = "raw_seg_path"
COL_SPLIT = "split"


# ---------------------------------------------------------------------------
# Preprocessing-derived columns (added by `preprocess_training_data.py`)
# ---------------------------------------------------------------------------
COL_PREP_IMG_PATH = "prep_img_path"
COL_PREP_SEG_PATH = "prep_seg_path"
COL_PREP_LATENT_PATH = "latent_path"
COL_PREP_TISSUE_MASKS_PATH = "tissue_masks_path"
COL_PREP_OK = "prep_ok"


# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
SHAPE_PREP_IMG = (192, 256, 192)
SHAPE_LATENT = (4, 48, 64, 48)

PREPROCESSING_IMAGE_STEPS = ["super_resolution", "register", "segment_registered", "resize", "normalize"]


# ---------------------------------------------------------------------------
# Covariate encodings
# ---------------------------------------------------------------------------
DX_MAPPING = {"CN": 0, "MCI": 1, "AD": 2}
SEX_MAPPING = {"F": 0, "M": 1}


# ---------------------------------------------------------------------------
# ROI / structure configuration
# ---------------------------------------------------------------------------
# NOTE: the order of the structures is important, since it is used to index
# the latent space and the ROI-volume conditioning vector everywhere else
# in the codebase (BrainST-vol and BrainST-img both rely on this ordering).
STRUCTURE_INDEX_DICT = {
    "total": ufs.TOTAL_96,
    "surrounding_csf": ufs.SURROUNDING_CSF,
    "cortical_gm": ufs.CEREBRAL_CORTEX_96,
    "cerebral_wm": ufs.CEREBRAL_WM,
    "lateral_ventricles": ufs.LATERAL_VENTRICLES,
    "third_ventricle": ufs.THIRD_VENTRICLE,
    "fourth_ventricle": ufs.FOURTH_VENTRICLE,
    "thalamus": ufs.THALAMUS_V0 + ufs.THALAMUS_V1,
    "hippocampus": ufs.HIPPOCAMPUS,
    "amygdala": ufs.AMYGDALA,
    "putamen": ufs.PUTAMEN,
    "pallidum": ufs.PALLIDUM,
    "caudate": ufs.CAUDATE,
    "accumbens_area": ufs.ACCUMBENS_AREA,
    "ventral_dc": ufs.VENTRAL_DC,
    "cerebellum_gm": ufs.CEREBELLUM_GM,
    "cerebellum_wm": ufs.CEREBELLUM_WM,
    "brainstem": ufs.BRAINSTEM,
}

# Attention-mask variant of the structure->label mapping: identical to
# STRUCTURE_INDEX_DICT except that the lateral-ventricles mask also
# includes the (smaller) inferior lateral ventricles, since attention
# supervision benefits from the more complete anatomical mask.
STRUCTURE_MASKS_INDEX_DICT = {name: index for name, index in STRUCTURE_INDEX_DICT.items()}
STRUCTURE_MASKS_INDEX_DICT["lateral_ventricles"] = ufs.LATERAL_VENTRICLES + ufs.INFERIOR_LATERAL_VENTRICLES

NB_STRUCTURES = len(STRUCTURE_INDEX_DICT)

MASK_RESOLUTION_LIST = [1, 2, 4, 8]
NB_MASK_RESOLUTIONS = len(MASK_RESOLUTION_LIST)

STRUCTURE_NAME_LIST = list(STRUCTURE_INDEX_DICT.keys())
STRUCTURE_INDEX_LIST = list(STRUCTURE_INDEX_DICT.values())
# "_vol" suffixed names are the column names used in the training/generation
# CSVs and JSON condition dicts (e.g. "hippocampus" -> "hippocampus_vol").
STRUCTURE_NAME_LIST_VOL = [f"{vol}_vol" for vol in STRUCTURE_NAME_LIST]

COVARS_LIST = ["age", "sex", "dx"]
# Maps each "_vol" structure name to its FreeSurfer/SynthSeg label list,
# used when computing ROI volumes directly from a segmentation.
STRUCTURE_INDEX_VOL_DICT = {name: index for name, index in zip(STRUCTURE_NAME_LIST_VOL, STRUCTURE_INDEX_LIST)}


# ---------------------------------------------------------------------------
# Model architecture / checkpoint paths
# ---------------------------------------------------------------------------
PATH_MODELS = os.path.join(PATH_BASE, "models")

PATH_MODELS_ARCHITECTURES = os.path.join(PATH_MODELS, "architectures")
PATH_AUTOENCODER_ARCHITECTURE = os.path.join(PATH_MODELS_ARCHITECTURES, "autoencoder.pt")
PATH_BRAINST_IMG_ARCHITECTURE = os.path.join(PATH_MODELS_ARCHITECTURES, "brainst_img.json")
PATH_BRAINST_VOL_ARCHITECTURE = os.path.join(PATH_MODELS_ARCHITECTURES, "brainst_vol.json")

# Patch the architecture JSONs with the number of conditioning structures,
# so the config files themselves don't need to hardcode NB_STRUCTURES.
ARCHITECTURE_BRAINST_IMG = json.load(open(PATH_BRAINST_IMG_ARCHITECTURE, "r"))
ARCHITECTURE_BRAINST_IMG["conditions_model"]["num_conditions"] = NB_STRUCTURES

ARCHITECTURE_BRAINST_VOL = json.load(open(PATH_BRAINST_VOL_ARCHITECTURE, "r"))
ARCHITECTURE_BRAINST_VOL["diffusion_mlp"]["d_in"] = NB_STRUCTURES

# Normalization parameters (fitted z-score standardizer for age + ROI volumes)
PATH_NORMALIZATION_PARAMS = os.path.join(PATH_MODELS, "normalization", "zscore_standardizer_params.json")

# Pretrained checkpoints
PATH_MODELS_WEIGHTS = os.path.join(PATH_MODELS, "weights")
PATH_AUTOENCODER_CHK = os.path.join(PATH_MODELS_WEIGHTS, "autoencoder_epoch273.pt")
PATH_BRAINST_IMG_CHK = os.path.join(PATH_MODELS_WEIGHTS, "brainst_img_step200000.pt")
PATH_BRAINST_VOL_CHK = os.path.join(PATH_MODELS_WEIGHTS, "brainst_vol_step2000.pt")


# ---------------------------------------------------------------------------
# Training output paths
# ---------------------------------------------------------------------------
PATH_MODELS_TRAINING = os.path.join(PATH_MODELS, "training")


# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
BRAINST_IMG_NUM_INFERENCE_STEPS = 50
BRAINST_VOL_NUM_INFERENCE_STEPS = 50
BRAINST_IMG_FREE_GUIDANCE_RATIO = 2.0
BRAINST_VOL_FREE_GUIDANCE_RATIO = 1.0