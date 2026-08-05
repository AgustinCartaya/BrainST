# BrainST: Structural Volume–Guided Diffusion Modelling for Counterfactual Brain MRI and Longitudinal Prediction

![BrainST](brainst_intro.gif)
- **BrainST - Ode to Joy with sound (MP4):** [ode_to_joy.mp4](tutorials/brain_concert/data/outputs/concert_videos/ode_to_joy.mp4)
- **BrainST - Fur Elise with sound (MP4):** [fur_elise.mp4](tutorials/brain_concert/data/outputs/concert_videos/fur_elise.mp4)
- **BrainST - Note test with sound (MP4):** [note_test.mp4](tutorials/brain_concert/data/outputs/concert_videos/note_test.mp4)


**Author:** Agustin Cartaya Lathulerie

BrainST is a diffusion-based framework for the **controlled synthesis, anatomical transformation, and longitudinal prediction of T1-weighted brain MRI**, trained entirely on cross-sectional data. It enables fine-grained, region-specific control by conditioning image generation on volumetric measurements of **18 brain regions of interest (ROIs)**, while preserving anatomical plausibility through a conditioning alignment penalty.

The framework couples two diffusion models:

- **BrainST-vol** — a lightweight 1D diffusion model that predicts a subject's ROI-volume profile from demographic/clinical covariates (age, sex, diagnosis).
- **BrainST-img** — a 3D latent diffusion model (MAISI-style UNet + KL-autoencoder) that generates/transforms a brain MRI conditioned on an ROI-volume profile via cross-attention.

> This repository accompanies the associated research paper (see [Citation](#citation)).

---

## Table of Contents

- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pretrained Models](#pretrained-models)
- [Generation / Inference](#generation--inference)
- [Training](#training)
  - [1. Preprocessing Training Data](#1-preprocessing-training-data)
  - [2. Training](#2-training)
- [Configuration](#configuration)
- [Bonus: The "Brain Concert" Demo](#bonus-the-brain-concert-demo)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Key Features

- **Cross-sectional MRI generation** from manually specified or automatically predicted ROI volumes.
- **Localized anatomical transformations** of existing images while preserving subject-specific anatomy (via null-text inversion).
- **Longitudinal prediction** of brain changes associated with healthy aging or neurodegenerative disease (e.g. Alzheimer's disease), driven purely by cross-sectional training data.
- **Automatic ROI-volume prediction** from demographic and cognitive covariates (age, sex, diagnosis).
- **Counterfactual image synthesis** with precise, region-specific volumetric control over 18 anatomical structures.
- **Reusable preprocessing pipeline** (super-resolution → MNI registration → segmentation → resizing → intensity normalization) built on FreeSurfer's SynthSR/SynthSeg and a lightweight in-house registration module (USLR-lite).

---

## Requirements

### Operating System

- **Linux** is required. The preprocessing pipeline shells out to FreeSurfer command-line tools (`mri_synthseg`, `mri_synthsr`), which are only distributed for Linux/macOS. GPU training/inference has only been validated on Linux with an NVIDIA GPU.

### Software

| Component | Version / Notes |
|---|---|
| Python | 3.11 |
| CUDA-capable GPU | Recommended: 24 GB VRAM for training. For inference, at least 12 GB VRAM is recommended |
| [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (≥ 7.3, providing `mri_synthseg` / `mri_synthsr`) | Required for preprocessing raw MRIs (super-resolution and segmentation) |
| Conda / Miniconda | Recommended for environment management |

### Python Dependencies

All Python dependencies are listed in [`requirements.txt`](requirements.txt) and include (non-exhaustively): `torch`, `torchvision`, `monai`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `scikit-image`, `nibabel`, `SimpleITK`, `matplotlib`, `opencv-python`, `tensorboard`, `tqdm`, and, for the optional `brain_concert` demo, `pydub` and `moviepy`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AgustinCartaya/BrainST.git
cd BrainST

# 2. Create and activate a Python 3.11 conda environment
conda create --name BrainST python=3.11
conda activate BrainST

# 3. Install Python dependencies
# 3.1 Install a compatible version of PyTorch for your system. 
# See https://pytorch.org/get-started/locally/
# Example for CUDA 12.9:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 3.2 Install Python dependencies
pip install -r requirements.txt

# 4. Install FreeSurfer (external dependency, required for preprocessing)
#    Follow the official installation guide:
#    https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
#    Then make sure the following are set (typically in your shell rc file):
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

After installation, verify that the FreeSurfer command-line tools are on your `PATH`:

```bash
which mri_synthseg mri_synthsr
```

---

## Pretrained Models

`configs.py` expects the following pretrained checkpoints and architecture files under `models/` (relative to the repository root):

```
models/
├── architectures/
│   ├── autoencoder.json
│   ├── brainst_img.json
│   └── brainst_vol.json
├── weights/
│   ├── autoencoder_epoch273.pt
│   ├── brainst_img_step200000.pt
│   └── brainst_vol_step2000.pt
└── normalization/
    └── zscore_standardizer_params.json
```

> **TODO:** Add a download link / release for the pretrained checkpoints once they are published. Until then, checkpoints must be produced locally using the [training](#2-training) scripts described below.

---

## Generation / Inference

`main_generation.py` supports three generation modes:

### a) Synthesis — generate a brand-new brain image

```bash
# From explicit target ROI volumes (standardized z-score scale)
python main_generation.py \
    --generation_type synthesis \
    --path_target_roi_volumes ./data/generation/inputs/target_vol_standardized.json \
    --target_roi_volumes_scale standardized \
    --output_dir ./data/generation/outputs/ \
    --diffusion_steps 50 \
    --seed 2

# From target demographic covariates (age/sex/diagnosis)
python main_generation.py \
    --generation_type synthesis \
    --target_age 75 \
    --target_sex M \
    --target_dx CN \
    --output_dir ./data/generation/outputs/

# From a reference segmentation/image (copy its ROI-volume profile)
python main_generation.py \
    --generation_type synthesis \
    --path_reference_image ./data/generation/inputs/reference_image.nii.gz \
    --output_dir ./data/generation/outputs/
```

### b) Transformation — morph a real image toward a target ROI profile

```bash
# From explicit target ROI volumes (standardized z-score scale)
python main_generation.py \
    --generation_type transformation \
    --path_image ./data/generation/inputs/basal_image.nii.gz \
    --path_target_roi_volumes ./data/generation/inputs/target_vol_standardized.json \
    --target_roi_volumes_scale standardized \
    --output_dir ./data/generation/outputs/ \
    --diffusion_steps 50

# From a reference segmentation/image (copy its ROI-volume profile)
python main_generation.py \
    --generation_type transformation \
    --path_image ./data/generation/inputs/basal_image.nii.gz \
    --path_reference_image ./data/generation/inputs/reference_image.nii.gz \
    --output_dir ./data/generation/outputs/ \
    --diffusion_steps 50
```

### c) Longitudinal Prediction — simulate aging / disease progression

```bash
python main_generation.py \
    --generation_type longitudinal \
    --path_image ./data/generation/inputs/basal_image.nii.gz \
    --initial_age 79 \
    --initial_sex M \
    --initial_dx CN \
    --target_age 90 \
    --target_dx CN \
    --output_dir ./data/generation/outputs/ \
    --diffusion_steps 50
```

### CLI Argument Reference

| Argument | Applies to | Description |
|---|---|---|
| `--generation_type` | all | `synthesis` \| `transformation` \| `longitudinal` (required) |
| `--path_target_roi_volumes` | synthesis, transformation | JSON file with target ROI-volume dict |
| `--target_roi_volumes_scale` | synthesis, transformation | `standardized` (z-score) or `mm3` (raw volumes) |
| `--target_age`, `--target_sex`, `--target_dx` | synthesis, longitudinal | Target demographic covariates |
| `--initial_age`, `--initial_sex`, `--initial_dx` | longitudinal | Subject's covariates at the source timepoint |
| `--path_image`, `--path_segmentation` | transformation, longitudinal | Source image / segmentation |
| `--path_reference_image`, `--path_reference_segmentation` | synthesis, transformation | Reference to copy an ROI-volume profile from |
| `--apply_preprocessing` | transformation, longitudinal | Run the full preprocessing pipeline on `--path_image` before generation |
| `--diffusion_steps` | all | Number of DDIM denoising steps (default: 50) |
| `--seed` | synthesis | Random seed for the initial noise (default: 2) |
| `--output_dir`, `--output_name` | all | Output location for the generated NIfTI file |

---

## Training

### 1. Preprocessing Training Data

BrainST trains exclusively on **cross-sectional** T1-weighted MRI data. The batch preprocessing pipeline (`preprocess_training_data.py`) takes a single input CSV describing the raw dataset and produces everything needed for training.

#### 1.1 Input CSV

Create a CSV file with one row per subject/session, containing at minimum the following columns:

| Column | Description |
|---|---|
| `subject_id` | Unique subject identifier |
| `session_id` | Session identifier (expected format `m<months>`, e.g. `m000`, `m024`, used for longitudinal gap queries) |
| `age` | Subject age at the session |
| `sex` | `M` or `F` |
| `dx` | Diagnosis: `CN`, `MCI`, or `AD` |
| `t1w_raw_img_path` | Absolute path to the raw T1-weighted NIfTI file |
| `raw_seg_path` *(optional)* | Path to a precomputed raw-space segmentation, if available |
| `split` *(optional)* | `train` / other, used to fit normalization statistics only on the training split |

#### 1.2 Run the Preprocessing Pipeline

Run:

```bash
python preprocess_training_data.py \
    --input_csv /path/to/your/raw_data.csv \
    --prep_output_path /path/to/output/preprocessed \
    --threads n_threads
```

This performs, per subject/session:

1. **Super-resolution** (`mri_synthsr`) of the raw image.
2. **MNI registration** via lightweight landmark-based affine registration in `src/utils/USLR_lite`.
3. **Segmentation** (`mri_synthseg`, 96-label parcellation) of the registered image.
4. **Autoencoder latent encoding** of the preprocessed image.
5. **Multi-resolution tissue attention-mask** creation (used for the BrainST-img attention-supervision loss).
6. **Per-structure volume computation** and **z-score normalization** fitting.

#### 1.3 Resulting Directory Structure

```
output_preprocessed_path/
├── {dataset}/{subject_id}/{session_id}/
│   ├── {img_name}_preprocessed.nii.gz
│   ├── {img_name}_preprocessed_seg.nii.gz
│   ├── {img_name}_latent.npy
│   └── tissue_masks/
├── preprocessed_data.csv        # Original columns + paths to preprocessed outputs
├── brain_statistics.csv         # Per-structure voxel-count volumes
├── normalization_params.json    # Fitted z-score standardizer parameters (age + 18 ROIs)
└── training_data.csv            # Final training CSV (standardized volumes, integer-coded covariates)
```

The resulting `training_data.csv` and `normalization_params.json` are the two files consumed directly by the training scripts.


### 2. Training

```bash
# Train the BrainST-vol covariate-conditioned volume model
python training_brainst_vol.py

# Train the BrainST-img ROI-volume-conditioned image latent diffusion model
python training_brainst_img.py
```

Both scripts:

- Read their configuration from the `args_train` dictionary at the bottom of the file (edit before running).
- Log to TensorBoard under `{output_path}/{logs_dir_name}`.
- Periodically checkpoint to `{output_path}/{checkpoints_dir_name}` and save the best model (by validation loss) with a `_best` suffix.
- Periodically run validation (reconstruction accuracy via null-text inversion + distribution matching via Fréchet distance for `training_brainst_vol.py`; generation + segmentation-based ROI-volume accuracy for `training_brainst_img.py`).

Monitor training with:

```bash
tensorboard --logdir models/training/brainst_img/logs
tensorboard --logdir models/training/brainst_vol/logs
```

---

## Configuration

All global configuration lives in **`configs.py`**, which is imported everywhere in the codebase as `cfg`. Key settings you will likely need to adjust for your own environment:

| Setting | Description |
|---|---|
| `DEVICE` | Torch device string (e.g. `"cuda:0"`). |
| `PATH_DATA_TRAINING`, `PATH_DATA_GENERATION` | Default locations under `data/` for training and generation I/O |
| `PATH_MODELS_WEIGHTS`, `PATH_MODELS_ARCHITECTURES` | Locations of pretrained checkpoints (see [Pretrained Models](#pretrained-models)) |
| `PATH_NORMALIZATION_PARAMS` | Path to the fitted ROI z-score standardizer JSON |
| `STRUCTURE_INDEX_DICT` / `STRUCTURE_NAME_LIST_VOL` | Definition of the 18 conditioning ROIs (see below) |
| `BRAINST_IMG_NUM_INFERENCE_STEPS`, `BRAINST_VOL_NUM_INFERENCE_STEPS` | Number of DDIM denoising steps used at inference time (default: 50 for both) |
| `BRAINST_IMG_FREE_GUIDANCE_RATIO`, `BRAINST_VOL_FREE_GUIDANCE_RATIO` | Classifier-free-guidance strength for each model |
| `DX_MAPPING`, `SEX_MAPPING` | Integer encodings for diagnosis / sex covariates |
| `SHAPE_PREP_IMG`, `SHAPE_LATENT` | Fixed preprocessed-image shape `(192, 256, 192)` and latent shape `(4, 48, 64, 48)` |

The 18 conditioning ROIs are: `total`, `surrounding_csf`, `lateral_ventricles`, `fourth_ventricle`, `third_ventricle`, `cortical_gm`, `thalamus`, `hippocampus`, `caudate`, `putamen`, `pallidum`, `amygdala`, `accumbens_area`, `ventral_dc`, `cerebral_wm`, `cerebellum_gm`, `cerebellum_wm`, `brainstem`.

### Training Configuration

`training_brainst_img.py` and `training_brainst_vol.py` are **not** CLI-driven. Each file defines an `args_train` dictionary near the bottom of the script (paths to the training CSV, normalization params, batch size, learning rate, EMA settings, guidance ratios, validation schedule, etc.) — edit this dictionary directly before running. Key fields to update:

- `training_dataset_path_name` → your `training_data.csv`
- `normalizer_params` → your `normalization_params.json`
- `path_name_ref_img` (image model only) → a reference preprocessed image used to recover the correct NIfTI affine when saving validation outputs
- `output_path` → where checkpoints/logs/validation images will be written

### Generation Configuration

`main_generation.py` exposes a full `argparse` CLI (see [`parse_args()`](main_generation.py)). See [Generation / Inference](#generation--inference) above.

---


## Bonus: The "Brain Concert" Demo

`tutorials/brain_concert/generate_concert.ipynb` is a creative demonstration that maps a musical melody (e.g. "Ode to Joy") onto a sequence of BrainST-generated/-transformed images — enlarging or shrinking a different brain structure for each note — and renders a synchronized audio-visual video using `pydub` and `moviepy`.

```bash
jupyter notebook tutorials/brain_concert/generate_concert.ipynb
```
### Example videos
- **BrainST - Ode to Joy with sound (MP4):** [ode_to_joy.mp4](tutorials/brain_concert/data/outputs/concert_videos/ode_to_joy.mp4)
- **BrainST - Fur Elise with sound (MP4):** [fur_elise.mp4](tutorials/brain_concert/data/outputs/concert_videos/fur_elise.mp4)
- **BrainST - Note test with sound (MP4):** [note_test.mp4](tutorials/brain_concert/data/outputs/concert_videos/note_test.mp4)

---


## Reproducibility Notes

- Every generation and training entry point exposes (or hard-codes, in the training scripts) a `seed` used to seed NumPy, PyTorch CPU/GPU RNGs via `set_seed()` helpers in `src/brainst_img/utils_generation.py` and `src/brainst_vol/utils_generation.py`.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are set during training and generation to encourage deterministic behavior on a given GPU/CUDA version. Full bit-for-bit reproducibility across different hardware/CUDA versions is not guaranteed.
- Initial diffusion noise is sampled with a **dedicated CPU-based `torch.Generator`** (see `gen_random_latents`), decoupled from the global RNG state, so that the same `--seed` always produces the same initial noise regardless of prior random calls.
- Null-text inversion (used for transformation/longitudinal prediction) is itself a deterministic procedure given a fixed model, image, and hyperparameters (`num_inner_steps`, `early_stop_epsilon`, `free_guidance_ratio`), but its optimization loop (Adam) means results can vary slightly across PyTorch/CUDA versions.
- Classifier-free guidance ratios (`BRAINST_IMG_FREE_GUIDANCE_RATIO = 2.0`, `BRAINST_VOL_FREE_GUIDANCE_RATIO = 1.0`) and inference step counts (`= 50`) are fixed defaults in `configs.py` and directly affect generation quality/reproducibility if changed.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lathulerie2026structural,
  title   = {Structural Volume--Guided Diffusion Modelling for Counterfactual Brain MRI and Longitudinal Prediction},
  author  = {Lathulerie, Agustin Cartaya and Casamitjana, Adrià and Lisazo, Clara and Oliver, Arnau and Llado, Xavier},
  journal = {SSRN Electronic Journal},
  year    = {2026},
  doi     = {10.2139/ssrn.6614393}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE.txt) file for details.

---

## Acknowledgments

This project builds on and is grateful to the following open-source tools and research:

- **[MAISI](https://arxiv.org/pdf/2409.11169)** — the autoencoder and diffusion UNet backbone architecture used by BrainST-img.
- **[MONAI](https://monai.io/)** — medical imaging deep learning framework used for MAISI autoencoder/UNet architectures, diffusion schedulers and sliding-window inference.
- **[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) / SynthSeg / SynthSR** — used for robust whole-brain segmentation (`mri_synthseg`) and super-resolution (`mri_synthsr`) during preprocessing.
- **[Null-text Inversion](https://arxiv.org/abs/2211.09794)** — Mokady, R., Hertz, A., Aberman, K., Pritch, Y., & Cohen-Or, D. (2022). *Null-text Inversion for Editing Real Images using Guided Diffusion Models.* Used here to invert real ROI-volume vectors and image latents for anatomically faithful transformation and longitudinal prediction.
