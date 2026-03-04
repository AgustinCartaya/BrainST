# BrainST: Structural Volume–Guided Diffusion Modelling for Counterfactual Brain MRI and Longitudinal Prediction

BrainST is a diffusion-based framework for the **controlled synthesis, anatomical transformation, and longitudinal prediction of T1-weighted brain MRI**, trained entirely on cross-sectional data. It enables fine-grained, region-specific control by conditioning image generation on volumetric measurements of **18 brain regions of interest (ROIs)**, while preserving anatomical plausibility through a conditioning alignment penalty.

## Features

- **Cross-sectional MRI generation** from manually specified or automatically predicted ROI volumes  
- **Localized anatomical transformations** of existing images while preserving subject-specific anatomy  
- **Longitudinal prediction** of brain changes associated with healthy aging or neurodegenerative diseases (e.g., Alzheimer’s disease)  
- **Automatic ROI volume prediction** from demographic and cognitive variables  
- **Counterfactual image synthesis** with precise region-specific control  

BrainST is designed to **support research on brain aging, neurodegeneration, and structural variability**, particularly when longitudinal MRI data are limited.

## Installation

```bash
# Clone the repository
git clone https://github.com/AgustinCartaya/BrainST.git
cd BrainST

# (Optional) Create a virtual environment
conda create --name BrainST python=3.11
conda deactivate
conda activate BrainST

# Install required packages
pip install torch torchvision monai tensorboard ipykernel tqdm matplotlib opencv-python pandas nibabel scikit-image scikit-learn SimpleITK ipympl





