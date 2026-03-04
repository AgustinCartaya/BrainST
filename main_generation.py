import os
import argparse
import json
from pathlib import Path
import pandas as pd

import src.utils.data_normalization as data_normalization
import src.preprocessing.prep_segmentation as prep_segmentation

import generation as generation

DX_MAPPING = {"CN":0, "MCI":1, "AD":2}
SEX_MAPPING = {"F":0, "M":1}

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TESTS_PATH = os.path.join(CURRENT_PATH, "tests")
TESTS_INPUT_PATH = os.path.join(TESTS_PATH, "inputs")
TESTS_OUTPUT_PATH = os.path.join(TESTS_PATH, "outputs")
TESTS_TEMP_PATH = os.path.join(TESTS_PATH, "temp")

AUTOENCODER_CHK_PATH = os.path.join(CURRENT_PATH, "models", "autoencoder", "weights", "autoencoder_epoch273.pt")

NORMALIZER_PARAMS = os.path.join(CURRENT_PATH, "src", "preprocessing", "normalizer_params.json")

# --------------- Synthesis examples ------------------
# python main_generation.py \
#     --generation_type synthesis \
#     --brainst_img_config_path ./models/brainst_img/cond18_masked/model_config.json \
#     --target_roi_volumes_path ./tests/inputs/target_vol_standardized.json \
#     --diffusion_steps 50

# python main_generation.py \
#     --generation_type synthesis \
#     --brainst_img_config_path ./models/brainst_img/cond18_masked/model_config.json \
#     --brainst_vol_config_path ./models/brainst_vol/test1_covars_add_ema4/model_config.json \
#     --target_age 75 \
#     --target_sex M \
#     --target_dx CN 

# --------------- transformation example -------------------------------
# python main_generation.py \
#     --generation_type transformation \
#     --image_path ./tests/inputs/image.nii.gz \
#     --segmentation_path ./tests/inputs/segmentation.nii.gz \
#     --brainst_img_config_path ./models/brainst_img/cond18_masked/model_config.json \
#     --target_roi_volumes_path ./tests/inputs/target_vol_standardized.json \
#     --target_roi_volumes_scale standardized \
#     --diffusion_steps 10
     
# python main_generation.py \
#     --generation_type transformation \
#     --image_path ./tests/inputs/image.nii.gz \
#     --segmentation_path ./tests/inputs/segmentation.nii.gz \
#     --brainst_img_config_path ./models/brainst_img/cond18_masked/model_config.json \
#     --target_roi_volumes_path ./tests/inputs/target_vol_mm.json \
#     --target_roi_volumes_scale mm3 \
#     --diffusion_steps 10

# python main_generation.py \
#     --generation_type transformation \
#     --image_path ./tests/inputs/image.nii.gz \
#     --segmentation_path ./tests/inputs/segmentation.nii.gz \
#     --brainst_img_config_path ./models/brainst_img/cond18_masked/model_config.json \
#     --brainst_vol_config_path ./models/brainst_vol/test1_covars_add_ema4/model_config.json \
#     --initial_age 79 \
#     --initial_sex M \
#     --initial_dx CN \
#     --target_age 65 \
#     --target_sex M \
#     --target_dx AD \
#     --diffusion_steps 10

# --------------- longitudinal example -------------------------------
# python main_generation.py \
#     --generation_type longitudinal \
#     --image_path ./tests/inputs/image.nii.gz \
#     --segmentation_path ./tests/inputs/segmentation.nii.gz \
#     --brainst_img_config_path ./models/brainst_img/cond18_masked/model_config.json \
#     --brainst_vol_config_path ./models/brainst_vol/test1_covars_add_ema4/model_config.json \
#     --initial_age 79 \
#     --initial_sex M \
#     --initial_dx CN \
#     --target_age 90 \
#     --target_dx CN \
#     --diffusion_steps 50

def parse_args():
    parser = argparse.ArgumentParser(
        description="MRI Generation Framework: Synthesis, Transformation, and Longitudinal Prediction"
    )

    # -------------------------------------------------
    # Core mode selection
    # -------------------------------------------------
    parser.add_argument(
        "--generation_type",
        type=str,
        required=True,
        choices=["synthesis", "transformation", "longitudinal"],
        help="Type of generation to perform."
    )
    
    # Models
    parser.add_argument("--brainst_img_config_path", required=True, type=str, help="Path to the BrainST-img config file.")
    parser.add_argument("--brainst_vol_config_path", required=False, type=str, help="Path to the BrainST-vol config file.")
    parser.add_argument("--autoencoder_chk_path", required=False, type=str, help="Path to the autoencoder checkpoint file.", default=AUTOENCODER_CHK_PATH)
    
    # Common optional inputs
    parser.add_argument("--target_roi_volumes_path",type=str, help="Path to JSON file containing target ROI volumes dictionary.")

    parser.add_argument("--target_age", type=float, help="Target age.")
    parser.add_argument("--target_sex", type=str, choices=["M", "F"], help="Target sex.", default="F")
    parser.add_argument("--target_dx", type=str, help="Target diagnosis (e.g., CN, MCI, AD).", default="CN")

    # Synthesis specific inputs
    parser.add_argument("--seed", type=int, default=2, help="Seed for random number generation.")

    # Transformation and longitudinal specific inputs
    parser.add_argument("--image_path", type=str, help="Path to input MRI image.")
    parser.add_argument("--segmentation_path", type=str, help="Path to segmentation file.")

    parser.add_argument("--initial_age", type=float, help="Initial age.")
    parser.add_argument("--initial_sex", type=str, choices=["M", "F"], help="Initial sex.", default="F")
    parser.add_argument("--initial_dx", type=str, help="Initial diagnosis (e.g., CN, MCI, AD).", default="CN")

    # Output
    parser.add_argument( "--output_dir", type=str, help="Directory to save generated outputs.", default=TESTS_OUTPUT_PATH)
    
    # other parameters
    parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of diffusion steps to perform.")


    parser.add_argument(
        "--target_roi_volumes_scale",
        type=str,
        choices=["mm3", "standardized"],
        default="standardized",
        help=(
            "Scale used to interpret the provided target ROI volumes. "
            "'mm3' expects absolute volumes in cubic millimeters, total_vol is the sum of all ROI volumes."
            "'standardized' expects z-scored volumes normalized according to the training distribution."
        )
    )

    return parser.parse_args()


def load_roi_dict(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def create_covariates_dict(args, normalizer=None, initial=False):
    if initial:
        return {
            "age": normalizer.transform_single(args.initial_age, "age"),
            "sex": SEX_MAPPING[args.initial_sex],
            "dx": DX_MAPPING[args.initial_dx]
        }
    else:
        return {
            "age": normalizer.transform_single(args.target_age, "age"),
            "sex": SEX_MAPPING[args.target_sex],
            "dx": DX_MAPPING[args.target_dx]
        }

def verify_target_roi_volumes_dict(target_roi_volumes_dict, target_roi_volumes_scale, normalizer=None):
    if target_roi_volumes_dict is None:
        return None
    
    if target_roi_volumes_scale == "mm3":
        # verify that all values are positive
        for key, value in target_roi_volumes_dict.items():
            if value <= 0:
                raise ValueError(f"Volume for {key} is negative or zero: {value}. All volumes must be positive.")
        structure_names = list(target_roi_volumes_dict.keys())
        target_roi_volumes_dict = data_normalization.normalize_by_icv(pd.DataFrame([target_roi_volumes_dict]), structure_names=structure_names, icv_column="total_vol", percentage=False).iloc[0].to_dict()
        target_roi_volumes_dict = normalizer.transform(pd.DataFrame([target_roi_volumes_dict])).iloc[0].to_dict()

        print("------------ Target ROI volumes dict:")
        print(target_roi_volumes_dict)
    if target_roi_volumes_scale == "standardized":
        # verify that all values are between -10 and 10 (assuming z-scoring)
        for key, value in target_roi_volumes_dict.items():
            if value < -10 or value > 10:
                print(f"Warning: Volume for {key} is {value}, which is outside the expected range for standardized values. Generated image may be of poor quality.")
    return target_roi_volumes_dict


def verify_segmentation(segmentation_path, img_path):
    # verify that the segmentation file exists, if not, run SynthSeg to create it
    if segmentation_path is None or not os.path.exists(segmentation_path):
        out_path_name = os.path.join(TESTS_OUTPUT_PATH, "temp_segmentation.nii.gz")
        print(f"Segmentation file not found at {segmentation_path}. Running SynthSeg to create segmentation...")
        prep_segmentation.save_synthseg_segmentation(img_path, out_path_name, verify=False, verbose=True, robust=True, cortical_parcelation=True)
        
        # verify that the segmentation was created
        if not os.path.exists(out_path_name):
            raise RuntimeError("Segmentation file not found after running SynthSeg. Please check that SynthSeg is installed and working correctly.")
        return out_path_name
    else:
        return segmentation_path

    

def generate_synthesis(args, normalizer=None):
    print("Running synthesis generation...")

    output_path_name = os.path.join(args.output_dir, "synthesized_image.nii.gz")
    if args.target_roi_volumes_path is not None:
        target_roi_volumes_dict = load_roi_dict(args.target_roi_volumes_path)
        generation.brainst_synthesis(
            brainst_img_config_path=args.brainst_img_config_path,
            autoencoder_chk_path=args.autoencoder_chk_path,
            output_path_name=output_path_name,
            target_roi_volumes_dict=target_roi_volumes_dict,
            seed=args.seed,
            diffusion_steps=args.diffusion_steps
        )
    elif args.brainst_vol_config_path is not None:
        # If we have a brainst_vol config, we can sample target volumes from the brainst_vol model and then use those volumes to generate the image with the brainst_img model
        generation.brainst_synthesis(
            brainst_img_config_path=args.brainst_img_config_path,
            autoencoder_chk_path=args.autoencoder_chk_path,
            output_path_name=output_path_name,
            brainst_vol_config_path=args.brainst_vol_config_path,
            target_covariates_dict=create_covariates_dict(args, normalizer, initial=False),
            seed=args.seed,
            diffusion_steps=args.diffusion_steps
        )
   


def generate_transformation(args, normalizer=None):
    print("Running transformation generation...")
    output_path_name = os.path.join(args.output_dir, "transformed_image.nii.gz")
    if args.target_roi_volumes_path is not None:
        target_roi_volumes_dict = load_roi_dict(args.target_roi_volumes_path)
        generation.brainst_transformation(
            image_path=args.image_path,
            segmentation_path=args.segmentation_path,
            brainst_img_config_path=args.brainst_img_config_path,
            autoencoder_chk_path=args.autoencoder_chk_path,
            normalizer=normalizer,
            output_path_name=output_path_name,
            target_roi_volumes_dict=target_roi_volumes_dict,
            diffusion_steps=args.diffusion_steps
        )
    elif args.brainst_vol_config_path is not None:
        # If we have a brainst_vol config, we can sample target volumes from the brainst_vol model and then use those volumes to generate the image with the brainst_img model
        generation.brainst_transformation(
            image_path=args.image_path,
            segmentation_path=args.segmentation_path,
            brainst_img_config_path=args.brainst_img_config_path,
            autoencoder_chk_path=args.autoencoder_chk_path,
            normalizer=normalizer,
            output_path_name=output_path_name,
            brainst_vol_config_path=args.brainst_vol_config_path,
            initial_covariates_dict=create_covariates_dict(args, normalizer, initial=True),
            target_covariates_dict=create_covariates_dict(args, normalizer, initial=False),
            diffusion_steps=args.diffusion_steps
        )
  
def generate_longitudinal(args, normalizer=None):
    print("Running longitudinal generation...")
    
    if args.brainst_vol_config_path is None:
        raise ValueError("brainst_vol_config_path must be provided for longitudinal generation.")
    
    output_path_name = os.path.join(args.output_dir, "longitudinal_image.nii.gz")

    initial_covariates_dict=create_covariates_dict(args, normalizer, initial=True)
    target_covariates_dict=create_covariates_dict(args, normalizer, initial=False)
    target_covariates_dict["sex"] = initial_covariates_dict["sex"] 
    
    generation.brainst_transformation(
            image_path=args.image_path,
            segmentation_path=args.segmentation_path,
            brainst_img_config_path=args.brainst_img_config_path,
            autoencoder_chk_path=args.autoencoder_chk_path,
            normalizer=normalizer,
            output_path_name=output_path_name,
            brainst_vol_config_path=args.brainst_vol_config_path,
            initial_covariates_dict=initial_covariates_dict,
            target_covariates_dict=target_covariates_dict,
            diffusion_steps=args.diffusion_steps
        )  
    
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    normalizer = data_normalization.SavedNormalizerBrainStructures(NORMALIZER_PARAMS)

    
    if args.target_roi_volumes_path is None and args.brainst_vol_config_path is None:
        raise ValueError("If target_roi_volumes_path is not provided, brainst_vol_config_path must be provided to sample target volumes from the BrainST-vol module.")
    if args.target_roi_volumes_path is not None:
        target_roi_volumes_dict = load_roi_dict(args.target_roi_volumes_path)
        target_roi_volumes_dict = verify_target_roi_volumes_dict(target_roi_volumes_dict, args.target_roi_volumes_scale, normalizer)
        
    # =====================================================
    # SYNTHESIS
    # =====================================================
    if args.generation_type == "synthesis":
        """
        Requires:
            - target_roi_volumes_dict OR
            - target covariates (age, sex, dx)
        """
        if not any([
            args.target_roi_volumes_path is not None,
            args.target_age is not None
        ]):
            raise ValueError(
                "Synthesis requires target ROI volumes or target covariates."
            )

        # Call synthesis pipeline -------------------------
        generate_synthesis(args, normalizer=normalizer)

    # =====================================================
    # TRANSFORMATION / LONGITUDINAL PREDICTION
    # =====================================================
    elif args.generation_type in ["transformation", "longitudinal"]:
        """
        Requires:
            - image_path
            - segmentation_path (or SynthSeg installed)
            - target ROI volumes OR target covariates
        """
        if args.image_path is None:
            raise ValueError("Transformation and longitudinal generation require --image_path.")
        # verify segmentation, if not provided, create it with SynthSeg
        args.segmentation_path = verify_segmentation(args.segmentation_path, args.image_path)

        # Call transformation pipeline -------------------------
        if args.generation_type == "transformation":
            if not any([
            args.target_roi_volumes_path is not None,
            args.target_age is not None
            ]):
                raise ValueError(
                    "Transformation requires target ROI volumes or target covariates."
                )
            generate_transformation(args, normalizer=normalizer)
        else:
            if args.initial_age is None or  args.target_age is None:
                raise ValueError(
                    "Longitudinal prediction requires, initial_age, target_age."
                )
            generate_longitudinal(args, normalizer=normalizer)

    
    else:
        raise ValueError(f"Unknown generation type: {args.generation_type}")


if __name__ == "__main__":
    main()