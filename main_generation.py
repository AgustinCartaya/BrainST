"""Entry point for the MRI Generation Framework.

Supports three generation modes, all driven by the BrainST diffusion models:

1. **synthesis**      -- generate a brand-new synthetic brain image from
                          scratch, conditioned on either a target ROI-volume
                          profile, a target covariates set (age / sex / dx),
                          or a target segmentation/image.
2. **transformation**  -- take an existing (real) brain image and morph it
                          toward a target ROI-volume profile or a target
                          segmentation/image, while preserving subject
                          identity as much as possible.
3. **longitudinal**    -- take an existing brain image at an initial
                          age/sex/dx and simulate how it would look at a
                          target age/dx (used to model disease progression
                          or healthy aging).

Typical usage (see ``cli_generation_examples.txt`` for full CLI invocations)::

    python main_generation.py --generation_type synthesis ...
    python main_generation.py --generation_type transformation ...
    python main_generation.py --generation_type longitudinal ...

Note:
    ``main()`` currently calls ``create_default_args()`` instead of
    ``parse_args()`` -- this is convenient for local/IDE debugging (edit
    ``create_default_args`` directly) but means the CLI arguments defined
    in ``parse_args`` are not actually read at runtime. Swap the call in
    ``main()`` back to ``parse_args()`` for normal CLI use.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

import configs as cfg
import generation as generation
import src.utils.functions as fc
import src.utils.nifti_functions as nfc
import src.utils.utils_io as uio
from src.preprocessing import preprocess_images
from src.utils import data_normalization, prep_segmentation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Build and parse the command-line interface for this script.

    Returns:
        Parsed CLI arguments.
    """
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
        help="Type of generation to perform.",
    )

    # Common optional inputs
    parser.add_argument(
        "--path_target_roi_volumes",
        type=str,
        help="Path to JSON file containing target ROI volumes dictionary.",
    )
    parser.add_argument("--target_age", type=float, help="Target age.")
    parser.add_argument("--target_sex", type=str, choices=["M", "F"], default="F", help="Target sex.")
    parser.add_argument("--target_dx", type=str, default="CN", help="Target diagnosis (e.g., CN, MCI, AD).")

    # Synthesis-specific inputs
    parser.add_argument("--seed", type=int, default=2, help="Seed for random number generation.")

    # Synthesis- and Transformation-specific inputs
    parser.add_argument("--path_reference_image", type=str, help="Path to MRI image to copy roi volumes from.")
    parser.add_argument(
        "--path_reference_segmentation", type=str, help="Path to MRI segmentation to copy roi volumes from."
    )

    # Transformation- and longitudinal-specific inputs
    parser.add_argument("--path_image", type=str, help="Path to input MRI image.")
    parser.add_argument("--path_segmentation", type=str, help="Path to segmentation file.")
    parser.add_argument(
        "--apply_preprocessing", action="store_true", help="Whether to apply preprocessing to the input image."
    )

    parser.add_argument("--initial_age", type=float, help="Initial age.")
    parser.add_argument("--initial_sex", type=str, choices=["M", "F"], default="F", help="Initial sex.")
    parser.add_argument("--initial_dx", type=str, default="CN", help="Initial diagnosis (e.g., CN, MCI, AD).")

    # Output
    parser.add_argument("--output_dir", type=str, default=cfg.PATH_TEMP, help="Directory to save generated outputs.")
    parser.add_argument("--output_name", type=str, help="Name of the output file.")

    # Other parameters
    parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of diffusion steps to perform.")

    parser.add_argument(
        "--target_roi_volumes_scale",
        type=str,
        choices=["mm3", "standardized"],
        default="standardized",
        help=(
            "Scale used to interpret the provided target ROI volumes. "
            "'mm3' expects absolute volumes in cubic millimeters, total_vol is the sum of all ROI volumes. "
            "'standardized' expects z-scored volumes normalized according to the training distribution."
        ),
    )

    return parser.parse_args()


def create_covariates_dict(args: argparse.Namespace, normalizer=None, is_target: bool = False) -> dict:
    """Build the ``{age, sex, dx}`` covariates dict expected by the models.

    Args:
        args: Parsed arguments containing either the ``initial_*`` or
            ``target_*`` covariate fields.
        normalizer: Fitted normalizer used to standardize the raw age
            value.
        is_target: If True, build the covariates dict from the
            ``target_*`` fields; otherwise use the ``initial_*`` fields.

    Returns:
        Dictionary with normalized ``age`` and integer-coded ``sex``/``dx``.
    """
    prefix = "target" if is_target else "initial"
    return {
        "age": normalizer.transform_single(getattr(args, f"{prefix}_age"), "age"),
        "sex": cfg.SEX_MAPPING[getattr(args, f"{prefix}_sex")],
        "dx": cfg.DX_MAPPING[getattr(args, f"{prefix}_dx")],
    }


def verify_target_roi_volumes_dict(target_roi_volumes_dict: dict, target_roi_volumes_scale: str, normalizer) -> dict:
    """Validate and, if needed, convert a target ROI-volumes dict to the standardized scale.

    Args:
        target_roi_volumes_dict: Mapping of ROI name -> volume.
        target_roi_volumes_scale: Either ``"mm3"`` (raw cubic-millimeter
            volumes, requires an ICV normalization step) or
            ``"standardized"`` (already z-scored).
        normalizer: Fitted normalizer used to standardize volumes when
            converting from the ``"mm3"`` scale.

    Returns:
        The ROI-volumes dict on the standardized (z-scored) scale.

    Raises:
        ValueError: If ``target_roi_volumes_scale`` is not recognized, or
            if any raw volume is non-positive (invalid) when
            ``target_roi_volumes_scale == "mm3"``.
    """
    if target_roi_volumes_scale == "mm3":
        # All raw volumes must be strictly positive to be physically valid.
        for key, value in target_roi_volumes_dict.items():
            if value <= 0:
                raise ValueError(f"Volume for {key} is negative or zero: {value}. All volumes must be positive.")

        # Convert absolute volumes -> percentage of ICV -> standardized (z-score) scale.
        structure_names = list(target_roi_volumes_dict.keys())
        target_roi_volumes_dict = data_normalization.normalize_by_icv(
            pd.DataFrame([target_roi_volumes_dict]),
            structure_names=structure_names,
            icv_column="total_vol",
            percentage=False,
        ).iloc[0].to_dict()
        target_roi_volumes_dict = normalizer.transform(pd.DataFrame([target_roi_volumes_dict])).iloc[0].to_dict()

        logger.info("Target ROI volumes dict: %s", target_roi_volumes_dict)

    elif target_roi_volumes_scale == "standardized":
        # Sanity check: standardized (z-scored) values should typically fall
        # within roughly [-10, 10]. Values outside this range likely
        # indicate a data error and may degrade generation quality.
        for key, value in target_roi_volumes_dict.items():
            if value < -10 or value > 10:
                logger.warning(
                    "Volume for %s is %s, which is outside the expected range for standardized values. "
                    "Generated image may be of poor quality.",
                    key,
                    value,
                )
    else:
        raise ValueError(
            f"Unknown target_roi_volumes_scale: {target_roi_volumes_scale}. Must be 'mm3' or 'standardized'."
        )
    return target_roi_volumes_dict


def verify_segmentation(path_segmentation: str | None, img_path: str, output_name: str) -> str:
    """Ensure a segmentation file exists for ``img_path``, generating one with SynthSeg if missing.

    Args:
        path_segmentation: Path to an existing segmentation file, or
            ``None``/non-existent if it still needs to be generated.
        img_path: Path to the source image to segment (used only if
            ``path_segmentation`` is missing).
        output_name: Name of the output segmentation file.

    Returns:
        Path to a valid segmentation file (either the original
        ``path_segmentation`` or the newly generated one).

    Raises:
        RuntimeError: If SynthSeg runs but fails to produce an output file.

    Side Effects:
        May run ``mri_synthseg`` as a subprocess and write a segmentation
        file to ``cfg.PATH_TEMP``.
    """
    if path_segmentation is None or not os.path.exists(path_segmentation):
        out_path_name = os.path.join(cfg.PATH_TEMP, output_name)
        logger.info("Segmentation file not found at %s. Running SynthSeg to create segmentation...", path_segmentation)
        prep_segmentation.save_synthseg_segmentation(
            img_path, out_path_name, verify=False, verbose=True, robust=True, cortical_parcelation=True
        )

        if not os.path.exists(out_path_name):
            raise RuntimeError(
                "Segmentation file not found after running SynthSeg. "
                "Please check that SynthSeg is installed and working correctly."
            )
        return out_path_name
    return path_segmentation


def verify_preprocessing(input_path_name: str, seg_path_name: str | None = None, apply_preprocessing: bool = True) -> dict:
    """Preprocess (or lightly validate) an input image + segmentation pair.

    Args:
        input_path_name: Path to the raw input MRI image.
        seg_path_name: Path to the raw segmentation for that image. If
            ``None``, the segmentation is assumed to be missing and will
            be generated via SynthSeg.
        apply_preprocessing: If True, run the full preprocessing pipeline
            (super-resolution, registration, segmentation, resizing,
            normalization). If False, only run the last two steps
            (resize + normalize) -- just enough to guarantee the correct
            shape/scale, assuming the image was already preprocessed
            upstream.

    Returns:
        ``{"img_prep", "seg_prep", "img_org", "aff_org", "aff_prep"}``
        containing the preprocessed and original image/affine data, ready
        to be consumed by the generation functions and later used to map
        results back to the original image space.

    Side Effects:
        Writes intermediate/preprocessed files under ``cfg.PATH_TEMP``.
    """
    if apply_preprocessing:
        preprocessed_img_path, preprocessed_seg_path = preprocess_images.preprocess_image(
            input_path_name=input_path_name,
            output_path=cfg.PATH_TEMP,
            shape_prep_img=cfg.SHAPE_PREP_IMG,
            steps=cfg.PREPROCESSING_IMAGE_STEPS,
            seg_raw_path_name=seg_path_name,
            verify=False,
            verbose=True,
        )
    else:
        # Skip the heavy steps (super-resolution/registration/segmentation)
        # and only apply the final resize + normalize steps, since the
        # image is assumed to already be preprocessed.
        preprocessed_img_path, preprocessed_seg_path = preprocess_images.preprocess_image(
            input_path_name=input_path_name,
            output_path=cfg.PATH_TEMP,
            shape_prep_img=cfg.SHAPE_PREP_IMG,
            steps=cfg.PREPROCESSING_IMAGE_STEPS[-2:],
            seg_raw_path_name=seg_path_name,
            verify=False,
            verbose=True,
        )

    org_img, org_aff = nfc.load_nifti(input_path_name)
    img, aff = nfc.load_nifti(preprocessed_img_path)
    seg, _ = nfc.load_nifti(preprocessed_seg_path)

    prep_dict = {
        "img_prep": img,
        "seg_prep": seg,
        "img_org": org_img,
        "aff_org": org_aff,
        "aff_prep": aff,
    }

    return prep_dict


def postprocessing_and_save(
    img: np.ndarray,
    output_path_name: str,
    org_shape: tuple[int, int, int] | None = None,
    org_aff=None,
) -> None:
    """Undo preprocessing (if applicable) and write the result to disk.

    Args:
        img: Generated image array (in preprocessed space).
        output_path_name: Destination path for the saved NIfTI file.
        org_shape: Original (pre-preprocessing) image shape. If provided,
            the image is resampled/cropped back to this shape before
            saving.
        org_aff: Original affine (and header) metadata to attach to the
            saved NIfTI file. If ``None``, an identity/default affine is
            used by ``nfc.save_nifti``.

    Side Effects:
        Writes ``output_path_name`` to disk.
    """
    if org_shape is not None:
        img = preprocess_images.postprocess_image(img, org_shape)
    nfc.save_nifti(img, org_aff, output_path_name)


def generate_synthesis(args: argparse.Namespace, normalizer=None):
    """Run the *synthesis* generation pipeline: create a new brain image from scratch.

    Priority of conditioning signal (first available wins):

    1. ``args.target_roi_volumes_dict``
    2. ``args.target_age`` (+ target sex/dx covariates)
    3. ``args.path_reference_segmentation`` (target segmentation)

    Args:
        args: Parsed/assembled arguments (see priority list above).
        normalizer: Fitted normalizer, used when synthesizing from a
            target segmentation.

    Returns:
        The synthesized image array.
    """
    logger.info("Running synthesis generation...")

    if getattr(args, "target_roi_volumes_dict", None) is not None:
        logger.info("Using target ROI volumes dict for synthesis.")
        img_gen = generation.brainst_synthesis(
            target_roi_volumes_dict=args.target_roi_volumes_dict,
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
        )
    elif getattr(args, "target_age", None) is not None:
        logger.info("Using target covariates (age, sex, dx) for synthesis.")
        img_gen = generation.brainst_synthesis(
            target_covariates_dict=create_covariates_dict(args, normalizer, is_target=True),
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
        )
    else:
        logger.info("Using target segmentation for synthesis.")
        target_seg = nfc.load_nifti(args.path_reference_segmentation, is_label=True)[0]
        img_gen = generation.brainst_synthesis(
            target_seg=target_seg,
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
            normalizer=normalizer,
        )

    return img_gen


def generate_transformation(args: argparse.Namespace, prep_dict: dict, normalizer=None):
    """Run the *transformation* generation pipeline.

    Morphs an existing (preprocessed) brain image toward a target
    ROI-volume profile or a target segmentation.

    Args:
        args: Parsed arguments, expected to hold
            ``target_roi_volumes_dict`` and/or
            ``path_reference_segmentation``.
        prep_dict: Output of :func:`verify_preprocessing` for the source
            image.
        normalizer: Fitted normalizer for ROI volumes.

    Returns:
        The transformed image array.
    """
    logger.info("Running transformation generation...")
    if getattr(args, "target_roi_volumes_dict", None) is not None:
        logger.info("Using target ROI volumes dict for transformation.")
        img_gen = generation.brainst_transformation(
            img=prep_dict["img_prep"],
            seg=prep_dict["seg_prep"],
            normalizer=normalizer,
            target_roi_volumes_dict=args.target_roi_volumes_dict,
            diffusion_steps=args.diffusion_steps,
        )
    else:
        logger.info("Using target segmentation for transformation.")
        target_seg = nfc.load_nifti(args.path_reference_segmentation, is_label=True)[0]
        img_gen = generation.brainst_transformation(
            img=prep_dict["img_prep"],
            seg=prep_dict["seg_prep"],
            normalizer=normalizer,
            target_seg=target_seg,
            diffusion_steps=args.diffusion_steps,
        )

    return img_gen


def generate_longitudinal(args: argparse.Namespace, prep_dict: dict, normalizer=None):
    """Run the *longitudinal* generation pipeline.

    Simulates how an existing brain image would look at a different
    (target) age/diagnosis, keeping sex fixed to the subject's initial
    sex.

    Args:
        args: Parsed arguments, expected to hold ``initial_age/sex/dx``
            and ``target_age/dx``.
        prep_dict: Output of :func:`verify_preprocessing` for the source
            image.
        normalizer: Fitted normalizer for the age covariate.

    Returns:
        The longitudinally-predicted image array.
    """
    logger.info("Running longitudinal generation...")

    initial_covariates_dict = create_covariates_dict(args, normalizer, is_target=False)
    target_covariates_dict = create_covariates_dict(args, normalizer, is_target=True)
    # Sex is a fixed subject attribute -- it cannot change over time, so
    # force the target sex to match the initial sex regardless of what
    # (if anything) was passed in via --target_sex.
    target_covariates_dict["sex"] = initial_covariates_dict["sex"]

    img_gen = generation.brainst_longitudinal(
        img=prep_dict["img_prep"],
        seg=prep_dict["seg_prep"],
        normalizer=normalizer,
        initial_covariates_dict=initial_covariates_dict,
        target_covariates_dict=target_covariates_dict,
        diffusion_steps=args.diffusion_steps,
    )
    return img_gen


def create_default_args() -> argparse.Namespace:
    """Build a hard-coded ``argparse.Namespace`` for local/IDE debugging.

    Bypasses the CLI entirely. Only the *synthesis* branch is currently
    active; the *transformation* and *longitudinal* variants are kept
    below (commented out) as ready-to-use templates -- uncomment
    whichever one you need and comment out the others.

    Returns:
        A hard-coded ``argparse.Namespace`` suitable for
        :func:`synthesis_controller`.
    """
    # ---- synthesis default args -------------------------------------
    default_args = argparse.Namespace(
        generation_type="synthesis",
        # path_target_roi_volumes=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "target_vol_standardized.json"),
        # target_age=99,
        # target_sex="F",
        # target_dx="CN",
        seed=2,
        path_reference_image=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "reference_image.nii.gz"),
        path_reference_segmentation=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "reference_segmentation.nii.gz"),
        output_dir=os.path.join(cfg.PATH_DATA_GENERATION, "outputs"),
        diffusion_steps=cfg.BRAINST_IMG_NUM_INFERENCE_STEPS,
        target_roi_volumes_scale="standardized",
    )

    # # ---- transformation default args (template) ----------------------
    # default_args = argparse.Namespace(
    #     generation_type="transformation",
    #     path_target_roi_volumes=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "target_vol_standardized.json"),
    #     seed=2,
    #     path_image=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "basal_image.nii.gz"),
    #     path_segmentation=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "basal_segmentation.nii.gz"),
    #     # path_reference_image=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "reference_image.nii.gz"),
    #     # path_reference_segmentation=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "reference_segmentation.nii.gz"),
    #     apply_preprocessing=False,
    #     output_dir=os.path.join(cfg.PATH_DATA_GENERATION, "outputs"),
    #     diffusion_steps=cfg.BRAINST_IMG_NUM_INFERENCE_STEPS,
    #     target_roi_volumes_scale="standardized"
    # )

    # # ---- longitudinal default args (template) -------------------------
    # default_args = argparse.Namespace(
    #     generation_type="longitudinal",
    #     target_age=89,
    #     target_sex="F",
    #     target_dx="CN",
    #     seed=2,
    #     path_image=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "basal_image.nii.gz"),
    #     path_segmentation=os.path.join(cfg.PATH_DATA_GENERATION, "inputs", "basal_segmentation.nii.gz"),
    #     apply_preprocessing=False,
    #     initial_age=74,
    #     initial_sex="F",
    #     initial_dx="CN",
    #     output_dir=os.path.join(cfg.PATH_DATA_GENERATION, "outputs"),
    #     diffusion_steps=cfg.BRAINST_IMG_NUM_INFERENCE_STEPS,
    #     target_roi_volumes_scale="standardized"
    # )
    return default_args


def synthesis_controller(args: argparse.Namespace) -> None:
    """Top-level driver: resolve targets, run the requested generation mode, save the result.

    Currently invoked with ``create_default_args()`` (hard-coded args for
    local debugging) rather than ``parse_args()`` (real CLI parsing) --
    see :func:`main`.

    Args:
        args: Fully-specified generation arguments (see
            :func:`create_default_args` / :func:`parse_args`).

    Raises:
        ValueError: If no valid target-conditioning signal is provided,
            or if required arguments for the selected ``generation_type``
            are missing.

    Side Effects:
        Writes the generated NIfTI image to
        ``{args.output_dir}/{generation_type}_{timestamp}.nii.gz``, and
        may write intermediate preprocessing/segmentation files under
        ``cfg.PATH_TEMP``.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = data_normalization.SavedNormalizerBrainStructures(cfg.PATH_NORMALIZATION_PARAMS)
    img_gen = None
    prep_dict = None

    # -----------------------------------------------------------------
    # Resolve which "target" conditioning signal to use. Exactly one of
    # the following is required:
    #   - path_target_roi_volumes  (highest priority)
    #   - target_age (+ sex/dx)
    #   - path_reference_segmentation / path_reference_image
    # -----------------------------------------------------------------
    if (
        getattr(args, "path_target_roi_volumes", None) is None
        and getattr(args, "target_age", None) is None
        and (
            getattr(args, "path_reference_segmentation", None) is None
            and getattr(args, "path_reference_image", None) is None
        )
    ):
        raise ValueError(
            "Please provide either path_target_roi_volumes, target_age, or "
            "(path_reference_segmentation or path_reference_image)."
        )

    # If multiple target signals were provided, prioritize:
    #   path_target_roi_volumes > target_age > path_reference_segmentation/path_reference_image
    if getattr(args, "path_target_roi_volumes", None) is not None and (
        getattr(args, "target_age", None) is not None
        or getattr(args, "path_reference_segmentation", None) is not None
        or getattr(args, "path_reference_image", None) is not None
    ):
        logger.warning("Multiple target parameters provided. Using path_target_roi_volumes.")
        args.target_age = None
        args.path_reference_segmentation = None
        args.path_reference_image = None
    elif getattr(args, "target_age", None) is not None and (
        getattr(args, "path_reference_segmentation", None) is not None
        or getattr(args, "path_reference_image", None) is not None
    ):
        logger.warning("Multiple target parameters provided. Using target_age.")
        args.path_reference_segmentation = None
        args.path_reference_image = None

    # Load + convert the ROI-volumes target (mm3 -> standardized) if provided.
    if getattr(args, "path_target_roi_volumes", None) is not None:
        target_roi_volumes_dict = uio.load_json(args.path_target_roi_volumes)
        target_roi_volumes_dict = verify_target_roi_volumes_dict(
            target_roi_volumes_dict, args.target_roi_volumes_scale, normalizer
        )
        args.target_roi_volumes_dict = target_roi_volumes_dict

    # Ensure a target segmentation exists (generate via SynthSeg if needed).
    if getattr(args, "path_reference_segmentation", None) is not None or getattr(args, "path_reference_image", None) is not None:
        path_reference_segmentation = verify_segmentation(
            getattr(args, "path_reference_segmentation", None),
            getattr(args, "path_reference_image", None),
            output_name="temp_reference_segmentation.nii.gz"
        )
        args.path_reference_segmentation = path_reference_segmentation

    # =====================================================
    # SYNTHESIS
    # =====================================================
    if args.generation_type == "synthesis":
        # Requires: target_roi_volumes_dict OR target covariates (age, sex, dx)
        img_gen = generate_synthesis(args, normalizer=normalizer)

    # =====================================================
    # TRANSFORMATION / LONGITUDINAL PREDICTION
    # =====================================================
    elif args.generation_type in ("transformation", "longitudinal"):
        # Requires: path_image, path_segmentation (or SynthSeg installed),
        # and target ROI volumes OR target covariates or target image/segmentation.
        if args.path_image is None or not os.path.exists(args.path_image):
            raise ValueError("Transformation and longitudinal generation require a valid --path_image.")

        prep_dict = verify_preprocessing(
            args.path_image, args.path_segmentation,  apply_preprocessing=args.apply_preprocessing
        )

        if args.generation_type == "transformation":
            if (
                getattr(args, "target_roi_volumes_dict", None) is None
                and getattr(args, "path_reference_segmentation", None) is None
            ):
                raise ValueError(
                    "Transformation requires either target_roi_volumes_dict or "
                    "path_reference_segmentation/path_reference_image."
                )
            img_gen = generate_transformation(args, prep_dict, normalizer=normalizer)
        else:
            if getattr(args, "initial_age", None) is None or getattr(args, "target_age", None) is None:
                raise ValueError("Longitudinal prediction requires initial_age and target_age.")
            img_gen = generate_longitudinal(args, prep_dict, normalizer=normalizer)

    else:
        raise ValueError(f"Unknown generation type: {args.generation_type}")

    # Save the result, mapping back to the original image space/affine
    # when the source image was preprocessed (transformation/longitudinal).
    time_stamp = fc.get_time_stamp()
    output_name = getattr(args, "output_name", None)
    if output_name is None:
        output_name = f"{args.generation_type}_{time_stamp}.nii.gz"
    output_path_name = os.path.join(args.output_dir, output_name)
    if prep_dict is not None:
        postprocessing_and_save(img_gen, output_path_name, org_shape=prep_dict["img_org"].shape, org_aff=prep_dict["aff_org"])
    else:
        postprocessing_and_save(img_gen, output_path_name)



def main() -> None:
    """Script entry point (currently uses hard-coded debug args; see module docstring)."""
    args = parse_args()
    # args = create_default_args()
    synthesis_controller(args)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()