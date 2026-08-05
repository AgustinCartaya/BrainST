"""Model instantiation and checkpoint loading for BrainST-vol.

BrainST-vol is a 1D (per-ROI-scalar) diffusion model: a small MLP-based
UNet (`MLPDiffusion`) conditioned on demographic/clinical covariates via
a `ConditioningModule`, using a DDIM noise scheduler.
"""

import torch

from .networks_declaration import models as diffusion_model_unet1D
from .networks_declaration.ddim import DDIMScheduler


def instantiate_conditioned_models(networks_config, device: torch.device, dm_num_inference_steps: int) -> dict:
    """Build (but do not load weights for) the BrainST-vol model bundle.

    Args:
        networks_config: Architecture config (``argparse.Namespace``),
            typically ``fc.dict_to_args(cfg.ARCHITECTURE_BRAINST_VOL, deep_conversion=True)``.
        device: Device to build the noise scheduler's config for (models
            themselves are returned un-moved; callers are responsible for
            ``.to(device)``).
        dm_num_inference_steps: Number of DDIM inference steps to
            configure the noise scheduler with.

    Returns:
        ``{"unet", "noise_scheduler", "networks_config", "conditions_model"}``.
    """
    args = networks_config

    unet = diffusion_model_unet1D.MLPDiffusion(
        d_in=args.diffusion_mlp.d_in,
        dim_t=args.diffusion_mlp.dim_t,
        conditioning_type=args.diffusion_mlp.conditioning_type,
        num_heads=args.diffusion_mlp.num_heads,
    )

    conditions_model = diffusion_model_unet1D.ConditioningModule(
        covar_dimension=args.conditions_mlp.covar_dimension,
        dim_t=args.conditions_mlp.dim_t,
        covar_embed_dim=args.conditions_mlp.covar_embed_dim,
        conditioning_type=args.conditions_mlp.conditioning_type,
    )

    noise_scheduler = DDIMScheduler(
        beta_start=args.noise_scheduler.beta_start,
        beta_end=args.noise_scheduler.beta_end,
        num_train_timesteps=args.noise_scheduler.num_train_timesteps,
        schedule=args.noise_scheduler.schedule,
        clip_sample=args.noise_scheduler.clip_sample,
    )
    
    noise_scheduler.set_timesteps(num_inference_steps=dm_num_inference_steps)

    return {"unet": unet, 
            "noise_scheduler": noise_scheduler,
              "networks_config": args,
              "conditions_model": conditions_model
            }


def instantiate_model_and_load(networks_config, brainst_vol_chk_path: str, device: torch.device, dm_num_inference_steps: int = 50) -> dict:
    """Build the BrainST-vol model bundle and load trained weights from a checkpoint.

    Args:
        networks_config: Architecture config (see :func:`instantiate_conditioned_models`).
        brainst_vol_chk_path: Path to a ``.pt`` checkpoint (as saved by
            ``training_brainst_vol.py``'s ``save_model``).
        device: Device to load the checkpoint onto and move models to.
        dm_num_inference_steps: Number of DDIM inference steps.

    Returns:
        The model bundle from :func:`instantiate_conditioned_models`,
        with ``unet``/``conditions_model`` weights loaded and both set to
        ``.eval()`` mode on ``device``. If the checkpoint contains an
        ``"ema_state_dict"``, those (EMA) weights are used for the UNet
        instead of the raw ``"unet_state_dict"``.
    """
    models = instantiate_conditioned_models(networks_config, device, dm_num_inference_steps)
 
    # ---- load unet checkpoint
    checkpoint = torch.load(brainst_vol_chk_path, weights_only=False, map_location=device)
    if "ema_state_dict" in checkpoint:
        models["unet"].load_state_dict(checkpoint["ema_state_dict"], strict=True)
    else:
        models["unet"].load_state_dict(checkpoint["unet_state_dict"], strict=True) 
    models["conditions_model"].load_state_dict(checkpoint["conditions_model_state_dict"], strict=False)
 
    models["unet"].to(device).eval()
    models["conditions_model"].to(device).eval()

    return models