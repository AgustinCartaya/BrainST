"""Model instantiation and checkpoint loading for BrainST-img.

BrainST-img is a 3D latent diffusion model: a MAISI-architecture UNet
conditioned on ROI-volume profiles via a cross-attention `ConditionTokens`
module, using a DDIM noise scheduler and a separately-trained MAISI
autoencoder for latent<->image decoding.
"""


import torch

from .autoencoder_declaration import AutoencoderPrediction
from .networks_declaration.ddim import DDIMScheduler
from .networks_declaration.diffusion_model_unet_maisi_mask_att import (
    DiffusionModelUNetMaisi,
)
from .networks_declaration.volumne_encoder import ConditionTokens


def instantiate_conditioned_models(networks_config, autoencoder_chk_path: str, device: torch.device, dm_num_inference_steps: int, half: bool = True) -> dict:
    """Build (and load the pretrained autoencoder for) the BrainST-img model bundle.

    Note:
        Unlike the UNet/conditions model (whose weights are loaded
        separately via :func:`instantiate_model_and_load`), the
        autoencoder is loaded here directly from ``autoencoder_chk_path``,
        since it is a fixed, separately-trained component not part of the
        BrainST-img training run itself.

    Args:
        networks_config: Architecture config (``argparse.Namespace``),
            typically ``fc.dict_to_args(cfg.ARCHITECTURE_BRAINST_IMG, deep_conversion=True)``.
        autoencoder_chk_path: Path to the pretrained autoencoder checkpoint.
        device: Device to instantiate the autoencoder on.
        dm_num_inference_steps: Number of DDIM inference steps to
            configure the noise scheduler with.
        half: Whether to run the autoencoder in fp16 (passed through to
            :class:`~src.brainst_img.autoencoder_declaration.AutoencoderPrediction`).

    Returns:
        ``{"unet", "conditions_model", "noise_scheduler", "autoencoder"}``.
    """
    
    args = networks_config
    
    # autoencoder (just for validation)
    autoencoder = AutoencoderPrediction(autoencoder_chk_path, device, half=half)

    # unet
    unet = DiffusionModelUNetMaisi(
        spatial_dims=args.diffusion_unet_def.spatial_dims,
        in_channels=args.diffusion_unet_def.in_channels,
        out_channels=args.diffusion_unet_def.out_channels,
        num_res_blocks=args.diffusion_unet_def.num_res_blocks,
        num_channels=args.diffusion_unet_def.num_channels,
        self_attention_levels=args.diffusion_unet_def.self_attention_levels,
        cross_attention_levels=args.diffusion_unet_def.cross_attention_levels,
        num_self_head_channels=args.diffusion_unet_def.num_self_head_channels,
        num_cross_head_channels=args.diffusion_unet_def.num_cross_head_channels,
        with_conditioning=args.diffusion_unet_def.with_conditioning,
        transformer_num_layers=args.diffusion_unet_def.transformer_num_layers,
        cross_attention_dim=args.diffusion_unet_def.cross_attention_dim,
        upcast_attention=args.diffusion_unet_def.upcast_attention,
        use_flash_attention=args.diffusion_unet_def.use_flash_attention,
    )
    
    noise_scheduler = DDIMScheduler(
        beta_start=args.noise_scheduler.beta_start,
        beta_end=args.noise_scheduler.beta_end,
        num_train_timesteps=args.noise_scheduler.num_train_timesteps,
        schedule=args.noise_scheduler.schedule,
        clip_sample=args.noise_scheduler.clip_sample
    )
    noise_scheduler.set_timesteps(num_inference_steps=dm_num_inference_steps)

        
    conditions_model = ConditionTokens(
        num_conditions=args.conditions_model.num_conditions,
        embed_dim=args.conditions_model.embed_dim,
        hidden_dim=args.conditions_model.hidden_dim,
        use_self_attention=args.conditions_model.use_self_attention,
        n_heads=args.conditions_model.n_heads,
        n_layers=args.conditions_model.n_att_layers,
        use_gelu=args.conditions_model.use_gelu,
    )



    return {"unet": unet, 
            "conditions_model": conditions_model,
            "noise_scheduler": noise_scheduler,
              "autoencoder": autoencoder, 
              }


def instantiate_model_and_load(networks_config, brainst_img_chk_path: str, autoencoder_chk_path: str, device: torch.device, dm_num_inference_steps: int = 50, half: bool = True) -> dict:
    """Build the BrainST-img model bundle and load trained UNet/conditions-model weights.

    Args:
        networks_config: Architecture config (see :func:`instantiate_conditioned_models`).
        brainst_img_chk_path: Path to a ``.pt`` checkpoint (as saved by
            ``training_brainst_img.py``'s ``save_model``).
        autoencoder_chk_path: Path to the pretrained autoencoder checkpoint.
        device: Device to load the checkpoint onto and move models to.
        dm_num_inference_steps: Number of DDIM inference steps.
        half: Whether to run the autoencoder in fp16.

    Returns:
        The model bundle from :func:`instantiate_conditioned_models`,
        with ``unet``/``conditions_model`` weights loaded and both set to
        ``.eval()`` mode on ``device``. If the checkpoint contains an
        ``"ema_state_dict"``, those (EMA) weights are used for the UNet
        instead of the raw ``"unet_state_dict"``.
    """

    # -------- instantiate models
    models = instantiate_conditioned_models(networks_config, autoencoder_chk_path, device, dm_num_inference_steps, half=half)
    checkpoint = torch.load(brainst_img_chk_path, weights_only=False, map_location=device)
    if "ema_state_dict" in checkpoint:
        models["unet"].load_state_dict(checkpoint["ema_state_dict"], strict=True)
    else:
        models["unet"].load_state_dict(checkpoint["unet_state_dict"], strict=True)
    models["conditions_model"].load_state_dict(checkpoint["conditions_model_state_dict"], strict=True)

    models["unet"].to(device).eval()
    models["conditions_model"].to(device).eval()

    return models