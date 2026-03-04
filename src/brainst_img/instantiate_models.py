



import os

import torch


from .networks_declaration.diffusion_model_unet_maisi_mask_att import DiffusionModelUNetMaisi
from .networks_declaration.ddim import DDIMScheduler
from .networks_declaration.volumne_encoder import ConditionTokens
from .autoencoder_declaration import AutoencoderPrediction



def instantiate_conditioned_models(networks_config, autoencoder_chk_path, device, dm_num_inference_steps, half=True):
    
    args = networks_config
    
    # autoencoder (just for validation)
    autoencoder = AutoencoderPrediction(autoencoder_chk_path, device, half=half)

    # unet
    unet = DiffusionModelUNetMaisi(spatial_dims = args.diffusion_unet_def.spatial_dims,
                                                in_channels = args.diffusion_unet_def.in_channels,
                                                out_channels = args.diffusion_unet_def.out_channels,
                                                num_res_blocks = args.diffusion_unet_def.num_res_blocks,
                                                num_channels = args.diffusion_unet_def.num_channels,
                                                self_attention_levels = args.diffusion_unet_def.self_attention_levels,
                                                cross_attention_levels = args.diffusion_unet_def.cross_attention_levels,
                                                num_self_head_channels = args.diffusion_unet_def.num_self_head_channels,
                                                num_cross_head_channels = args.diffusion_unet_def.num_cross_head_channels,
                                                with_conditioning = args.diffusion_unet_def.with_conditioning,
                                                transformer_num_layers = args.diffusion_unet_def.transformer_num_layers,
                                                cross_attention_dim = args.diffusion_unet_def.cross_attention_dim,
                                                upcast_attention = args.diffusion_unet_def.upcast_attention,
                                                use_flash_attention = args.diffusion_unet_def.use_flash_attention,
                                                )
    
    noise_scheduler = DDIMScheduler(
                            beta_start=args.noise_scheduler.beta_start,
                            beta_end=args.noise_scheduler.beta_end,
                            num_train_timesteps=args.noise_scheduler.num_train_timesteps,
                            schedule=args.noise_scheduler.schedule,
                            clip_sample=args.noise_scheduler.clip_sample
                        )
    noise_scheduler.set_timesteps(num_inference_steps=dm_num_inference_steps)

        
    conditions_model = ConditionTokens(num_conditions=args.conditions_model.num_conditions, 
                                    embed_dim=args.conditions_model.embed_dim,
                                    hidden_dim=args.conditions_model.hidden_dim,
                                    use_self_attention=args.conditions_model.use_self_attention, 
                                    n_heads=args.conditions_model.n_heads, 
                                    n_layers=args.conditions_model.n_att_layers,
                                    # dropout=args.conditions_model.dropout,
                                    use_gelu=args.conditions_model.use_gelu,)



    return {"unet": unet, 
            "conditions_model": conditions_model,
            "noise_scheduler": noise_scheduler,
              "autoencoder": autoencoder, 
              }



    
def instantiate_model_and_load(model_description, autoencoder_chk_path, device, dm_num_inference_steps=50, half=True):

    # -------- instantiate models
    # model_description = load_parameters(model_description_path)
    
    # YOU NEED TO FIX THIS
    # autoencoder_chk_path = f"{model_description.args_train.autoencoder_chk_path_name}"
    models = instantiate_conditioned_models(model_description.networks_config, autoencoder_chk_path, device, dm_num_inference_steps, half=half)

    # ---- load unet checkpoint
    # checkpoint_path_name = f"{model_description.args_train.output_path}/{model_description.args_train.checkpoints_dir_name}/{model_description.args_train.model_best_chk_name}"
    checkpoint_path_name = f"{model_description.args_train.model_best_chk_path_name}"
    # verify if checkpoint exists
    if not os.path.exists(checkpoint_path_name):
        print(f"Checkpoint {checkpoint_path_name} does not exist")

    checkpoint = torch.load(checkpoint_path_name, weights_only=False, map_location=device)
    if model_description.args_train.use_ema:
        models["unet"].load_state_dict(checkpoint["ema_state_dict"], strict=True)
    else:
        models["unet"].load_state_dict(checkpoint["unet_state_dict"], strict=True) 
    models["conditions_model"].load_state_dict(checkpoint["conditions_model_state_dict"], strict=True)

    models["unet"].to(device).eval()
    models["conditions_model"].to(device).eval()
    
    models["model_description"] = model_description

    return models

