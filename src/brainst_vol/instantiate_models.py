



import os
import torch

from .networks_declaration import models as diffusion_model_unet1D
from .networks_declaration.ddim import DDIMScheduler


# def instantiate_models(num_conditions, covars_dimension, conditioning_type="add", noise_scheduler_type="ddim", merge_conditioning_with="x"):
def instantiate_conditioned_models(networks_config, device, dm_num_inference_steps):
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
    # noise scheduler
    noise_scheduler = DDIMScheduler(
                        beta_start=args.noise_scheduler.beta_start,
                        beta_end=args.noise_scheduler.beta_end,
                        num_train_timesteps=args.noise_scheduler.num_train_timesteps,
                        schedule=args.noise_scheduler.schedule,
                        clip_sample=args.noise_scheduler.clip_sample
                    )
    noise_scheduler.set_timesteps(num_inference_steps=dm_num_inference_steps)

    return {"unet": unet, 
            "noise_scheduler": noise_scheduler,
              "networks_config": args,
              "conditions_model": conditions_model
            }


# def load_parameters(model_description_path):
#     # load model_description_path as a json
#     with open(model_description_path, 'r') as f:
#         model_description = json.load(f)
#     model_description = fc.dict_to_args(model_description, deep_conversion=True)
#     return model_description

def instantiate_model_and_load(model_description, device, dm_num_inference_steps=50):

    # -------- instantiate models
    # model_description = load_parameters(model_description_path)
    # print(f"Instantiating models: {model_description.args_train.model_name}")

    models = instantiate_conditioned_models(model_description.networks_config, device, dm_num_inference_steps)

    # ---- load unet checkpoint
    checkpoint_path_name = f"{model_description.args_train.model_best_chk_path_name}"
    # verify if checkpoint exists
    if not os.path.exists(checkpoint_path_name):
        print(f"Checkpoint {checkpoint_path_name} does not exist")

    checkpoint = torch.load(checkpoint_path_name, weights_only=False, map_location=device)
    if model_description.args_train.use_ema:
        models["unet"].load_state_dict(checkpoint["ema_state_dict"], strict=True)
    else:
        models["unet"].load_state_dict(checkpoint["unet_state_dict"], strict=True) 
    models["conditions_model"].load_state_dict(checkpoint["conditions_model_state_dict"], strict=False)

    models["unet"].to(device).eval()
    models["conditions_model"].to(device).eval()
    models["model_description"] = model_description

    return models

