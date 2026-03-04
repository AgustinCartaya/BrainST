


import torch

from . import utils_generation

@torch.no_grad()
def diffusion_loop(
                    initial_noise, 
                    unet, 
                   conditions_model,
                   noise_scheduler, 
                    covars_list_dict, 
                    covars_keys_ordered=["age", "sex", "dx"],
                    uncond_embeddings = None,
                   free_guidance_ratio=1.0,
                   return_noisy_steps=False):
    
    device = next(unet.parameters()).device
    initial_noise = initial_noise.to(device)

    covars = utils_generation.prepare_condition_tensor(covars_list_dict, covars_keys_ordered)
    covars = covars.to(device)

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    conditioning_emb = conditions_model(covars)

    if uncond_embeddings is None:
        uncond_embeddings_ = torch.zeros_like(conditioning_emb)
    else:
        uncond_embeddings_ = None

    volumens = initial_noise
    if return_noisy_steps:
        denoising_steps = []
        
    def denoising_step(x, model, t, context=None, next_t=None, fgr=1.0):
        # free guidance setup
        using_free_guidance = False
        batch_size = x.shape[0]
        
        if context.shape[0] == x.shape[0] * 2:
            using_free_guidance = True
            x = torch.cat([x] * 2)
            
        timesteps = torch.full((x.shape[0],), fill_value=t, dtype=all_timesteps.dtype, device=device)
        noise_pred = model(x=x,timesteps=timesteps,context=context)
        
        if using_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + fgr * (noise_pred_cond - noise_pred_uncond)
            # noise_pred = noise_pred_cond + free_guidance_ratio * (noise_pred_uncond - noise_pred_cond) # mine
            x = x[:batch_size]

        # if not isinstance(noise_scheduler, RFlowScheduler):
        #     x, _ = noise_scheduler.step(noise_pred, t, x)
        # else:
        #     x, _ = noise_scheduler.step(noise_pred, t, x, next_t)  # type: ignore
            x, _ = noise_scheduler.step(noise_pred, t, x)
        
        return x
    
    # for i, (t, next_t) in enumerate(progress_bar):
    for i, (t, next_t) in enumerate(zip(all_timesteps, all_next_timesteps)):

        if uncond_embeddings_ is None: # means thath we have optimized the uncond embeddings
            context = torch.cat([uncond_embeddings[i].expand(*conditioning_emb.shape).to(device), conditioning_emb], dim=0)
        else: # means that we are reconstructing the image only from the noised latents
            context = torch.cat([uncond_embeddings_, conditioning_emb], dim=0)
        
        volumens = denoising_step(volumens, unet, t, context=context, next_t=next_t, fgr=free_guidance_ratio)

        if return_noisy_steps:
            denoising_steps.append(volumens.cpu().numpy())
    volumens = volumens.cpu().numpy()
    if return_noisy_steps:
        return volumens, denoising_steps
    return volumens
