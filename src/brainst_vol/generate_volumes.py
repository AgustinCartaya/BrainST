"""Reverse-diffusion sampling loop for BrainST-vol (ROI-volume generation)."""

import numpy as np
import torch

from . import utils_generation


@torch.no_grad()
def diffusion_loop(
                    initial_noise: torch.Tensor, 
                    unet: torch.nn.Module, 
                   conditions_model: torch.nn.Module,
                   noise_scheduler, 
                    covars_list_dict: list[dict] | dict, 
                    covars_keys_ordered: list[str] = ["age", "sex", "dx"],
                    uncond_embeddings: list[torch.Tensor] | None = None,
                   free_guidance_ratio: float = 1.0,
                   return_noisy_steps: bool = False) -> np.ndarray | tuple:
    """Run the reverse-diffusion sampling loop over ROI-volume vectors.

    Public-API counterpart to ``training_brainst_vol.py``'s (private,
    training-script-local) ``diffusion_loop`` -- this version builds its
    own conditioning tensor from ``covars_list_dict`` via
    :func:`src.brainst_vol.utils_generation.prepare_condition_tensor`
    rather than requiring a pre-built tensor, making it suitable for
    direct use from ``generation.py``.

    Args:
        initial_noise: Starting point of the reverse diffusion, shape
            ``(batch, num_conditions)``.
        unet: Diffusion model predicting noise/sample/velocity.
        conditions_model: Model embedding the covariates into a context vector.
        noise_scheduler: Noise scheduler (DDIM-style) already configured
            with inference timesteps.
        covars_list_dict: One covariates dict (or list of dicts, one per
            batch element) with keys matching ``covars_keys_ordered``.
        covars_keys_ordered: Ordered covariate keys to build the
            conditioning tensor from.
        uncond_embeddings: Optional per-timestep unconditional embeddings
            (see :func:`src.brainst_vol.training_brainst_vol.diffusion_loop`
            for the two supported CFG modes -- identical here).
        free_guidance_ratio: Classifier-free-guidance strength.
        return_noisy_steps: If True, also return the intermediate
            (noisy) volume vectors at every denoising step.

    Returns:
        The final denoised ROI-volume vector(s) as a numpy array, shape
        ``(batch, num_conditions)``. If ``return_noisy_steps=True``,
        returns ``(volumes, denoising_steps)``.
    """
    
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
        """Perform a single reverse-diffusion step, applying classifier-free guidance if configured."""
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
            x = x[:batch_size]
            x, _ = noise_scheduler.step(noise_pred, t, x)
        
        return x
    
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