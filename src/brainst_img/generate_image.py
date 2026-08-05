"""Reverse-diffusion sampling and decoding loop for BrainST-img (image generation)."""

import contextlib

import numpy as np
import torch
from tqdm import tqdm

from . import utils_generation


@torch.no_grad()
def diffusion_loop(
    noisy_latents: torch.Tensor,
    unet: torch.nn.Module,
    conditions_model: torch.nn.Module,
    noise_scheduler,
    autoencoder,
    conditions_list: list[dict],
    conditions_keys_ordered: list[str],
    uncond_embeddings: list[torch.Tensor] | None = None,
    free_guidance_ratio: float = 0,
    decode_img: bool = True,
    decode_first: bool = True,  # this is for not decoding the reconstructed if it is first
    decode_complete: bool = True,
    sliding_window_size: tuple[int, int, int] = (48, 48, 48), 
    overlap: float = 0.25,
    return_denoising_steps: bool = False,
) -> dict:
    """Run the reverse-diffusion loop over image latents and decode to image space.

    Args:
        noisy_latents: Starting latent tensor, shape ``(1, C, D, H, W)``
            (expanded to the batch size implied by ``conditions_list``).
        unet: Diffusion UNet predicting noise/sample/velocity.
        conditions_model: Model embedding the ROI-volume conditions into
            cross-attention context tokens.
        noise_scheduler: Noise scheduler (DDIM-style), already configured
            with the desired number of inference timesteps (a suffix of
            its full ``timesteps`` schedule is used, sized to
            ``len(noise_scheduler.timesteps)`` at call time via
            ``start_time``).
        autoencoder: Latent-space decoder
            (:class:`~src.brainst_img.autoencoder_declaration.AutoencoderPrediction`).
        conditions_list: List of ROI-volume condition dicts, one per
            batch element (determines the batch size).
        conditions_keys_ordered: Ordered list of condition keys.
        uncond_embeddings: Optional per-timestep unconditional embeddings
            (for null-text-inversion-based reconstruction); if ``None``,
            a zero embedding is used at every step (standard CFG).
        free_guidance_ratio: Classifier-free-guidance strength.
        decode_img: If True, decode the final latents to image space.
        decode_first: If False, skip decoding the first (index 0) batch
            element -- used when the first latent is a "reconstruction
            reference" that doesn't need to be materialized as an image.
        decode_complete: Passed to ``autoencoder.decode``: if True, decode
            the full volume in one pass; if False, use sliding-window
            patch decoding (for large volumes / limited memory).
        sliding_window_size: Patch size used when ``decode_complete=False``.
        overlap: Patch overlap fraction used when ``decode_complete=False``.
        return_denoising_steps: If True, return the intermediate latents
            at each denoising step (for visualization / analysis).

    Returns:
        ``{"latents": np.ndarray, "images": np.ndarray}`` -- the final
        latents (batch, on CPU as numpy) and, if ``decode_img=True``, the
        decoded images (one per batch element actually decoded, per
        ``decode_first``), stacked along axis 0.

    Side Effects:
        Frees CUDA cache after the denoising loop and after each
        per-sample decode (to reduce peak memory for large batches).
    """

    device = next(unet.parameters()).device
    batch_size = len(conditions_list)

    conditioning = utils_generation.prepare_condition_tensor(conditions_list, conditions_keys_ordered)
    conditioning_emb = conditions_model(conditioning.to(device))
    
    if uncond_embeddings is None:
        uncond_embeddings_ = torch.zeros_like(conditioning_emb)
    else:
        uncond_embeddings_ = None

    start_time = len(noise_scheduler.timesteps)

    latents = noisy_latents.expand(batch_size, *noisy_latents.shape[1:]).to(device)

    if return_denoising_steps:
        denoising_steps = [latents.cpu().numpy()]

    # synthesize latents
    ctx = torch.amp.autocast("cuda") if autoencoder.half else contextlib.nullcontext()
    with ctx:
        for i, t in enumerate(tqdm(noise_scheduler.timesteps[-start_time:])):
            if uncond_embeddings_ is None: # means thath we have optimized the uncond embeddings
                context = torch.cat([uncond_embeddings[i].expand(*conditioning_emb.shape).to(device), conditioning_emb], dim=0)
            else: # means that we are reconstructing the image only from the noised latents
                context = torch.cat([uncond_embeddings_, conditioning_emb], dim=0)
            
            latents = torch.cat([latents] * 2)

            noise_pred = unet(
                x=latents,
                timesteps=torch.Tensor((t,)).to(device),
                context=context,
            )
            
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + free_guidance_ratio * (noise_pred_cond - noise_pred_uncond)
            latents = latents[:batch_size]

            if return_denoising_steps:
                denoising_steps.append(latents.cpu().numpy())
            latents, _ = noise_scheduler.step(noise_pred, t, latents)

        del noise_pred
        torch.cuda.empty_cache()

        if decode_img:

            synthetic_images_list = []
            # decode one by one
            for i in range(batch_size):
                if not decode_first and i==0:
                    continue
                try:
                    synthetic_images = autoencoder.decode(latents[i].unsqueeze(0), decode_complete=decode_complete, sliding_window_size=sliding_window_size, overlap=overlap)
                    synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).squeeze().cpu().numpy()
                except RuntimeError as e:
                    # print the error
                    print(f"Error decoding image {i}: {e}")
                    synthetic_images = np.zeros(np.array(noisy_latents.shape[2:])*4)

                synthetic_images_list.append(synthetic_images)
                torch.cuda.empty_cache()
                
            synthetic_images = np.stack(synthetic_images_list, axis=0)
            
        return {
            "latents": latents.cpu().numpy(),
            "images": synthetic_images,
            "denoising_steps": denoising_steps if return_denoising_steps else None,
        }