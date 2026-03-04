

from tqdm import tqdm
import numpy as np

import torch
import contextlib

from . import utils_generation

@torch.no_grad()
def diffusion_loop(
    noisy_latents,
    unet,
    conditions_model,
    noise_scheduler,
    autoencoder,
    conditions_list,
    conditions_keys_ordered,
    uncond_embeddings = None,
    memory_reserver=None,
    free_guidance_ratio=0,
    decode_img=True,
    decode_first=True,  # this is for not decoding the reconstructed if it is first
    decode_complete=True,
    sliding_window_size=(48, 48, 48), 
    overlap=0.25
):

    # # free memory for the whole process
    if memory_reserver is not None:
        memory_reserver.free()

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

    # synthesize latents
    # with torch.amp.autocast("cuda"):
    ctx = torch.amp.autocast("cuda") if autoencoder.half else contextlib.nullcontext()
    with ctx:
        # for t in tqdm(noise_scheduler.timesteps):
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

        # loock memory for the autoencoder
        if memory_reserver is not None:
            memory_reserver.reserve_previous()
        # return res_return
        return {
            "latents": latents.cpu().numpy(),
            "images": synthetic_images
        }
            