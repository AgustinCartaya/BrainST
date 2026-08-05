"""Null-text inversion for BrainST-img: DDIM inversion + optional unconditional-embedding refinement.

See ``src.brainst_vol.null_inversion`` for the underlying technique
(identical algorithm, applied here to image latents instead of
ROI-volume vectors). This module additionally provides disk-caching
helpers (:func:`create_save_load_null_inversion_results`,
:func:`load_results`, :func:`save_results`) so expensive inversions can
be reused across multiple generation calls for the same source image.
"""

# pytorch
import json
import os

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from tqdm import tqdm

from . import utils_generation


class NullInversion:
    """Encapsulates DDIM inversion and null-text optimization for one BrainST-img model."""

    def _prev_step(self, noise_pred: torch.Tensor, timestep: int, latent: torch.Tensor) -> torch.Tensor:
        """Standard DDIM reverse (denoising) update: latent at ``t`` -> latent at ``t - step``.

        See ``src.brainst_vol.null_inversion.NullInversion._prev_step`` for details.
        """
        prev_timestep = timestep - self.noise_scheduler.num_train_timesteps // self.noise_scheduler.num_inference_steps
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def _next_step(self, noise_pred: torch.Tensor, timestep: int, latent: torch.Tensor) -> torch.Tensor:
        """DDIM inversion update (algebraic inverse of :meth:`_prev_step`).

        See ``src.brainst_vol.null_inversion.NullInversion._next_step`` for details.
        """
        # next step becomes the current step
        next_timestep = timestep
        # find the prev timestep
        timestep = min(timestep - self.noise_scheduler.num_train_timesteps // self.noise_scheduler.num_inference_steps, 999)
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.noise_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def _get_noise_pred_single(self, latents: torch.Tensor, t: int, context: torch.Tensor) -> torch.Tensor:
        """Predict noise using a single (already-selected) context embedding (no CFG mixing)."""
        noise_pred = self.unet(x=latents, timesteps=torch.tensor([t]).to(self.device), context=context)
        return noise_pred

    def _get_noise_pred(self, latents: torch.Tensor, t: int, is_adding_noise: bool = True, context: torch.Tensor | None = None) -> torch.Tensor:
        """Predict noise with classifier-free guidance and apply one DDIM step (forward or inverse)."""
        # predict the noise using free guidance (uncond and cond)
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_adding_noise else self.free_guidance_ratio
        noise_pred = self.unet(latents_input, torch.tensor([t]).to(self.device), context=context)
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond) 
        if is_adding_noise:
            latents = self._next_step(noise_pred, t, latents)
        else:
            latents = self._prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def _ddim_loop(self, latent: torch.Tensor) -> list[torch.Tensor]:
        """Run DDIM inversion: walk a clean latent forward into increasingly noisy states."""
        # obtain the noise from the latent (all steps)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        # first latent is the original latent (no noise)
        all_latent = [latent]
        # Ensure the latent tensor is a clone without history (detached) to avoid unwanted gradient tracking.
        latent = latent.clone().detach()
        # Iterate through all DDIM steps.
        if self.verbose:
            bar = tqdm(total=self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            # obtain from the last to the first timestep
            t = self.noise_scheduler.timesteps[self.num_ddim_steps - i - 1]
            # Predict noise using the conditional embedding.
            noise_pred = self._get_noise_pred_single(latent, t, cond_embeddings)
            # Update the latent sample using the noise prediction (add noise to the latent).
            latent = self._next_step(noise_pred, t, latent)
            all_latent.append(latent)

            if self.verbose:
                bar.update()
        if self.verbose:
            bar.close()
        return all_latent


    @torch.no_grad()
    def _ddim_inversion(self, latent) -> list[torch.Tensor]:
        """Convert a numpy array to a tensor and run :meth:`_ddim_loop`."""
        # convert latent to pytorch tensor
        latent = torch.from_numpy(latent).float().to(self.device)
        # Run the DDIM loop to get a sequence of latent states (from the least to the most noised).
        ddim_latents = self._ddim_loop(latent) 
        return ddim_latents 


    def _null_optimization(self, ddim_latents: list[torch.Tensor], num_inner_steps: int, epsilon: float) -> list[torch.Tensor]:
        """
            This method performs null-text optimization to refine the unconditional embedding
            so that the inversion better matches the original image.

            Args:
                ddim_latents: List of latents obtained from the DDIM inversion.
                num_inner_steps: Number of inner optimization steps for null-text optimization.
                epsilon: Threshold for early stopping in optimization.
            Returns:
                uncond_embeddings_list: List of optimized unconditional embeddings.
        """

        # Split the current context into unconditional and conditional embeddings.
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        # List to collect the optimized unconditional embeddings from each DDIM step.
        uncond_embeddings_list = []
        # Start with the most noised latent sample (the last one from the DDIM loop).
        latent_cur = ddim_latents[-1]
        # Iterate through each DDIM timestep.
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps, desc="Null-text optimization")
        for i in range(self.num_ddim_steps):
            # Clone the unconditional embeddings and ensure gradients are computed.
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True

            # Set up the Adam optimizer to update the unconditional embedding,
            # using a learning rate that decays slightly with each DDIM step.
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))

            # Get the "target" previous latent sample from the sequence
            # (starting from the end and moving backwards).
            latent_prev = ddim_latents[len(ddim_latents) - i - 2] 

            # Retrieve the corresponding timestep for the current DDIM step.
            # starting from the first (most noised step) to the last (least noised step)
            t = self.noise_scheduler.timesteps[i] 

            # Predict the conditional noise (which is not modified during optimization)
            # using the current latent and the fixed conditional embedding.
            with torch.no_grad():
                noise_pred_cond = self._get_noise_pred_single(latent_cur, t, cond_embeddings) # do not modify the conditioning

            # Inner loop: refine the unconditional embedding to help approximate the
            # transition from the current latent to the target previous latent.
            for j in range(num_inner_steps):
                # Predict noise based on the current latent and the currently optimized unconditional embedding.
                noise_pred_uncond = self._get_noise_pred_single(latent_cur, t, uncond_embeddings)

                # Combine the unconditional and conditional noise predictions using the free guidance ratio.
                noise_pred = noise_pred_uncond + self.free_guidance_ratio * (noise_pred_cond - noise_pred_uncond)

                # Compute the reconstructed latent sample using the predicted noise, the time step and the current latent (remove noise from the latent).
                latents_prev_rec = self._prev_step(noise_pred, t, latent_cur)

                # Compute the mean squared error loss between the reconstructed latent and the real.
                # update the unconditional embedding to minimize this loss.
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()

                bar.update(1)
                epsilon_th = epsilon + i * (2*epsilon)
                logs = {"loss": loss.detach().item(), "min_th": epsilon_th}
                bar.set_postfix(**logs)
                
                # penalize the first steps
                if loss_item < epsilon_th:
                    break

            # Update the progress bar for any remaining inner iterations if the loop exited early.
            for j in range(j + 1, num_inner_steps):
                bar.update()

            # Append the optimized unconditional embedding (keeping the dimensions) to the list.
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())

            # Update the current latent using the optimized unconditional embedding combined with the fixed conditional one.
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self._get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list


    def _embed_conditions(self, conditioning_tensor: torch.Tensor) -> None:
        """Embed the target ROI-volume profile and build the ``[unconditional, conditional]`` context.

        Side Effects:
            Sets ``self.context``.
        """
        conditioning_emb = self.conditions_model(conditioning_tensor.to(self.device))
        # add uncond conditioning
        self.context = torch.cat([torch.zeros_like(conditioning_emb), conditioning_emb], dim=0)



    def invert(self, latent, conditioning_tensor: torch.Tensor, num_inner_steps: int = 10, early_stop_epsilon: float = 1e-5, verbose: bool = False, compute_uncond_embeddings: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Orchestrate the full inversion process (embed -> DDIM invert -> optional null-text optimize).

        Args:
            latent: The initial (clean) image latent to invert.
            conditioning_tensor: ROI-volume conditioning tensor describing
                ``latent``'s subject.
            num_inner_steps: Inner optimization steps per DDIM step.
            early_stop_epsilon: Early-stopping loss threshold.
            verbose: If True, prints progress information.
            compute_uncond_embeddings: If False, skip null-text optimization.

        Returns:
            ``(ddim_latents, uncond_embeddings)`` -- see
            ``src.brainst_vol.null_inversion.NullInversion.invert``.
        """
        self.verbose = verbose
        # First, embed the external conditions into a context tensor.
        self._embed_conditions(conditioning_tensor)
        if verbose:
            print("DDIM inversion...")
        # Run the DDIM inversion process to obtain a list of latent states, from the least to the most noisy.
        ddim_latents = self._ddim_inversion(latent)
        if compute_uncond_embeddings:
            if verbose:
                print("Null-text optimization...")
            # Perform null-text optimization on the obtained DDIM latents to refine unconditional embeddings.
            uncond_embeddings = self._null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        else:
            uncond_embeddings = None
            if verbose:
                print("Skipping null-text optimization as per user request.")
            
        return ddim_latents, uncond_embeddings

    def __init__(self, unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, free_guidance_ratio: float):
        """
        Args:
            unet: Diffusion model to invert against.
            conditions_model: Model embedding ROI-volume conditions.
            noise_scheduler: Configured DDIM noise scheduler.
            free_guidance_ratio: Classifier-free-guidance strength used
                during the null-text optimization refinement step.
        """
        self.unet = unet
        self.conditions_model = conditions_model
        self.noise_scheduler = noise_scheduler
        self.context = None
        self.device = next(unet.parameters()).device

        self.num_ddim_steps = len(noise_scheduler.timesteps)
        self.free_guidance_ratio = free_guidance_ratio
        self.verbose = False



def invert_latents(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, 
                    latents, conditions: dict | list[dict], conditions_keys_ordered: list[str],
                    free_guidance_ratio: float = 7.5, num_inner_steps: int = 10, early_stop_epsilon: float = 1e-8,
                    compute_uncond_embeddings: bool = True, verbose: bool = True) -> dict:
    """Convenience wrapper: instantiate :class:`NullInversion` and invert a real image latent.

    Args:
        unet: Diffusion model.
        conditions_model: ROI-volume conditions embedding model.
        noise_scheduler: Configured DDIM noise scheduler.
        latents: The real (clean) image latent to invert.
        conditions: ROI-volume condition dict(s) describing the image.
        conditions_keys_ordered: Ordered condition keys.
        free_guidance_ratio: Classifier-free-guidance strength.
        num_inner_steps: Inner optimization steps per DDIM step.
        early_stop_epsilon: Early-stopping loss threshold.
        compute_uncond_embeddings: If False, skip null-text optimization.
        verbose: If True, show progress bars/prints.

    Returns:
        ``{"noisy_latents", "ddim_latents", "uncond_embeddings"}``, all
        detached to CPU tensors.

    Side Effects:
        Clears the CUDA cache after detaching all returned tensors.
    """
    # -------- Null inversion
    # instantiate null inversion and invert
    null_inversion = NullInversion(unet, conditions_model, noise_scheduler, free_guidance_ratio=free_guidance_ratio)
    
    # prepare the conditions tensor
    conditioning_tensor = utils_generation.prepare_condition_tensor(conditions, conditions_keys_ordered)
    ddim_latents, uncond_embeddings = null_inversion.invert(latents, conditioning_tensor, num_inner_steps=num_inner_steps, early_stop_epsilon=early_stop_epsilon, verbose=verbose, compute_uncond_embeddings=compute_uncond_embeddings)

    # detach all latents to free memory
    ddim_latents = [__lt.detach().cpu() for __lt in ddim_latents]
    if uncond_embeddings is not None:
        uncond_embeddings = [__uemb.detach().cpu() for __uemb in uncond_embeddings]
    noisy_latent = ddim_latents[-1]
    torch.cuda.empty_cache()

    return {
        "noisy_latents": noisy_latent,
        "ddim_latents": ddim_latents,
        "uncond_embeddings": uncond_embeddings
    }


def load_results(latents_output_path_name: str, uncond_embeddings_path_name: str) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Load a previously cached inversion result from disk.

    Args:
        latents_output_path_name: Path to the saved noisy-latents ``.npy`` file.
        uncond_embeddings_path_name: Path to the saved unconditional-embeddings
            ``.npy`` file. If it doesn't exist, ``uncond_embeddings`` is
            returned as ``None`` (with a warning printed).

    Returns:
        ``(noisy_latents, uncond_embeddings)``.
    """
    noisy_latents = np.load(latents_output_path_name)
    noisy_latents = torch.from_numpy(noisy_latents)

    uncond_embeddings = None
    if os.path.exists(uncond_embeddings_path_name):
        __uncond_embeddings = np.load(uncond_embeddings_path_name)
        uncond_embeddings = []
        for ult_index in range(__uncond_embeddings.shape[0]):
            uncond_embeddings.append(torch.from_numpy(__uncond_embeddings[ult_index]))
    else:
        print(f"WARNING: uncond_embeddings file {uncond_embeddings_path_name} not found, returning None for uncond_embeddings")
    return noisy_latents, uncond_embeddings

def save_results(noisy_latents: torch.Tensor, uncond_embeddings: list[torch.Tensor] | None, path_name_latents: str, path_name_uncond_emb: str) -> None:
    """Save an inversion result to disk for later reuse.

    Args:
        noisy_latents: The most-noised latent from DDIM inversion.
        uncond_embeddings: Optional list of per-step unconditional
            embeddings; if ``None``, ``path_name_uncond_emb`` is not written.
        path_name_latents: Output path for the noisy-latents ``.npy`` file.
        path_name_uncond_emb: Output path for the unconditional-embeddings
            ``.npy`` file.

    Side Effects:
        Writes ``path_name_latents`` and (if applicable)
        ``path_name_uncond_emb``.
    """
    np.save(path_name_latents, noisy_latents.numpy())
    if uncond_embeddings is not None:
        used_uncond_embeddings_to_save = []
        for ult_index in range(len(uncond_embeddings)):
            used_uncond_embeddings_to_save.append(uncond_embeddings[ult_index].numpy())
        np.save(path_name_uncond_emb, np.array(used_uncond_embeddings_to_save))




def create_save_load_null_inversion_results(unet: torch.nn.Module, conditions_model: torch.nn.Module, noise_scheduler, 
                                            latents, conditions: dict | list[dict], conditions_keys_ordered: list[str],
                                            free_guidance_ratio: float = 7.5, num_inner_steps: int = 10, early_stop_epsilon: float = 1e-8,
                                            compute_uncond_embeddings: bool = True, verbose: bool = True,
                                            latents_output_path_name: str | None = None, uncond_embeddings_path_name: str | None = None,
                                            ) -> dict:
    """Invert a real image latent, transparently caching the result to/from disk.

    If ``latents_output_path_name`` is given and a cached result already
    exists (respecting ``num_inner_steps==0`` as "no unconditional
    embeddings expected"), loads and returns it instead of recomputing.
    Otherwise runs :func:`invert_latents` and, if a cache path was given,
    saves the result plus a small JSON description of the parameters used.

    Args:
        unet: Diffusion model.
        conditions_model: ROI-volume conditions embedding model.
        noise_scheduler: Configured DDIM noise scheduler.
        latents: The real (clean) image latent to invert.
        conditions: ROI-volume condition dict(s) describing the image.
        conditions_keys_ordered: Ordered condition keys.
        free_guidance_ratio: Classifier-free-guidance strength.
        num_inner_steps: Inner optimization steps per DDIM step.
        early_stop_epsilon: Early-stopping loss threshold.
        compute_uncond_embeddings: If False, skip null-text optimization.
        verbose: If True, show progress bars/prints.
        latents_output_path_name: Optional path to cache/reuse the
            inverted noisy latents. If ``None``, caching is disabled and
            inversion is always recomputed.
        uncond_embeddings_path_name: Optional path to cache/reuse the
            null-text embeddings (required alongside
            ``latents_output_path_name`` for caching to be used, unless
            ``num_inner_steps == 0``).

    Returns:
        ``{"noisy_latents", "uncond_embeddings"}`` (and, when freshly
        computed rather than loaded from cache, also ``"ddim_latents"``
        -- see :func:`invert_latents`).

    Side Effects:
        When caching is enabled and no cached result exists yet: creates
        the output directory, writes the latents/embeddings ``.npy``
        files, and writes a ``*_desc.json`` file recording the inversion
        parameters used.
    """
    if latents_output_path_name is not None: 

        save_results_path =  os.path.dirname(latents_output_path_name)
        json_path =  os.path.splitext(os.path.basename(latents_output_path_name))[0] + "_desc.json"

        if os.path.exists(latents_output_path_name) and (os.path.exists(uncond_embeddings_path_name) or num_inner_steps==0):
            # load
            noisy_latents, uncond_embeddings = load_results(latents_output_path_name, uncond_embeddings_path_name)
            print(f"Using caching for null inversion results: {latents_output_path_name}")
            return {"noisy_latents": noisy_latents, "uncond_embeddings": uncond_embeddings}
            

        else:
            inverted_data = invert_latents(
                unet,
                conditions_model,
                noise_scheduler,
                latents,
                conditions,
                conditions_keys_ordered,
                free_guidance_ratio=free_guidance_ratio,
                compute_uncond_embeddings=compute_uncond_embeddings,
                num_inner_steps=num_inner_steps,
                early_stop_epsilon=early_stop_epsilon,
                verbose=verbose,
            )

            # save results
            os.makedirs(save_results_path, exist_ok=True)
            save_results(inverted_data["noisy_latents"], inverted_data["uncond_embeddings"], latents_output_path_name, uncond_embeddings_path_name)
            
            # create json to save the parameters used
            description = {
                "fgr": free_guidance_ratio,
                "inner_steps": num_inner_steps,
                "compute_uncond_embeddings": compute_uncond_embeddings,
                "early_stop_epsilon": early_stop_epsilon,
                "conditions": conditions
            }
            
            # save as json
            with open(os.path.join(save_results_path, json_path), 'w') as f:
                json.dump(description, f, indent=4)

    else:
        inverted_data = invert_latents(
            unet,
            conditions_model,
            noise_scheduler,
            latents,
            conditions,
            conditions_keys_ordered,
            free_guidance_ratio=free_guidance_ratio,
            compute_uncond_embeddings=compute_uncond_embeddings,
            num_inner_steps=num_inner_steps,
            early_stop_epsilon=early_stop_epsilon,
            verbose=verbose,
        )

    return inverted_data