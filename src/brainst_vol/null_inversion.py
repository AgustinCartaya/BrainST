
from tqdm import tqdm

# pytorch
import torch
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from . import utils_generation


class NullInversion:

    def _prev_step(self, noise_pred, timestep, latent):
        prev_timestep = timestep - self.noise_scheduler.num_train_timesteps // self.noise_scheduler.num_inference_steps
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.noise_scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def _next_step(self, noise_pred, timestep, latent):
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

    def _get_noise_pred_single(self, latents, t, context):
        # predict the noise using just one contex embedding
        noise_pred = self.unet(x=latents, timesteps=torch.tensor([t]).to(self.device), context=context)
        return noise_pred

    def _get_noise_pred(self, latents, t, is_adding_noise=True, context=None):
        # predict the noise using free guidance (uncond and cond)
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_adding_noise else self.free_guidance_ratio
        # guidance_scale = 0 if is_adding_noise else self.free_guidance_ratio # mine
        noise_pred = self.unet(latents_input, torch.tensor([t]).to(self.device), context=context)
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond) 
        # noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text) # mine
        if is_adding_noise:
            latents = self._next_step(noise_pred, t, latents)
        else:
            latents = self._prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def _ddim_loop(self, latent):
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
    def _ddim_inversion(self, latent):
        # convert latent to pytorch tensor
        latent = torch.from_numpy(latent).float().to(self.device)
        # Run the DDIM loop to get a sequence of latent states (from the least to the most noised).
        ddim_latents = self._ddim_loop(latent) 
        return ddim_latents 


    def _null_optimization(self, ddim_latents, num_inner_steps, epsilon):
        """
            This method performs null-text optimization to refine the unconditional embedding
            so that the inversion better matches the original image.
            Parameters:
                - ddim_latents: List of latents obtained from the DDIM inversion.
                - num_inner_steps: Number of inner optimization steps for null-text optimization.
                - epsilon: Threshold for early stopping in optimization.
            Returns:
                - uncond_embeddings_list: List of optimized unconditional embeddings.
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
                # noise_pred = noise_pred_cond + self.free_guidance_ratio * (noise_pred_uncond - noise_pred_cond) # mine

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
                # if loss_item < epsilon + i * 2e-5:
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


    def _embed_conditions(self, conditioning_tensor):
        conditioning_emb = self.conditions_model(conditioning_tensor.to(self.device))
        # add uncond conditioning
        self.context = torch.cat([torch.zeros_like(conditioning_emb), conditioning_emb], dim=0)



    def invert(self, latent, conditioning_tensor, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, compute_uncond_embeddings=True):
        """
            This method orchestrates the complete inversion process:
            1. It embeds the provided conditions.
            2. Runs DDIM inversion to retrieve a sequence of latent samples.
            3. Optimizes the unconditional embeddings via null-text optimization.
            Parameters:
                - latent: The initial latent representation to be inverted.
                - conditioning_tensor: A tensor containing the conditions to be used for inversion.
                - num_inner_steps: Number of inner optimization steps for null-text optimization.
                - early_stop_epsilon: Threshold for early stopping in optimization.
                - verbose: If True, prints progress information.
            Returns:
                - ddim_latents: List of latents obtained from the DDIM inversion.
                - uncond_embeddings: List of optimized unconditional embeddings.
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

    def __init__(self, unet, conditions_model, noise_scheduler, free_guidance_ratio):
        self.unet = unet
        self.conditions_model = conditions_model
        self.noise_scheduler = noise_scheduler
        self.context = None
        self.device = next(unet.parameters()).device

        self.num_ddim_steps = len(noise_scheduler.timesteps)
        self.free_guidance_ratio = free_guidance_ratio
        self.verbose = False


def invert_latents(unet, conditions_model, noise_scheduler, 
                    input_vec, covars_list_dict, covars_keys_ordered=["age", "sex", "dx"],
                    free_guidance_ratio=1.0, num_inner_steps=10, early_stop_epsilon=1e-8,
                    compute_uncond_embeddings=True):
    # -------- Null inversion
    # instantiate null inversion and invert
    null_inversion_class = NullInversion(unet, conditions_model, noise_scheduler, free_guidance_ratio=free_guidance_ratio)
    
    # prepare the conditions tensor
    # conditioning_tensor = conditions_model(covars.to(device)).to(device)
    # convert covars to tensor if not already
    conditioning_tensor = utils_generation.prepare_condition_tensor(covars_list_dict, covars_keys_ordered)
    ddim_latents, uncond_embeddings = null_inversion_class.invert(input_vec, conditioning_tensor, num_inner_steps=num_inner_steps, early_stop_epsilon=early_stop_epsilon, verbose=False, compute_uncond_embeddings=compute_uncond_embeddings)

    # detach all latents to free memory
    ddim_latents = [__lt.detach().cpu() for __lt in ddim_latents]
    if uncond_embeddings is not None:
      uncond_embeddings = [__uemb.detach().cpu() for __uemb in uncond_embeddings]
    noisy_latent = ddim_latents[-1]
    torch.cuda.empty_cache()

    # return noisy_latent, ddim_latents, uncond_embeddings
    return {
        "noisy_latents": noisy_latent,
        "ddim_latents": ddim_latents,
        "uncond_embeddings": uncond_embeddings
    }
