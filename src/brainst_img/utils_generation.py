"""Generation-time utilities for BrainST-img: RNG seeding, condition-tensor
building, and structure-name helpers.
"""

import numpy as np
import torch


def gen_random_latents(latents_shape: tuple[int, ...] | int, seed: int = 42, device: torch.device | None = None, half: bool = False) -> torch.Tensor:
    """Sample a reproducible standard-normal noise tensor.

    Args:
        latents_shape: Shape of the tensor to sample (or a single int for
            a 1D tensor).
        seed: Seed for the (dedicated, CPU-based) generator used.
        device: Optional device to move the result to.
        half: If True, cast to fp16.

    Returns:
        The sampled noise tensor.
    """
    gen_randn = torch.Generator().manual_seed(seed) 
    latents = torch.randn(latents_shape, generator=gen_randn)
    if half:
        latents = latents.half()
    if device is not None:
        latents = latents.to(device)
    return latents


def prepare_condition_tensor(conditions_list_dict: dict | list[dict], conditions_keys_ordered: list[str]) -> torch.Tensor:
    """Build a batched ROI-volume condition tensor from one or more condition dicts.

    Args:
        conditions_list_dict: A single ROI-volume condition dict, or a
            list of them (one per batch element).
        conditions_keys_ordered: Ordered condition/structure keys.

    Returns:
        Float tensor of shape ``(batch, 1, num_conditions)`` -- the
        middle dimension matches the "sequence length 1" expected by the
        cross-attention conditioning model.
    """
    # verify if conditions_list_dict is a list of dictionaries
    if isinstance(conditions_list_dict, dict):
        conditions_list_dict = [conditions_list_dict]
        
    cond_list = np.zeros((len(conditions_list_dict), len(conditions_keys_ordered)))
    for i in range(len(conditions_list_dict)):
        for j in range(len(conditions_keys_ordered)):
            cond_list[i,j] = conditions_list_dict[i][conditions_keys_ordered[j]]
    conditioning = torch.tensor(cond_list).float().unsqueeze(1).permute(0,2,1)
    return conditioning
