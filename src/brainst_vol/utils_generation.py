"""Generation-time utilities for BrainST-vol: RNG seeding and condition-tensor building."""

import torch
import torch.nn.functional as F


def gen_random_latents(
    latents_shape: tuple[int, ...] | int,
    seed: int = 42,
    device: torch.device | None = None,
    half: bool = False,
) -> torch.Tensor:
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
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(latents_shape, generator=generator)
    if half:
        latents = latents.half()
    if device is not None:
        latents = latents.to(device)
    return latents


def prepare_condition_tensor(
    covars_list_dict: dict | list[dict],
    covars_list: list[str] = ["age", "sex", "dx"],
) -> torch.Tensor:
    """Build a batched covariates tensor from one or more covariate dicts.

    Args:
        covars_list_dict: A single covariates dict, or a list of them
            (one per batch element).
        covars_list: Ordered covariate keys to include. Special handling:
            ``"age"``/``"sex"`` are used as raw scalars; ``"dx"`` is
            one-hot encoded (3 classes).

    Returns:
        Float tensor of shape ``(batch, num_covar_features)`` where
        ``num_covar_features`` depends on which keys are present (e.g.
        age + sex + one-hot dx = 1 + 1 + 3 = 5).
    """
    if isinstance(covars_list_dict, dict):
        covars_list_dict = [covars_list_dict]

    per_subject_features = []
    for covars in covars_list_dict:
        subject_features = []
        for key in covars_list:
            if key == "age":
                subject_features.append(torch.tensor([[covars["age"]]]))
            elif key == "sex":
                subject_features.append(torch.tensor([[covars["sex"]]]))
            elif key == "dx":
                subject_features.append(F.one_hot(torch.tensor([covars["dx"]]), num_classes=3))
        per_subject_features.append(subject_features)

    row_tensors = [torch.cat(row, dim=1) for row in per_subject_features]
    return torch.cat(row_tensors, dim=0).float()