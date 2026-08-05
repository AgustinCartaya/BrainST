# =============================================================================
# autoencoder_utils.py
#
# Utilities for loading, running, and computing perceptual similarity with a
# MAISI KL-autoencoder (MONAI).  Two public entry points:
#   - AutoencoderPrediction : encode / decode 3-D medical images
#   - EncoderLPIPS          : LPIPS-style perceptual distance in latent space
# =============================================================================

import contextlib

import torch
from monai.bundle import ConfigParser
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def instantiate_autoencoder(chk_path_name, device, half=True):
    """
    Build and return a MAISI KL-autoencoder loaded from a checkpoint.

    The architecture is defined inline as a MONAI ConfigParser dict so that
    no external config file is required.

    Args:
        chk_path_name (str | Path): Path to the .pt checkpoint file.
        device (torch.device):      Target device (CPU or CUDA).
        half (bool):                If True, enable float16 normalisation layers
                                    inside the autoencoder (norm_float16).

    Returns:
        autoencoder (nn.Module): Loaded model in eval mode, on `device`.
    """
    networks_config = {
        "autoencoder_def": {
            "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
            "spatial_dims": 3,          # 3-D volumes
            "in_channels": 1,           # single-channel input (e.g. CT/MRI)
            "out_channels": 1,
            "latent_channels": 4,       # bottleneck channel depth
            "num_channels": [64, 128, 256],          # encoder/decoder widths per level
            "num_res_blocks": [2, 2, 2],              # residual blocks per level
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [False, False, False],# no self-attention (memory saving)
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": half,       # cast norm layers to fp16 when half=True
            "num_splits": 8,            # split factor for memory-efficient computation
            "dim_split": 1,             # dimension along which to split
        }
    }

    # Parse the config dict and instantiate the network
    parser = ConfigParser(networks_config)
    parser.parse(True)

    # Move model to the target device and load weights
    autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(chk_path_name, weights_only=True, map_location=device)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    return autoencoder


# ---------------------------------------------------------------------------
# Tensor preparation helpers
# ---------------------------------------------------------------------------

def prepare_image_to_encode(image, device):
    """
    Convert a raw numpy array (or array-like) image to a 5-D batch tensor.

    Adds a batch dimension and a channel dimension so the output shape is
    [1, 1, D, H, W], ready for the autoencoder encoder.

    Args:
        image: Array-like of shape (D, H, W).
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Shape [1, 1, D, H, W] on `device`.
    """
    image = torch.tensor(image).to(device)
    image = image.unsqueeze(0).unsqueeze(0)  # -> [1, 1, D, H, W]
    return image


def prepare_latent_to_encode(latent, device):
    """
    Convert a raw numpy latent array to a 5-D batch tensor.

    Adds only a batch dimension; the channel dim is already present in the
    latent representation. Output shape: [1, C, D, H, W].

    Args:
        latent: Array-like of shape (C, D, H, W).
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Shape [1, C, D, H, W] on `device`.
    """
    latent = torch.tensor(latent).to(device)
    latent = latent.unsqueeze(0)  # -> [1, C, D, H, W]
    return latent


# ---------------------------------------------------------------------------
# Thin wrapper around the autoencoder for encode / decode
# ---------------------------------------------------------------------------

class ReconModel(torch.nn.Module):
    """
    Lightweight nn.Module wrapper around the MAISI autoencoder that applies
    an optional scale factor to the latent space.

    Using a scale factor can help normalise the magnitude of latents and
    improve downstream diffusion training stability.

    Args:
        autoencoder (nn.Module): Pretrained MAISI autoencoder.
        scale_factor (float):   Multiplicative scale applied to latents on
                                encode and inverted on decode. Default 1.0
                                (no scaling).
        deterministic (bool):   If True, use the deterministic `encode` path
                                (returns mean only, discards log-variance).
                                If False, use `encode_stage_2_inputs` which
                                samples from the posterior.
    """

    def __init__(self, autoencoder, scale_factor=1.0, deterministic=False):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor
        self.deterministic = deterministic

    def encode(self, x):
        """
        Encode an image to its latent representation and apply the scale factor.

        Args:
            x (torch.Tensor): Image tensor [B, 1, D, H, W].

        Returns:
            torch.Tensor: Scaled latent [B, C_lat, D', H', W'].
        """
        if self.deterministic:
            # Deterministic path: return posterior mean, ignore log-variance
            z, _ = self.autoencoder.encode(x)
        else:
            # Stochastic path: sample from q(z|x)
            z = self.autoencoder.encode_stage_2_inputs(x)
        return z * self.scale_factor

    def forward(self, z):
        """
        Decode a (possibly scaled) latent back to image space.

        This is the method called by sliding_window_inference, so it must
        accept a single tensor argument.

        Args:
            z (torch.Tensor): Scaled latent [B, C_lat, D', H', W'].

        Returns:
            torch.Tensor: Reconstructed image [B, 1, D, H, W].
        """
        recon_pt_nda = self.autoencoder.decode(z / self.scale_factor)
        return recon_pt_nda


# ---------------------------------------------------------------------------
# High-level encode / decode API
# ---------------------------------------------------------------------------

class AutoencoderPrediction:
    """
    High-level wrapper for encoding and decoding 3-D medical images with the
    MAISI KL-autoencoder.

    Supports:
      - Full-volume encoding / decoding (small volumes that fit in GPU memory).
      - Patch-based decoding via sliding-window inference (large volumes).

    Args:
        chk_path_name (str | Path): Path to the autoencoder checkpoint.
        device (torch.device):      Inference device.
        half (bool):                Use fp16 autocast during inference.
        scale_factor (float):       Latent scale factor forwarded to ReconModel.
        deterministic (bool):       If True, use the deterministic `encode` path
                                    (returns mean only, discards log-variance).
                                    If False, use `encode_stage_2_inputs` which
                                    samples from the posterior.
    """

    def __init__(self, chk_path_name, device, half=True, scale_factor=1.0, deterministic=False):
        self.autoencoder = instantiate_autoencoder(chk_path_name, device, half=half)
        self.device = device
        self.half = half
        self.scale_factor = scale_factor
        self.recon_model = ReconModel(self.autoencoder, scale_factor=self.scale_factor, deterministic=deterministic)

    @torch.no_grad()
    def encode(self, image, seed=0):
        """
        Encode a 3-D image volume to its latent representation.

        Args:
            image (np.ndarray | torch.Tensor): Shape (D, H, W) or
                [1, 1, D, H, W] if already a batch tensor.
            seed (int | None): Random seed for deterministic behaviour.
                               Pass None to skip seeding.

        Returns:
            torch.Tensor: Latent tensor [1, C_lat, D', H', W'] on `self.device`.
        """
        if not torch.is_tensor(image):
            image = prepare_image_to_encode(image, self.device)

        if seed is not None:
            set_determinism(seed=seed)

        # Use fp16 autocast on CUDA for speed / memory; no-op on CPU
        ctx = torch.amp.autocast("cuda") if self.half else contextlib.nullcontext()
        image = image.half() if self.half else image
        with ctx:
            res = self.recon_model.encode(image)
        return res

    @torch.no_grad()
    def decode(self, latents, decode_complete=True,
               sliding_window_size=(48, 48, 48), overlap=0.25, seed=0):
        """
        Decode a latent tensor back to image space.

        Two strategies are available:
          - Full decode (decode_complete=True): decode the entire latent at
            once; fast but may OOM for large volumes.
          - Patch decode (decode_complete=False): use sliding-window inference
            over latent patches; slower but memory-efficient.

        Args:
            latents (np.ndarray | torch.Tensor): Shape (C, D', H', W') or
                [1, C, D', H', W'] if already a batch tensor.
            decode_complete (bool):          Full-volume decode when True.
            sliding_window_size (tuple[int]): (D, H, W) patch size for patch
                                              decode. Clamped to latent shape.
            overlap (float):                 Fractional overlap between patches
                                             (0-1) for the sliding window.
            seed (int | None):               Random seed; pass None to skip.

        Returns:
            torch.Tensor: Reconstructed image [1, 1, D, H, W].
        """
        if not torch.is_tensor(latents):
            latents = prepare_latent_to_encode(latents, self.device)

        if seed is not None:
            set_determinism(seed=seed)

        ctx = torch.amp.autocast("cuda") if self.half else contextlib.nullcontext()
        with ctx:
            if decode_complete:
                reconstructed_image = self._decode_complete(latents)
            else:
                reconstructed_image = self._decode_by_patches(
                    latents, sliding_window_size=sliding_window_size, overlap=overlap
                )
        return reconstructed_image

    def _decode_complete(self, latents):
        """Decode the full latent volume in a single forward pass."""
        return self.recon_model(latents)

    def _decode_by_patches(self, latents, sliding_window_size=(48, 48, 48), overlap=0.25):
        """
        Decode a latent volume patch-by-patch using MONAI's sliding-window
        inference with Gaussian blending at patch boundaries.

        The roi_size is clamped per axis so it never exceeds the actual latent
        spatial dimensions (avoids zero-padding artefacts on small volumes).

        Args:
            latents (torch.Tensor): [1, C, D', H', W'].
            sliding_window_size (tuple[int]): Desired (D, H, W) patch size.
            overlap (float):                 Fractional overlap (0–1).

        Returns:
            torch.Tensor: Reconstructed image [1, 1, D, H, W].
        """
        spatial_shape = latents.shape[2:]  # (D, H, W) of the latent
        # Clamp each axis so roi never exceeds the available spatial extent
        roi_size = tuple(
            min(spatial_shape[i], sliding_window_size[i])
            for i in range(len(spatial_shape))
        )

        res = sliding_window_inference(
            inputs=latents,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=self.recon_model,  # called as predictor(patch) -> recon patch
            mode="gaussian",             # Gaussian weighting reduces boundary artefacts
            overlap=overlap,
            sw_device=latents.device,    # run inference on same device as input
            device=latents.device,       # accumulate results on same device
            progress=True,
        )
        return res

