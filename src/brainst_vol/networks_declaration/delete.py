import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Embedding Posicional (sinusoidal)
# ------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels=512, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x[:, None] * freqs[None, :]
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


# ------------------------------------------------------------
# Módulo de Conditioning
# ------------------------------------------------------------
class ConditioningModule(nn.Module):
    def __init__(self, 
                 covar_dimension=18, 
                 dim_t=512, 
                 covar_embed_dim=128):
        super().__init__()
        self.dim_t = dim_t

        self.embed = nn.Sequential(
            nn.Linear(covar_dimension, covar_embed_dim),
            nn.SiLU(),
            nn.Linear(covar_embed_dim, dim_t)
        )

    def forward(self, covars):
        return self.embed(covars)

# ------------------------------------------------------------
# Modelo MLP Diffusion 1D modular
# ------------------------------------------------------------
class MLPDiffusion(nn.Module):
    def __init__(self,
                 d_in=18, 
                 dim_t=512): # input, middle, output):
        super().__init__()
        # Configuración del modelo desde diccionario
        self.d_in = d_in
        self.dim_t = dim_t
        self.dim_t = dim_t

        # --- Embedding temporal ---
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        # --- Proyección inicial ---
        self.proj = nn.Linear(d_in, dim_t)

        # --- Red principal ---
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x, timesteps, context=None):
        """
        Args:
            x: tensor [B, d_in]
            timesteps: tensor [B]
            context: tensor [B, ConditioningModule_dimension]
        """
        # Embedding temporal
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)


        x = self.proj(x)
        x = x + context
        
        # Procesamiento principal
        x = x + emb
        out = self.mlp(x)
        return out


