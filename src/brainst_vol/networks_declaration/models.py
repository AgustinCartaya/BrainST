import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Embedding Posicional (sinusoidal)
# ------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
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
                 covar_dimension, 
                 dim_t, 
                 covar_embed_dim=128, 
                 conditioning_type="add"):
        super().__init__()
        self.conditioning_type = conditioning_type
        self.dim_t = dim_t
        self.use_covars = covar_dimension > 0

        if not self.use_covars:
            return

        if conditioning_type == "add":
            self.embed = nn.Sequential(
                nn.Linear(covar_dimension, covar_embed_dim),
                nn.SiLU(),
                nn.Linear(covar_embed_dim, dim_t)
            )

        elif conditioning_type == "film":
            self.embed = nn.Sequential(
                nn.Linear(covar_dimension, covar_embed_dim),
                nn.SiLU(),
                nn.Linear(covar_embed_dim, 2 * dim_t)  # gamma y beta
            )

        elif conditioning_type == "crossattn":
            self.embed = nn.Sequential(
                nn.Linear(covar_dimension, covar_embed_dim),
                nn.SiLU(),
                nn.Linear(covar_embed_dim, dim_t)
            )
            self.attn = nn.MultiheadAttention(embed_dim=dim_t, num_heads=4, batch_first=True)

        else:
            raise ValueError(f"conditioning_type '{conditioning_type}' no reconocido.")


    def forward(self, covars):
        return self.embed(covars)

# ------------------------------------------------------------
# Modelo MLP Diffusion 1D modular
# ------------------------------------------------------------
class MLPDiffusion(nn.Module):
    def __init__(self,
                 d_in, 
                 dim_t=512, 
                 num_heads=4,
                 conditioning_type="add",
                 merge_conditioning_with="x"): # input, middle, output):
        super().__init__()
        # Configuración del modelo desde diccionario
        self.d_in = d_in
        self.dim_t = dim_t
        self.conditioning_type = conditioning_type

        self.dim_t = dim_t
        self.merge_conditioning_with = merge_conditioning_with

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

        if conditioning_type == "crossattn":
            self.attn = nn.MultiheadAttention(embed_dim=dim_t, num_heads=num_heads, batch_first=True)


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

        # Conditioning
        # emb = self.conditioning(emb, context)
        if context is not None:
            if self.conditioning_type == "add":
                if self.merge_conditioning_with == "x":
                    x = x + context
                elif self.merge_conditioning_with == "time_emb":
                    emb = emb + context
            elif self.conditioning_type == "film":
                gamma, beta = torch.chunk(context, 2, dim=-1)
                if self.merge_conditioning_with == "x":
                    x = x * (1 + gamma) + beta
                elif self.merge_conditioning_with == "time_emb":
                    emb = emb * (1 + gamma) + beta
            elif self.conditioning_type == "crossattn":
                c_emb = context.unsqueeze(1)  # [B, 1, dim_t]

                if self.merge_conditioning_with == "x":
                    query = x.unsqueeze(1)  # [B, 1, dim_t]
                    out, _ = self.attn(query, c_emb, c_emb)
                    x = x + out.squeeze(1)
                elif self.merge_conditioning_with == "time_emb":
                    query = emb.unsqueeze(1)  # [B, 1, dim_t]
                    out, _ = self.attn(query, c_emb, c_emb)
                    emb = emb + out.squeeze(1)
            


        # Procesamiento principal
        x = x + emb
        out = self.mlp(x)
        return out


