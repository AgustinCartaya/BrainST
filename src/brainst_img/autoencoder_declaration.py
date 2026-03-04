from monai.bundle import ConfigParser
import torch
from monai.utils import set_determinism



from monai.inferers import sliding_window_inference
import contextlib

def instantiate_autoencoder(chk_path_name, device, half=True):

    networks_config =  {
        "autoencoder_def": {
            "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "num_channels": [
                64,
                128,
                256
            ],
            "num_res_blocks": [2,2,2],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [
                False,
                False,
                False
            ],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": half,
            "num_splits": 8,
            "dim_split": 1
        },

    }

    # instantiate model
    parser = ConfigParser(networks_config)
    parser.parse(True)

    # autoencoder (just for validation)
    autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(chk_path_name, weights_only=True, map_location=device)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    return autoencoder



def prepare_image_to_encode(image, device):
    image = torch.tensor(image).to(device)
    image = image.unsqueeze(0).unsqueeze(0)
    return image

class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder, scale_factor=1.0):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


class AutoencoderPrediction:
    def __init__(self, chk_path_name, device, half=True):
        self.autoencoder = instantiate_autoencoder(chk_path_name, device, half=half)
        self.device = device
        self.half = half

    @torch.no_grad()
    def encode(self, image, seed=0):
        # verify if image is a tensor
        if not torch.is_tensor(image):
            image = prepare_image_to_encode(image, self.device)

        if seed is not None:
            set_determinism(seed=seed)
        ctx = torch.amp.autocast("cuda") if self.half else contextlib.nullcontext()
        image = image.half() if self.half else image
        with ctx:
            res = self.autoencoder.encode_stage_2_inputs(image)
        return res
    
    @torch.no_grad()
    def decode(self, latents, decode_complete=True, sliding_window_size=(48, 48, 48), overlap=0.25, seed=0):
        if seed is not None:
            set_determinism(seed=seed)
        ctx = torch.amp.autocast("cuda") if self.half else contextlib.nullcontext()
        # latents = latents.half() if self.half else latents
        with ctx:
            if decode_complete:
                res = self._decode_complete(latents)
            else:
                res = self._decode_by_patches(latents, sliding_window_size=sliding_window_size, overlap=overlap)
        return res

    def _decode_complete(self, latents):
        res = self.autoencoder.decode_stage_2_outputs(latents)
        return res

    def _decode_by_patches(self, latents, sliding_window_size=(48, 48, 48), overlap=0.25):
        recon_model = ReconModel(self.autoencoder)
        spatial_shape = latents.shape[2:]  # (W, H, D)
        roi_size = tuple(min(spatial_shape[i], sliding_window_size[i]) for i in range(len(spatial_shape)))

        res = sliding_window_inference(
            inputs=latents,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=recon_model,
            mode="gaussian",
            overlap=overlap,
            sw_device=latents.device,
            device=latents.device,
            progress=True,
        )
        return res
    
    



import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLPIPS(nn.Module):
    def __init__(self, encoder, layer_idxs=None, reduction="mean", device=None, half=True):
        """
        encoder: instancia de tu Encoder entrenado
        layer_idxs: lista de índices de bloques en encoder.blocks que quieres usar
        reduction: 'mean', 'sum', o None
        """
        super().__init__()
        if isinstance(encoder, str) and device is not None:
            encoder = AutoencoderPrediction(encoder, device, half=half).autoencoder.encoder

        self.encoder = encoder.eval()
        self.features = {}
        self.reduction = reduction
        self.device = next(encoder.parameters()).device
        self.half = half


        # si no pasas layer_idxs, elegimos puntos clave
        if layer_idxs is None:
            # por ejemplo, después de cada resolución + último
            layer_idxs = [2, 5, 8, 10]

        self.layer_idxs = layer_idxs

        # crear hooks
        for idx in self.layer_idxs:
            self.encoder.blocks[idx].register_forward_hook(self.save_activation(idx))

    def save_activation(self, idx):
        def hook(module, input, output):
            self.features[idx] = output
        return hook

    def normalize_3d_features(self, feats):
        # ---- LPIPS normalization
        # """Normalización optimizada para características 3D"""
        # B, C, D, H, W = feats.shape
        # feats_flat = feats.view(B, C, -1)
        # feats_norm = F.normalize(feats_flat, p=2, dim=2)
        # return feats_norm.view(B, C, D, H, W)

        # ---- my/chat normalization
        return F.normalize(feats, dim=1)

    def forward(self, x, y):
        if not torch.is_tensor(x):
            x = prepare_image_to_encode(x, self.device)
        if not torch.is_tensor(y):
            y = prepare_image_to_encode(y, self.device)

        # opcional: castear antes
        # if self.half:
        #     x = x.half()
        #     y = y.half()

        def get_feats(img):
            self.features = {}
            with torch.no_grad():
                with torch.amp.autocast("cuda") if self.half else contextlib.nullcontext():
                    _ = self.encoder(img)
            # detach + mover a CPU para liberar GPU
            feats = {k: v.detach().float().cpu() for k, v in self.features.items()}
            return feats

        feats_x = get_feats(x)
        feats_y = get_feats(y)

        dists = []
        for k in self.layer_idxs:
            fx, fy = feats_x[k], feats_y[k]

            # normalización canal-wise
            # fx = F.normalize(fx, dim=1)
            # fy = F.normalize(fy, dim=1)
            fx = self.normalize_3d_features(fx)
            fy = self.normalize_3d_features(fy)

            # distancia L2 promedio batch-wise
            dist = (fx - fy).pow(2).mean(dim=[1, 2, 3, 4])  # [batch]
            dists.append(dist)

        dists = torch.stack(dists, dim=1)  # [batch, n_layers]
        torch.cuda.empty_cache()
        if self.reduction == "mean":
            return dists.mean(dim=1).float()
        elif self.reduction == "sum":
            return dists.sum(dim=1).float()
        else:
            return dists  # devuelve un score por capa

    def compute_one_multiple(self, x, y_list):
        """
        Calcula la distancia LPIPS entre una imagen fija x y varias imágenes en y_list.
        
        Args:
            x: tensor o imagen a codificar una sola vez [B, C, D, H, W]
            y_list: lista de tensores/imágenes [N, C, D, H, W]
            half: usar float16 para menor consumo de memoria
        Returns:
            dists: tensor [len(y_list)] con la distancia de x con cada y
        """
        if not torch.is_tensor(x):
            x = prepare_image_to_encode(x, self.device)

        # Castear x
        # if half:
        #     x = x.half()

        # Extraer features de x solo una vez
        self.features = {}
        with torch.no_grad():
            with torch.amp.autocast("cuda") if self.half else contextlib.nullcontext():
                _ = self.encoder(x)
        feats_x = {k: v.detach().float().cpu() for k, v in self.features.items()}

        # normalize features of x
        for k in feats_x:
            # feats_x[k] = F.normalize(feats_x[k], dim=1)
            feats_x[k] = self.normalize_3d_features(feats_x[k])

        # Lista para resultados
        all_dists = []

        for y in y_list:
            if not torch.is_tensor(y):
                y = prepare_image_to_encode(y, self.device)
            # if half:
            #     y = y.half()

            # Extraer features de y
            self.features = {}
            with torch.no_grad():
                with torch.amp.autocast("cuda") if self.half else contextlib.nullcontext():
                    _ = self.encoder(y)
            feats_y = {k: v.detach().float().cpu() for k, v in self.features.items()}

            # Calcular distancia LPIPS
            dists = []
            for k in self.layer_idxs:
                fx, fy = feats_x[k], feats_y[k]
                # fy = F.normalize(fy, dim=1)
                fy = self.normalize_3d_features(fy)
                dist = (fx - fy).pow(2).mean(dim=[1, 2, 3, 4])
                dists.append(dist)

            dists = torch.stack(dists, dim=1)
            if self.reduction == "mean":
                dists = dists.mean(dim=1).float()
            elif self.reduction == "sum":
                dists = dists.sum(dim=1).float()

            all_dists.append(dists)

        # Concatenar resultados [N]
        all_dists = torch.cat(all_dists, dim=0)
        torch.cuda.empty_cache()
        return all_dists


