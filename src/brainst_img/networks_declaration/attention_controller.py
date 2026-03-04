

# prompt to prompt
import abc 
from typing import Union, Tuple, List, Callable, Dict, Optional
import torch.nn.functional as F

# pytorch
import torch
from torch.amp import GradScaler, autocast


# changing the forward method of the Attention class to inject the controller
def register_attention_control(model, controller, register_self=True):
    # visualize network architecture (if needed)
    def ca_forward_cross(self, place_in_unet):       
        def forward(x: torch.Tensor, context: Optional[torch.Tensor] = None):
            # print("ca_forward_cross: x.shape", x.shape, "context.shape", context.shape)
            # ca_forward_cross: x.shape torch.Size([4, 2304, 256]) context.shape torch.Size([4, 2, 256])

            # return x
            """
            Args:
                x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C
                context (torch.Tensor, optional): context tensor. B x (s_dim_1 * ... * s_dim_n) x C

            Return:
                torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
            """
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            b, t, c = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)

            q = self.input_rearrange(self.to_q(x))
            kv = context if context is not None else x
            _, kv_t, _ = kv.size()
            
            k = self.input_rearrange(self.to_k(kv))
            v = self.input_rearrange(self.to_v(kv))

            if self.attention_dtype is not None:
                q = q.to(self.attention_dtype)
                k = k.to(self.attention_dtype)

            # if self.use_flash_attention:
            #     x = torch.nn.functional.scaled_dot_product_attention(
            #         query=q, key=k, value=v, scale=self.scale, dropout_p=self.dropout_rate, is_causal=self.causal
            #     )
            # else:
            if True: # force to not use flash attention
                att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale
                # apply relative positional embedding if defined
                if self.rel_positional_embedding is not None:
                    att_mat = self.rel_positional_embedding(x, att_mat, q)

                if self.causal:
                    att_mat = att_mat.masked_fill(self.causal_mask[:, :, :t, :kv_t] == 0, float("-inf"))

                att_mat = att_mat.softmax(dim=-1)

                if self.save_attn:
                    # no gradients and new tensor;
                    # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                    self.att_mat = att_mat.detach()


                ########## ¡Aquí es donde se inyecta el controlador!
                # El controlador puede modificar o almacenar la atención.
                att_mat = controller(att_mat, True, place_in_unet)
                # att_mat = att_mat.view(b, self.num_heads, t, kv_t)
                # print("ca_forward_cross: att_mat", att_mat.shape, "place_in_unet", place_in_unet)
                # att_mat torch.Size([4, 8, 2304, 2])
                # continue with the normal attention calculation

                # att_mat = self.drop_weights(att_mat)
                x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
                # print("v.shape", v.shape)
                # v.shape torch.Size([4, 8, 2, 32])
                # print("cross att att_mat.sum:", att_mat.sum(), "x.sum()", x.sum(), "v.sum()", v.sum())

                
            x = self.out_rearrange(x)
            x = self.out_proj(x)
            x = self.drop_output(x)

            return x
        return forward


    def ca_forward_self(self, place_in_unet):
        def forward(x, attn_mask: Optional[torch.Tensor] = None):
            """
            Args:
                x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C
                attn_mask (torch.Tensor, optional): mask to apply to the attention matrix.
                B x (s_dim_1 * ... * s_dim_n). Defaults to None.

            Return:
                torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
            """
            if self.use_combined_linear:
                output = self.input_rearrange(self.qkv(x))
                q, k, v = output[0], output[1], output[2]
            else:
                q = self.input_rearrange(self.to_q(x))
                k = self.input_rearrange(self.to_k(x))
                v = self.input_rearrange(self.to_v(x))

            if self.attention_dtype is not None:
                q = q.to(self.attention_dtype)
                k = k.to(self.attention_dtype)

            if self.use_flash_attention:
                x = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=attn_mask,
                    scale=self.scale,
                    dropout_p=self.dropout_rate,
                    is_causal=self.causal,
                )
            else:
                att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale

                # apply relative positional embedding if defined
                if self.rel_positional_embedding is not None:
                    att_mat = self.rel_positional_embedding(x, att_mat, q)

                if self.causal:
                    if attn_mask is not None:
                        raise ValueError("Causal attention does not support attention masks.")
                    att_mat = att_mat.masked_fill(self.causal_mask[:, :, : x.shape[-2], : x.shape[-2]] == 0, float("-inf"))

                if attn_mask is not None:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                    attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
                    att_mat = att_mat.masked_fill(attn_mask == 0, float("-inf"))

                att_mat = att_mat.softmax(dim=-1)
                if self.save_attn:
                    # no gradients and new tensor;
                    # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                    self.att_mat = att_mat.detach()

                ########## ¡Aquí es donde se inyecta el controlador!
                # El controlador puede modificar o almacenar la atención.
                att_mat = controller(att_mat, False, place_in_unet)
                

                # continue with the normal attention calculation

                # att_mat = self.drop_weights(att_mat)
                x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
                # print("self att att_mat.sum:", att_mat.sum(), "x.sum()", x.sum(), "v.sum()", v.sum())


            x = self.out_rearrange(x)
            if self.include_fc:
                x = self.out_proj(x)
            x = self.drop_output(x)
            return x   
        return forward


    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ in ['CrossAttentionBlock', 'SABlock']:
            is_cross = net_.__class__.__name__ == 'CrossAttentionBlock' 
            if is_cross:
                net_.forward = ca_forward_cross(net_, place_in_unet)
                return count + 1
            else:
                if register_self:
                    net_.forward = ca_forward_self(net_, place_in_unet)
                    return count + 1
                else:
                    return count
            
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    nb_att_blocks = [0,0,0]
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
            nb_att_blocks[0] += cross_att_count - nb_att_blocks[1] - nb_att_blocks[2]
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
            nb_att_blocks[1] += cross_att_count - nb_att_blocks[0] - nb_att_blocks[2]
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
            nb_att_blocks[2] += cross_att_count - nb_att_blocks[0] - nb_att_blocks[1]

    print("Number of attention layers down: {0}, up: {1}, mid: {2}".format(*nb_att_blocks))

    controller.num_att_layers = cross_att_count
    




# Hereda de AttentionControl y se encarga de almacenar los mapas de atención en cada paso para su posterior análisis o visualización.
class AttentionStore:

    @staticmethod
    def get_empty_store():
        return {"down_cross": {}, "mid_cross": {}, "up_cross": {},
                "down_self": {},  "mid_self": {},  "up_self": {}}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        store_att_map = True
        if self.resolutions_list is not None and attn.shape[-2] not in self.resolutions_list:
            store_att_map = False

        if store_att_map and attn.shape[-2] not in self.step_store[key]:
            self.step_store[key][attn.shape[-2]] = []

        if self.evaluation_mode:
            # Se asume que la mitad superior del tensor de atención corresponde a la parte condicional (esto es porque se concatena el prompt no condicional con el no condicional y se hace en una sola pasada).
            h = attn.shape[0]
            # self.step_store[key].append(attn[h // 2:])
            if store_att_map:
                self.step_store[key][attn.shape[-2]].append(attn[h // 2:])

            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
        else:
            # self.step_store[key].append(attn)
            if store_att_map:
                self.step_store[key][attn.shape[-2]].append(attn)

        return attn

    def between_steps(self):
        # Se acumula la atención de cada paso.
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key_place_unet in self.attention_store:
                for key_res in (self.attention_store[key_place_unet]):
                    for i in range(len(self.attention_store[key_place_unet][key_res])):
                        # print("key_place_unet", key_place_unet, "key_res", key_res, "i", i)
                        # print(len(self.attention_store[key_place_unet][key_res]), len(self.step_store[key_place_unet][key_res]))
                        self.attention_store[key_place_unet][key_res][i] += self.step_store[key_place_unet][key_res][i]
                # for i in range(len(self.attention_store[key])):
                #     self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
 
    def get_average_attention(self): 
        # cuando la llamo self.cur_step = 50 (o por donde vaya) los que hace la media de todas las atenciones que se han guardado
        # average_attention = {key: [item / self.cur_step for item in self.attention_store[key]]
        #                      for key in self.attention_store}

        average_attention = {}
        for key_place_unet in self.attention_store:
            average_attention[key_place_unet] = {}
            for key_res in self.attention_store[key_place_unet]:
                average_attention[key_place_unet][key_res] = [
                    item / self.cur_step for item in self.attention_store[key_place_unet][key_res]
                ]
        return average_attention

    def reset(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.cur_step = 0
        self.cur_att_layer = 0


    def __init__(self, resolutions_list=None):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.evaluation_mode = False
        self.resolutions_list = resolutions_list

        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def eval(self):
        self.evaluation_mode = True

    def train(self):
        self.evaluation_mode = False



# def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, nb_prompts: int = 1):
#     out = []
#     attention_maps = attention_store.get_average_attention()
#     num_pixels = res[0] * res[1] * res[2] 
    
#     for location in from_where:
#         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]: # recorre cada capa de atencion (self o corss) guardada de la red en la ubicacion especifica (down, mid, up)
#             # print("item.shape 1", item.shape)
#             if item.shape[2] == num_pixels:
#                 # print("item.shape 2", item.shape) 
#                 # item.shape: [8, 256, 77] 
#                 # 8: lo mismo de las cabezas de atencion que se explico antes pero solo para el texto CONDICIONAL 
#                 # 256: es la imagen estirada 16x16
#                 # 77: numero maximo de tokens del texto

#                 cross_maps = item.reshape(nb_prompts, -1, res[0], res[1], res[2], item.shape[-1])
#                 # print("cross_maps.shape", cross_maps.shape)
#                 # cross_maps.shape: [1, 8, 16, 16, 77]
#                 # 1: numero de prompts
#                 # 8: lo mismo de las cabezas de atencion que se explico antes pero solo para el texto CONDICIONAL
#                 # 16,16: imagen
#                 # 77: numero maximo de tokens del texto

#                 cross_maps = cross_maps[select]
#                 out.append(cross_maps)

#     out = torch.cat(out, dim=0)
#     out = out.sum(0) / out.shape[0]
#     # print("out.shape",out.shape)
#     # out.shape: [16, 16, 77]
#     # 16,16: imagen
#     # 77: numero maximo de tokens del texto
    
#     return out.cpu()



def aggregate_attention(attention_store: AttentionStore, res: tuple[int], from_where: List[str], is_cross: bool, select: int, nb_prompts: int = 1):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res[0] * res[1] * res[2] 
    
    for location in from_where:
        maps_res = attention_maps[f"{location}_{'cross' if is_cross else 'self'}"][num_pixels]
        for item in maps_res:  # recorre cada capa de atencion (self o corss) guardada de la red en la ubicacion especifica (down, mid, up)
            # print("item.shape 1", item.shape)
        
            # print("item.shape 2", item.shape)  
            # item.shape: [8, 256, 77] 
            # 8: lo mismo de las cabezas de atencion que se explico antes pero solo para el texto CONDICIONAL 
            # 256: es la imagen estirada 16x16
            # 77: numero maximo de tokens del texto

            cross_maps = item.reshape(nb_prompts, -1, res[0], res[1], res[2], item.shape[-1])
            # print("cross_maps.shape", cross_maps.shape)
            # cross_maps.shape: [1, 8, 16, 16, 77]
            # 1: numero de prompts
            # 8: lo mismo de las cabezas de atencion que se explico antes pero solo para el texto CONDICIONAL
            # 16,16: imagen
            # 77: numero maximo de tokens del texto

            cross_maps = cross_maps[select]
            out.append(cross_maps)

    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    # print("out.shape",out.shape)
    # out.shape: [16, 16, 77]
    # 16,16: imagen
    # 77: numero maximo de tokens del texto
    
    return out.cpu()