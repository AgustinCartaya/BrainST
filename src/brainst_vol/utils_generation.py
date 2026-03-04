
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

def set_seed(seed: int):
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista



def gen_random_latents(latents_shape, seed=42, device=None, half=False):
    gen_randn = torch.Generator().manual_seed(seed) 
    latents = torch.randn(latents_shape, generator=gen_randn)
    if half:
        latents = latents.half()
    if device is not None:
        latents = latents.to(device)
    return latents

def prepare_condition_tensor(covars_list_dict, covars_list=["age", "sex", "dx"]):
    # verify if conditions_list_dict is a list of dictionaries
    if isinstance(covars_list_dict, dict):
        covars_list_dict = [covars_list_dict]

    tensor_list = []
    for i, covars in enumerate(covars_list_dict):
        item_list = []
        for j, key in enumerate(covars_list):
            if key == "age":
                item_list.append(torch.tensor([[covars["age"]]]))
            elif key == "sex":
                item_list.append(torch.tensor([[covars["sex"]]]))
            elif key == "dx":
                item_list.append(F.one_hot(torch.tensor([covars["dx"]]), num_classes=3))
        tensor_list.append(item_list)
        # tensor_list.append([torch.tensor([[covars["age"]]]), torch.tensor([[covars["sex"]]]), F.one_hot(torch.tensor([covars["dx"]]), num_classes=3)])
    row_tensors = [torch.cat(row, dim=1) for row in tensor_list]  # cada elemento es un Tensor
    tensor = torch.cat(row_tensors, dim=0).float()
    return tensor


def is_increasing_structure(brain_structure):
    # Define the structures that are considered to be increasing
    increase_structures = [
                            "csf_vol","surrounding_csf_vol","cerebral_ventricles_vol","lateral_ventricles_vol","left_lateral_ventricle_vol","right_lateral_ventricle_vol","fourth_ventricle_vol","third_ventricle_vol","inferior_lateral_ventricles_vol",
                            "surrounding_csf_no_sulcal_csf_c0_vol","sulcal_csf_c0_vol","total_cerebral_no_surrounding_csf_vol","cerebral_surrounding_csf_vol","cerebral_surrounding_csf_no_sulcal_csf_c0_vol","cerebral_sulcal_csf_c0_vol","cerebellum_brainstem_csf_vol",
                            "surrounding_csf_no_sulcal_csf_c0_x2_vol","sulcal_csf_c0_x2_vol","cerebral_surrounding_csf_no_sulcal_csf_c0_x2_vol","cerebral_sulcal_csf_c0_x2_vol",
                            "age"
                           ]
    if brain_structure in increase_structures:
        return True
    return False

def get_moving_conditions(conditions_keys_ordered, static_conditions=None):
    if static_conditions is None:
        static_conditions = {}
    moving_conditions = []
    for key in conditions_keys_ordered:
        if key not in static_conditions:
            moving_conditions.append(key)
    return moving_conditions

def order_moving_conditions(conditions_keys_ordered, moving_conditions=None):

    if moving_conditions is None:
        return conditions_keys_ordered
    ordered_moving_conditions = []
    for key in conditions_keys_ordered:
        if key in moving_conditions:
            ordered_moving_conditions.append(key)
    return ordered_moving_conditions

def acronim_conditions(condition):
    acronim_dict = {
        "total_vol": "total",
        "surrounding_csf_vol": "surcsf",
        "surrounding_csf_no_sulcal_csf_c0_vol": "surcsfns",
        "sulcal_csf_c0_vol": "sulcalcsf",
        "cortical_gm_vol": "cortgm",
        "cerebral_wm_vol": "cerebwm",
        "lateral_ventricles_vol": "vent",

        # legacy
        # "ventricular_vol": "vent",
        # "surrounding_csf": "surcsf",
        # "surrounding_csf_no_sulcal_csf_vol": "surcsfns",
        # "sulcal_csf_vol": "sulcalcsf",
    }
    if condition in acronim_dict:
        return acronim_dict[condition]
    return condition





def find_closest_rows(dataset_df, asked_conditions_list):
    """
    dataset_df: DataFrame con tus sujetos
    asked_conditions_list: lista de diccionarios con valores objetivo
    condition_columns: lista de columnas a considerar en la distancia
    """
    closest_rows = []
    condition_columns = list(asked_conditions_list[0].keys())

    # Convertir dataset a numpy para eficiencia
    X = dataset_df[condition_columns].to_numpy()

    for cond_dict in asked_conditions_list:
        # convertir dict a array
        target = np.array([cond_dict[col] for col in condition_columns])

        # calcular distancias Euclidianas
        distances = np.linalg.norm(X - target, axis=1)

        # encontrar índice del más cercano
        closest_idx = distances.argmin()
        closest_rows.append(dataset_df.iloc[closest_idx])

    # devolver DataFrame con los más cercanos
    return pd.DataFrame(closest_rows)

