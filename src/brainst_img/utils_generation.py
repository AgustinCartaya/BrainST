
import numpy as np
import torch
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

def prepare_condition_tensor(conditions_list_dict, conditions_keys_ordered):
    # verify if conditions_list_dict is a list of dictionaries
    if isinstance(conditions_list_dict, dict):
        conditions_list_dict = [conditions_list_dict]
        
    cond_list = np.zeros((len(conditions_list_dict), len(conditions_keys_ordered)))
    for i in range(len(conditions_list_dict)):
        for j in range(len(conditions_keys_ordered)):
            cond_list[i,j] = conditions_list_dict[i][conditions_keys_ordered[j]]
    conditioning = torch.tensor(cond_list).float().unsqueeze(1).permute(0,2,1)
    return conditioning

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



# import sys
# # sys.path.append('/home/agustin/phd/synthesis/tests/D3/maisi/results_evaluation/generated/test3_conditioned_prediction_only_clean_code/scripts/src')
# sys.path.append('/home/agustin/phd/synthesis')
# import utils.data_normalization as data_normalization

# def get_normalized_used_conditions_df(model_description, real_conditions_df):
#     # load most complete dataset:
#     complete_brain_structures_normalized_icv = "/home/agustin/phd/synthesis/data_analysers/results/aaco5590_dataset_zscore/brain_statistics_normalized_individually.csv"
#     complete_dataset = "/home/agustin/phd/synthesis/data_analysers/results/aaco5590_dataset_zscore/aaco5590_dataset_no_outliers_splitted.csv"
#     complete_dataset_df = pd.read_csv(complete_dataset)
#     complete_brain_structures_normalized_icv_df = pd.read_csv(complete_brain_structures_normalized_icv)
#     complete_df = pd.merge(complete_dataset_df, complete_brain_structures_normalized_icv_df, on=["id", "session_id"], how="inner")

#     # obtain just the ids in real_conditions_df
#     used_df = pd.merge(real_conditions_df[["cond_id", "id", "session_id"]], complete_df, on=["id", "session_id"], how="inner")
#     normalizer = data_normalization.SavedNormalizerBrainStructures(model_description.normalizer_params)
#     normalized_used_df = normalizer.transform(used_df)
#     return normalized_used_df



# def normalize_brain_statistics_individually(brain_stats_df, structure_names, icv_column="total_vol"):
#     res_list = []
#     for i, row in brain_stats_df.iterrows():
#         _row_res = [-1] * len(structure_names)
#         s_id, session_id = row["id"], row["session_id"]

#         if row[icv_column] > 0:
#             total_volume = row[icv_column]
#             for j, s in enumerate(structure_names):
#                 if s == icv_column:
#                     _row_res[j] = row[s]
#                 else:
#                     _row_res[j] = row[s] / total_volume
#         res_list.append([s_id, session_id] + _row_res)
#     columns = ["id", "session_id"] + structure_names
#     data_frame = pd.DataFrame(res_list, columns=columns)
#     return data_frame