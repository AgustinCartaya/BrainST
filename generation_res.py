

import os
import numpy as np
import pandas as pd
import json
# from tqdm import tqdm
# import nibabel as nib


# import utils.nifti_functions as nfc
# import utils.util as util
# import utils.data_normalization as data_normalization

# import evaluation.arppoaches.our.src.instantiate_models import instantiate_model_and_load
# import evaluation.arppoaches.our.src.load_data import get_test_ids, load_timepoint_data
# import evaluation.arppoaches.our.src.null_inversion import invert_latents
# import evaluation.arppoaches.our.src.generate_image import recover_image_from_noisy_latents

import src.utils.functions as fc
import src.utils.nifti_functions as nfc
import src.utils.data_normalization as data_normalization

import src.preprocessing.prep_volumes as prep_volumes
import src.preprocessing.prep_images as prep_images


import src.brainst_img.instantiate_models as instantiate_brainst_img
import src.brainst_img.generate_image as generate_image
import src.brainst_img.null_inversion as null_inversion
import src.brainst_img.utils_generation as utils_generation


import src.brainst_vol.instantiate_models as instantiate_brainst_vol
import src.brainst_vol.generate_volumes as generate_volumes
import src.brainst_vol.null_inversion as null_inversion_volumes



import torch
device_name = f"cuda:0"
device = torch.device(device_name)


def load_parameters(model_description_path):
    # load model_description_path as a json
    with open(model_description_path, 'r') as f:
        model_description = json.load(f)
    model_description = fc.dict_to_args(model_description, deep_conversion=True)
    return model_description



def brainst_img_synthesis(brainst_img, target_roi_volumes_dict, seed=2):
    conditions_keys_ordered = brainst_img["model_description"].args_train.conditions_keys_ordered
    
    # 1 generate random noise
    latents_shape = (1,4,48,64,48)
    noisy_latents = utils_generation.gen_random_latents(latents_shape, seed=seed, device=device)

    # 3 create image
    reconstrcuted_latents = generate_image.diffusion_loop(
        noisy_latents,
        brainst_img["unet"],
        brainst_img["conditions_model"],
        brainst_img["noise_scheduler"],
        brainst_img["autoencoder"],
        [target_roi_volumes_dict],
        conditions_keys_ordered,
        uncond_embeddings=None,
        free_guidance_ratio=2.0,
        decode_img=True, # to be removed
        decode_first=True, # to be removed
        decode_complete=True, 
    )

    # 4 save the reconstructed image
    recon_img = reconstrcuted_latents["images"][0].astype(np.float32)
    return recon_img


def brainst_vol_synthesis(brainst_vol, covariates_dict, seed=2):
    covars_keys_ordered = brainst_vol["model_description"].args_train.covars_list
    conditions_keys_ordered = brainst_vol["model_description"].args_train.conditions_keys_ordered
    
    # 2 generate random noise
    latents_shape = (1, len(conditions_keys_ordered))
    initial_noise = utils_generation.gen_random_latents(latents_shape, seed=seed, device=device)
    
    # 3 create volumes
    prediction = generate_volumes.diffusion_loop( 
                            initial_noise, 
                            brainst_vol["unet"],
                            brainst_vol["conditions_model"], 
                            brainst_vol["noise_scheduler"], 
                            covars_list_dict=[covariates_dict], 
                            covars_keys_ordered=covars_keys_ordered,
                            free_guidance_ratio=1.0, 
                            return_noisy_steps=False)
    
    # convert prediction to dict
    prediction = prediction[0]
    target_roi_volumes_dict = {key: float(prediction[i]) for i, key in enumerate(conditions_keys_ordered)}
    
    return target_roi_volumes_dict
    

def brainst_synthesis(covariates_dict, brainst_img_config_path, brainst_vol_config_path, output_path, seed=2):
    # to solve
    autoencoder_chk_path = "/home/agustin/phd/synthesis/final_code/models/brainst_img/autoencoder/weights/autoencoder_epoch273.pt"
    
    # 1. generate random volumes using the brainst_vol model
    # ---- 1.1 instantiate model
    brainst_vol_params = load_parameters(brainst_vol_config_path)
    brainst_vol = instantiate_brainst_vol.instantiate_model_and_load(brainst_vol_params, device=device, dm_num_inference_steps=50) 
    
    # ---- 1. 2 generate random volumes
    target_roi_volumes_dict = brainst_vol_synthesis(brainst_vol, covariates_dict, seed=seed)
    
    # 2. generate the image using the brainst_img model
    # ---- 2.1 instantiate model
    brainst_img_params = load_parameters(brainst_img_config_path)
    brainst_img = instantiate_brainst_img.instantiate_model_and_load(brainst_img_params, autoencoder_chk_path, device=device, dm_num_inference_steps=50) 
    
    # ---- 2.2 generate image
    recon_img = brainst_img_synthesis(brainst_img, target_roi_volumes_dict, seed=seed)
    
    # 3 save the reconstructed image
    output_path_name = os.path.join(output_path, "synthetic_image.nii.gz")
    nfc.save_nifti(recon_img, output_path_name)

    
    
    





def brainst_img_transformation(brainst_img, img, initial_roi_volumes_dict, target_roi_volumes_dict):
    conditions_keys_ordered = brainst_img["model_description"].args_train.conditions_keys_ordered
    
    # 1 generate random noise
    latents = brainst_img["autoencoder"].encode(img).cpu().numpy()
    inverted_data = null_inversion.invert_latents(
            brainst_img["unet"],
            brainst_img["conditions_model"],
            brainst_img["noise_scheduler"],
            latents,
            initial_roi_volumes_dict,
            conditions_keys_ordered,
            free_guidance_ratio=2.0,
            compute_uncond_embeddings=True,
            num_inner_steps=2,
            early_stop_epsilon=1e-8,
            verbose=False,
        )
                        
                        
    # 3 create image
    reconstrcuted_latents = generate_image.diffusion_loop(
            inverted_data["noisy_latents"], # last noisiest latents
            brainst_img["unet"],
            brainst_img["conditions_model"],
            brainst_img["noise_scheduler"],
            brainst_img["autoencoder"],
            [target_roi_volumes_dict],
            conditions_keys_ordered,
            uncond_embeddings=inverted_data["uncond_embeddings"],
            free_guidance_ratio=2.0,
            decode_img=True,
            decode_first=True,
            decode_complete=True,
        )

    # 4 save the reconstructed image
    recon_img = reconstrcuted_latents["images"][0].astype(np.float32)
    return recon_img







def brainst_vol_transformation(brainst_vol, initial_roi_volumes_dict, covars_from, covars_to, fgr=1.0, compute_uncond_embeddings=False):
    covars_keys_ordered = brainst_vol["model_description"].args_train.covars_list
    conditions_keys_ordered = brainst_vol["model_description"].args_train.conditions_keys_ordered
    
    
    input_vec = [initial_roi_volumes_dict[key] for key in conditions_keys_ordered]
    input_vec = np.expand_dims(input_vec, axis=0)
    
    
    inversion = null_inversion_volumes.invert_latents(brainst_vol["unet"], 
                    brainst_vol["conditions_model"], 
                    brainst_vol["noise_scheduler"],
                    input_vec=input_vec,
                    covars_list_dict=[covars_from],
                    covars_keys_ordered=covars_keys_ordered,
                    free_guidance_ratio=fgr,
                    num_inner_steps=4,
                    early_stop_epsilon=1e-10,
                    compute_uncond_embeddings=compute_uncond_embeddings
                    )

    prediction = generate_volumes.diffusion_loop(
                            inversion["noisy_latents"], 
                            brainst_vol["unet"],
                            brainst_vol["conditions_model"], 
                            brainst_vol["noise_scheduler"], 
                            covars_list_dict=[covars_to], 
                            covars_keys_ordered=covars_keys_ordered,
                            uncond_embeddings=inversion["uncond_embeddings"], 
                            free_guidance_ratio=fgr,
                            return_noisy_steps=False)
    
    # convert prediction to dict
    prediction = prediction[0]
    prediction_dict = {key: float(prediction[i]) for i, key in enumerate(conditions_keys_ordered)}
    return prediction_dict



# def brainst_transformation(image_path, target_roi_volumes_dict, brainst_img_config_path, brainst_vol_config_path, output_path, segmentation_path=None, seed=2):
def brainst_transformation(image_path, covars_from, covars_to, brainst_img_config_path, brainst_vol_config_path, output_path, segmentation_path=None, seed=2):
    # to solve
    autoencoder_chk_path = "/home/agustin/phd/synthesis/final_code/models/brainst_img/autoencoder/weights/autoencoder_epoch273.pt"
    normalizer_params = "/home/agustin/phd/synthesis/final_code/src/preprocessing/normalizer_params.json"
    normalizer = data_normalization.SavedNormalizerBrainStructures(normalizer_params)
    dx_mapping = {"CN":0, "MCI":1, "AD":2}
    sex_mapping = {"F":0, "M":1}
    
    # ---- 2.1 instantiate model
    brainst_img_params = load_parameters(brainst_img_config_path)
    brainst_img = instantiate_brainst_img.instantiate_model_and_load(brainst_img_params, autoencoder_chk_path, device=device, dm_num_inference_steps=20) 
    
    
    
    # 1. obtain current image volumes
    img, affine = nfc.load_nifti(image_path)
    seg, _ = nfc.load_nifti(segmentation_path)
    
    # 1.1 preprocess image
    org_shape = img.shape
    img, _ = prep_images.preprocess_image(img, affine)
    
    # 1.2 preprocess volumes
    # 1.2.1 obtain volumes
    seg_structure_names_list = brainst_img["model_description"].args_train.conditions_keys_ordered
    roi_volumes_dict = prep_volumes.get_volumes(seg, seg_structure_names_list)
    
    # 1.2.2 normalize volumes
    
    roi_volumes_df_norm_icv = data_normalization.normalize_by_icv(pd.DataFrame([roi_volumes_dict]), structure_names=seg_structure_names_list, icv_column="total_vol", percentage=False)
    roi_volumes_dict_norm = normalizer.transform(roi_volumes_df_norm_icv).iloc[0].to_dict()
    
    
    
    # --------- NEW
    # 1. transform the volumes using the brainst_vol model
    # ---- 1.1 instantiate model
    brainst_vol_params = load_parameters(brainst_vol_config_path)
    brainst_vol = instantiate_brainst_vol.instantiate_model_and_load(brainst_vol_params, device=device, dm_num_inference_steps=50) 
    
    covars_from["age"] = normalizer.transform_single(covars_from["age"], "age")
    covars_from["sex"] = sex_mapping[covars_from["sex"]]
    covars_from["dx"] = dx_mapping[covars_from["dx"]]
    
    
    covars_to["age"] = normalizer.transform_single(covars_to["age"], "age")
    covars_to["sex"] = sex_mapping[covars_to["sex"]]
    covars_to["dx"] = dx_mapping[covars_to["dx"]]      
                
    target_roi_volumes_dict = brainst_vol_transformation(brainst_vol, roi_volumes_dict_norm, covars_from, covars_to)
    # --------- END NEW
    
    
    # 2. transform the image using the brainst_img model

    # ---- 2.2 generate image
    recon_img = brainst_img_transformation(brainst_img, img, roi_volumes_dict_norm, target_roi_volumes_dict)
    
    # ---- 2.2 postprocess image (resize to original shape)
    recon_img = prep_images.postprocess_image(recon_img, org_shape)
    
    # 3 save the reconstructed image
    output_path_name = os.path.join(output_path, "transformed_img.nii.gz")
    nfc.save_nifti(recon_img, output_path_name, affine=affine)


# tienes que hacer algo con la dualidad de la syntheis o transformaciones



#  if __name__ == "__main__":
#      # obtain args
     
     
#      # generation_type = "synthesis" # ["synthesis", "transformation", "longitudinal"]
#      # if generation_type == "synthesis":
#         # target_roi_volumes_dict [satndarized, percentage, mm, segmentation]
#         # or segmentation_path # 
#         # or target_covariates # [age, sex, dx] = [55, "M", "CN"]
        
#     # if generation_type == "transformation":
#         # image_path
#         # segmentation_path or synthseg installed
#         # target_roi_volumes_dict [satndarized, percentage, mm, segmentation]
#         # or segmentation_path # 
#         # or inital_covariates and target_covariates # [age, sex, dx] = [55, "M", "CN"]
        
#     # if generation_type == "longitudinal":
#         # image_path
#         # segmentation_path # or synthseg installed
#         # inital_covariates and target_age, target_dx
        
        
        
    

# roi_volumes_dict = {
#         "total_vol": 0.0,
#         "surrounding_csf_vol": 0.0,
#         "cortical_gm_vol": 0.0,
#         "cerebral_wm_vol": 0.0,
#         "lateral_ventricles_vol": 0.0,
#         "third_ventricle_vol": 0.0,
#         "fourth_ventricle_vol": 0.0,
#         "thalamus_vol": 0.0,
#         "hippocampus_vol": 0.0,
#         "amygdala_vol": 0.0,
#         "putamen_vol": 0.0,
#         "pallidum_vol": 0.0,
#         "caudate_vol": 0.0,
#         "accumbens_area_vol": 0.0,
#         "ventral_dc_vol": 0.0,
#         "cerebellum_gm_vol": 0.0,
#         "cerebellum_wm_vol": 0.0,
#         "brainstem_vol": 0.0
#     }

# # covariates_dict = { # age shoudl be stadarized, sex should be 0 or 1, dx should be 0 1 or 2
# #     "age": 5.0,
# #     "sex": 1,
# #     "dx": 0
# # }


# covariates_from_dict = { # age shoudl be stadarized, sex should be 0 or 1, dx should be 0 1 or 2
#     "age": 55,
#     "sex": "M",
#     "dx": "CN"
# }

# covariates_to_dict = { # age shoudl be stadarized, sex should be 0 or 1, dx should be 0 1 or 2
#     "age": 90,
#     "sex": "M",
#     "dx": "AD"
# }

# brainst_img_config_path = "/home/agustin/phd/synthesis/final_code/models/brainst_img/cond18_masked/model_config.json"   
# brainst_vol_config_path = "/home/agustin/phd/synthesis/final_code/models/brainst_vol/test1_covars_add_ema4/model_config.json"   
# output_path = "/home/agustin/phd/synthesis/final_code/outputs"
# # brainst_img_synthesis(roi_volumes_dict, 
# #                       brainst_img_config_path, 
# #                       None, 
# #                       output_path)


# # brainst_synthesis(covariates_dict, 
# #                       brainst_img_config_path, 
# #                       brainst_vol_config_path, 
# #                       output_path)


# image_path="/home/agustin/phd/synthesis/data/MNI_template/usrl/mni_icbm152_t1synthsr_tal_nlin_sym_09a.nii.gz"
# segmentation_path="/home/agustin/phd/synthesis/data/MNI_template/usrl/mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz"
# # brainst_transformation(image_path,
# #                       target_roi_volumes_dict=roi_volumes_dict,
# #                       brainst_img_config_path=brainst_img_config_path,
# #                       brainst_vol_config_path=brainst_vol_config_path,
# #                       output_path=output_path,
# #                         segmentation_path=segmentation_path)


# brainst_transformation(image_path,
#                       covars_from=covariates_from_dict,
#                       covars_to=covariates_to_dict,
#                       brainst_img_config_path=brainst_img_config_path,
#                       brainst_vol_config_path=brainst_vol_config_path,
#                       output_path=output_path,
#                         segmentation_path=segmentation_path)