

import os
import numpy as np
import pandas as pd
import json

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


def get_volumes_from_segmentation(segmentation, structure_names_list, normalizer):
    roi_volumes_dict = prep_volumes.get_volumes(segmentation, structure_names_list)
    roi_volumes_dict = data_normalization.normalize_by_icv(pd.DataFrame([roi_volumes_dict]), structure_names=structure_names_list, icv_column="total_vol", percentage=False).iloc[0].to_dict()
    roi_volumes_dict = normalizer.transform(pd.DataFrame([roi_volumes_dict])).iloc[0].to_dict()

    return roi_volumes_dict

# =====================================================
# SYNTHESIS
# =====================================================
    
def brainst_vol_synthesis(brainst_vol, target_covariates_dict, seed=2):
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
                            covars_list_dict=[target_covariates_dict], 
                            covars_keys_ordered=covars_keys_ordered,
                            free_guidance_ratio=1.0, 
                            return_noisy_steps=False)
    
    # convert prediction to dict
    prediction = prediction[0]
    target_roi_volumes_dict = {key: float(prediction[i]) for i, key in enumerate(conditions_keys_ordered)}
    
    return target_roi_volumes_dict
    


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


def brainst_synthesis(brainst_img_config_path, autoencoder_chk_path, output_path_name, target_roi_volumes_dict=None, brainst_vol_config_path=None, target_covariates_dict=None, seed=2, diffusion_steps=50):
    # 1. generate random volumes using the brainst_vol model
    # ---- 1. 2 generate random volumes
    if target_roi_volumes_dict is None:
        if target_covariates_dict is None or brainst_vol_config_path is None:
            raise ValueError("If target_roi_volumes_dict is not provided, target_covariates_dict and brainst_vol_config_path must be provided")
            
        # ---- 1.1 instantiate model
        brainst_vol_params = load_parameters(brainst_vol_config_path)
        brainst_vol = instantiate_brainst_vol.instantiate_model_and_load(brainst_vol_params, device=device, dm_num_inference_steps=diffusion_steps) 
        
        target_roi_volumes_dict = brainst_vol_synthesis(brainst_vol, target_covariates_dict, seed=seed)
        
    # 2. generate the image using the brainst_img model
    # ---- 2.1 instantiate model
    brainst_img_params = load_parameters(brainst_img_config_path)
    brainst_img = instantiate_brainst_img.instantiate_model_and_load(brainst_img_params, autoencoder_chk_path, device=device, dm_num_inference_steps=diffusion_steps) 

    # ---- 2.2 generate image
    recon_img = brainst_img_synthesis(brainst_img, target_roi_volumes_dict, seed=seed)
    
    # 3 save the reconstructed image
    nfc.save_nifti(recon_img, output_path_name)

    
    
    
# =====================================================
# TRANSFORMATION
# =====================================================

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







def brainst_vol_transformation(brainst_vol, initial_roi_volumes_dict, initial_covariates_dict, target_covariates_dict, fgr=1.0, compute_uncond_embeddings=False):
    covars_keys_ordered = brainst_vol["model_description"].args_train.covars_list
    conditions_keys_ordered = brainst_vol["model_description"].args_train.conditions_keys_ordered
    
    
    input_vec = [initial_roi_volumes_dict[key] for key in conditions_keys_ordered]
    input_vec = np.expand_dims(input_vec, axis=0)
    
    
    inversion = null_inversion_volumes.invert_latents(brainst_vol["unet"], 
                    brainst_vol["conditions_model"], 
                    brainst_vol["noise_scheduler"],
                    input_vec=input_vec,
                    covars_list_dict=[initial_covariates_dict],
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
                            covars_list_dict=[target_covariates_dict], 
                            covars_keys_ordered=covars_keys_ordered,
                            uncond_embeddings=inversion["uncond_embeddings"], 
                            free_guidance_ratio=fgr,
                            return_noisy_steps=False)
    
    # convert prediction to dict
    prediction = prediction[0]
    prediction_dict = {key: float(prediction[i]) for i, key in enumerate(conditions_keys_ordered)}
    return prediction_dict




def brainst_transformation(image_path, segmentation_path, brainst_img_config_path, autoencoder_chk_path, normalizer, output_path_name, target_roi_volumes_dict=None, brainst_vol_config_path=None, initial_covariates_dict=None, target_covariates_dict=None, diffusion_steps=50):
    # covariates should be normalized and in the correct format (e.g. age should be standardized, sex should be 0 or 1, dx should be 0, 1 or 2)

    # ---- 2.1 instantiate model
    brainst_img_params = load_parameters(brainst_img_config_path)
    brainst_img = instantiate_brainst_img.instantiate_model_and_load(brainst_img_params, autoencoder_chk_path, device=device, dm_num_inference_steps=diffusion_steps) 
    
    # 1. obtain current image volumes
    img, affine = nfc.load_nifti(image_path)
    seg, _ = nfc.load_nifti(segmentation_path)
    
    # 1.1 preprocess image
    org_shape = img.shape
    img, _ = prep_images.preprocess_image(img, affine)
    
    # 1.2 preprocess volumes
    initial_roi_volumes_dict = get_volumes_from_segmentation(seg, brainst_img["model_description"].args_train.conditions_keys_ordered, normalizer)
    print("------------ Initial ROI volumes dict:")
    print(initial_roi_volumes_dict)

    # 2. transform the covariates
    if target_roi_volumes_dict is None:
        if initial_covariates_dict is None or target_covariates_dict is None or brainst_vol_config_path is None:
            raise ValueError("If target_roi_volumes_dict is not provided, initial_covariates_dict, target_covariates_dict and brainst_vol_config_path must be provided")
        # ---- 1.1 instantiate model
        brainst_vol_params = load_parameters(brainst_vol_config_path)
        brainst_vol = instantiate_brainst_vol.instantiate_model_and_load(brainst_vol_params, device=device, dm_num_inference_steps=diffusion_steps) 
                    
        target_roi_volumes_dict = brainst_vol_transformation(brainst_vol, initial_roi_volumes_dict, initial_covariates_dict, target_covariates_dict)
    
        print("------------ Target ROI volumes dict:")
        print(target_roi_volumes_dict)
    # 3. generate image
    # ---- 3.1 transform image
    recon_img = brainst_img_transformation(brainst_img, img, initial_roi_volumes_dict, target_roi_volumes_dict)
    
    # ---- 3.2 postprocess image (resize to original shape)
    recon_img = prep_images.postprocess_image(recon_img, org_shape)
    
    # 4. save the transformed image
    nfc.save_nifti(recon_img, output_path_name, affine=affine)
