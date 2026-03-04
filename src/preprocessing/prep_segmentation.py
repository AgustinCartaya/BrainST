
import sys
import os
import subprocess

import sys
sys.path.append('/home/agustin/phd/synthesis')
import utils.nifti_functions as nfc

# normal = 31seg
# not robust = 28seg 
# no robust no cortical_parcelation = 23seg

def save_synthseg_segmentation(img_path_name, out_path_name, verify=False, verbose=False, robust=True, cortical_parcelation=True, path_name_vol_csv=None, path_name_resampled_img=None, path_name_post_prob_img=None):
    """Run mri_synthseg to get the segmentation of the input image.
    Args:
        img_path_name (str): Path to the input image.
        out_path_name (str): Path to save the output segmentation.
        verify (bool): If True, check if the output file already exists.
        verbose (bool): If True, print additional information.
        robust (bool): If True, use the robust option in mri_synthseg.
        cortical_parcelation (bool): If True, use cortical parcelation in mri_synthseg.
        path_name_vol_csv (str): Path to save the volume CSV file.
        path_name_resampled_img (str): Path to save the resampled image.
        path_name_post_prob_img (str): Path to save the posteriors probability image.

    Returns:
        None: The function saves the segmentation to the specified output path.
    Example:
        prep_segmentation.save_synthseg_segmentation(
                                img_path_name=path_name_raw_t1w,
                                out_path_name=path_name_seg,
                                verify=True,
                                verbose=True,
                                robust=True,
                                cortical_parcelation=True,
                                path_name_vol_csv=path_neme_vol_csv,
                                path_name_resampled_img=path_neme_resampled_img,
                                path_name_post_prob_img=path_neme_post_prob_img,
                            )
    """
    
    # mri_synthseg --i <input> --o <output> --parc --robust --threads <threads> --cpu
    if verify:
        exist = os.path.exists(out_path_name)
        if exist:
            print(f"synthseg already done for: {img_path_name}\nfile in: {out_path_name}")
            return True
 
    command = [ "mri_synthseg", 
                "--i", img_path_name, 
                "--o", out_path_name, 
                # "--parc",
                # "--robust",
                "--cpu",
                "--threads", "84"
                ]
    
    if robust:
        command.append("--robust")
    if cortical_parcelation:
        command.append("--parc")
    if path_name_vol_csv is not None:
        command.extend(["--vol", path_name_vol_csv])
    if path_name_resampled_img is not None:
        command.extend(["--resample", path_name_resampled_img])
    if path_name_post_prob_img is not None:
        command.extend(["--post", path_name_post_prob_img])

    # print(command)
    
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        if verbose:
            print(f"synthseg done saved in: {out_path_name}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)


def get_synthseg_segmentation(img, affine, temp_out_dir, verbose=False):
    # mri_synthseg --i <input> --o <output> --parc --robust --threads <threads> --cpu

    os.makedirs(temp_out_dir, exist_ok=True)
    __img_path_name = os.path.join(temp_out_dir, "img.nii.gz")
    __seg_path_name = os.path.join(temp_out_dir, "seg.nii.gz")
    nfc.save_nifti(img, affine, __img_path_name)

    save_synthseg_segmentation(__img_path_name, __seg_path_name, verify=False, verbose=verbose)

    # Load the segmentation
    seg, _ = nfc.load_nifti(__seg_path_name)
    return seg