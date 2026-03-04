import nibabel as nib
import numpy as np

AFFINE = np.array([
    [1.,  0.,  0.,  -98.],
    [ 0., 1.,  0., -134.],
    [ 0.,  0., 1.,  -72.],
    [ 0.,  0.,  0.,   1.]  
])



# ------------ LOAD IMAGES
def load_nifti(path_name, transpose=False):
    imag_nifti = nib.load(path_name)
    img_data = imag_nifti.get_fdata()
    if transpose:
        img_data = np.transpose(img_data, (1, 0, 2))
    return img_data, (imag_nifti.affine, imag_nifti.header)

# ------------ SAVE IMAGES
def save_nifti(image_np, img_path_name, affine=None):
    if affine is None:
        img_nifti = nib.Nifti1Image(image_np, affine=AFFINE)
    elif isinstance(affine, tuple):
        img_nifti = nib.Nifti1Image(image_np, affine=affine[0], header=affine[1])
    nib.save(img_nifti, img_path_name)