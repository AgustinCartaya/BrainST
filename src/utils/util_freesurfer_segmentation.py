"""FreeSurfer/SynthSeg anatomical label definitions and segmentation helpers.

This module has two parts:

1. **Label constants** (module-level lists like ``CEREBELLUM_GM``,
   ``THALAMUS_V0``, ``CEREBRAL_CORTEX_LEFT_96``, ...): each is a list of
   integer label values from the FreeSurfer/SynthSeg segmentation
   protocol identifying one anatomical structure (often as a
   ``[left, right]`` pair). These are combined into coarser groupings
   (``GM_96``, ``WM``, ``CSF``, ``TOTAL_96``, ...) and finally into
   ``structures_index_dict``, the canonical name -> label-list mapping
   used elsewhere in the project (see also
   ``configs.STRUCTURE_INDEX_DICT``, which mirrors a subset of this).

   These numeric IDs are exact anatomical protocol values and are never
   modified by this refactor.

2. **Segmentation helper functions**: mask extraction, label merging
   (96-class -> 36-class -> 3-class), visualization utilities (RGB
   overlays), GMM statistics for intensity modeling, and SynthSeg CSV
   column reconciliation (structure name <-> FreeSurfer color LUT).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

current_path = os.path.dirname(__file__)


BG = [0]

# CEREBELLUM AND BRAINSTEM
CEREBELLUM_GM = [8, 47] 
CEREBELLUM_WM = [7, 46]
CEREBELLUM = CEREBELLUM_GM + CEREBELLUM_WM

BRAINSTEM = [16]
CEREBELLUM_BRAINSTEM = CEREBELLUM + BRAINSTEM

# CEREBRUM
# ---- GM
# there is a small proble with (9, 48) and (10,49) they are repeated (thalamus = thalamus prpoer*)
THALAMUS_V0 = [9, 48]
THALAMUS_V1 = [10, 49]
HIPPOCAMPUS = [17, 53]
CAUDATE = [11, 50] 
PUTAMEN = [12, 51]
PALLIDUM = [13, 52]
AMYGDALA = [18, 54]
ACCUMBENS_AREA = [26, 58]
VENTRAL_DC = [28,60]
CEREBRAL_SUB_CORTICAL_GM_LEFT = [THALAMUS_V0[0], THALAMUS_V1[0], CAUDATE[0], PUTAMEN[0], PALLIDUM[0], HIPPOCAMPUS[0], AMYGDALA[0], ACCUMBENS_AREA[0], VENTRAL_DC[0]] 
CEREBRAL_SUB_CORTICAL_GM_RIGHT = [THALAMUS_V0[1], THALAMUS_V1[1], CAUDATE[1], PUTAMEN[1], PALLIDUM[1], HIPPOCAMPUS[1], AMYGDALA[1], ACCUMBENS_AREA[1], VENTRAL_DC[1]] 

CEREBRAL_SUB_CORTICAL_GM = CEREBRAL_SUB_CORTICAL_GM_LEFT + CEREBRAL_SUB_CORTICAL_GM_RIGHT

# extra subcortical structures
HIPOTHALAMUS_DIENCEPHALIC_NUCLEI = [819, 820]
HIPOTHALAMUS_NUCLEI = [843, 844]
LIMBIC_NUCLEI = [865, 866]
SEPTAL_NUCLEI = [869, 870]
EXTRA_CEREBRAL_SUB_CORTICAL_GM = HIPOTHALAMUS_DIENCEPHALIC_NUCLEI + HIPOTHALAMUS_NUCLEI + LIMBIC_NUCLEI + SEPTAL_NUCLEI
EXTRA_CEREBRAL_SUB_CORTICAL_GM_LEFT = [HIPOTHALAMUS_DIENCEPHALIC_NUCLEI[0], HIPOTHALAMUS_NUCLEI[0], LIMBIC_NUCLEI[0], SEPTAL_NUCLEI[0]]
EXTRA_CEREBRAL_SUB_CORTICAL_GM_RIGHT = [HIPOTHALAMUS_DIENCEPHALIC_NUCLEI[1], HIPOTHALAMUS_NUCLEI[1], LIMBIC_NUCLEI[1], SEPTAL_NUCLEI[1]]

# -------- FreeSurfer labels 96 tissues

CEREBRAL_CORTEX_LEFT_96 = [1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035]
CEREBRAL_CORTEX_RIGHT_96 = [2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
CEREBRAL_CORTEX_96 = CEREBRAL_CORTEX_LEFT_96 + CEREBRAL_CORTEX_RIGHT_96

CEREBRAL_GM_96 = CEREBRAL_SUB_CORTICAL_GM + CEREBRAL_CORTEX_96


# -------- FreeSurfer labels 36 tissues
CEREBRAL_CORTEX_LEFT_36 = [3]
CEREBRAL_CORTEX_RIGHT_36 = [42]
CEREBRAL_CORTEX_36 = CEREBRAL_CORTEX_LEFT_36 + CEREBRAL_CORTEX_RIGHT_36

CEREBRAL_GM_36 = CEREBRAL_SUB_CORTICAL_GM + CEREBRAL_CORTEX_36


# ---- WM
CEREBRAL_WM_LEFT = [2]
CEREBRAL_WM_RIGHT= [41]
CEREBRAL_WM = CEREBRAL_WM_LEFT + CEREBRAL_WM_RIGHT
CEREBRAL_WM_HYPO = [77]
CEREBRAL_WM = CEREBRAL_WM + CEREBRAL_WM_HYPO

# extra cerebral white matter
FORNIX = [821, 822]
MID_ANTERIOR_COMMISSURE = [853]
OPTIC_CHIASM = [85]

EXTRA_CEREBRAL_WM = FORNIX + MID_ANTERIOR_COMMISSURE + OPTIC_CHIASM
EXTRA_CEREBRAL_WM_LEFT = [FORNIX[0], MID_ANTERIOR_COMMISSURE[0], OPTIC_CHIASM[0]]
EXTRA_CEREBRAL_WM_RIGHT = [FORNIX[1]]
# CSF
LATERAL_VENTRICLES_LEFT = [4]
LATERAL_VENTRICLES_RIGHT = [43]
LATERAL_VENTRICLES = LATERAL_VENTRICLES_LEFT + LATERAL_VENTRICLES_RIGHT

INFERIOR_LATERAL_VENTRICLES = [5, 44] # they are part of the lateral ventricles but are small and entered to the temporal lobe
THIRD_VENTRICLE = [14]
FOURTH_VENTRICLE = [15]
# CHOROID_PLEXUS = [31, 63] # not in synthseg

CEREBRAL_VENTRICLES = LATERAL_VENTRICLES + INFERIOR_LATERAL_VENTRICLES + THIRD_VENTRICLE
NO_CEREBRAL_VENTRICLES = FOURTH_VENTRICLE #+ CHOROID_PLEXUS
INTERNAL_CSF = CEREBRAL_VENTRICLES + NO_CEREBRAL_VENTRICLES

SURROUNDING_CSF = [24]
CSF = INTERNAL_CSF + SURROUNDING_CSF


# COMBINED
PARECHIMA_36 = CEREBRAL_GM_36 + CEREBRAL_WM
PARECHIMA_96 = CEREBRAL_GM_96 + CEREBRAL_WM

GM_36 = CEREBRAL_GM_36 + CEREBELLUM_GM + BRAINSTEM
GM_96 = CEREBRAL_GM_96 + CEREBELLUM_GM + BRAINSTEM

WM = CEREBRAL_WM + CEREBELLUM_WM

# TOTAL
TOTAL_96 = GM_96 + WM + CSF
TOTAL_36 = GM_36 + WM + CSF

TOTAL_96_NO_SURROUNDING_CSF = GM_96 + WM + INTERNAL_CSF
TOTAL_36_NO_SURROUNDING_CSF = GM_36 + WM + INTERNAL_CSF

WM += EXTRA_CEREBRAL_WM
GM_96 += EXTRA_CEREBRAL_SUB_CORTICAL_GM
GM_36 += EXTRA_CEREBRAL_SUB_CORTICAL_GM


def read_freesurfer_color_table(names_lower_case: bool = True) -> pd.DataFrame:
    """Load FreeSurfer's ``FreeSurferColorLUT.txt`` (label id <-> name <-> RGBA) table.

    Args:
        names_lower_case: If True, lowercase the ``NAME`` column.

    Returns:
        DataFrame with columns ``["ID", "NAME", "R", "G", "B", "A"]``
        (all as strings, matching the raw LUT file format).
    """
    file_path = f'{current_path}/FreeSurferColorLUT.txt'
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):  # Ignorar las líneas que empiezan con #
                continue
            line = line.strip()  # Eliminar espacios en blanco al principio y al final
            if line:  # Ignorar las líneas vacías
                parts = line.split()  # Dividir la línea en partes por los espacios en blanco
                # Ignorar las líneas que no tienen el número de columnas esperado
                if len(parts) == 6:
                    data.append(parts)  # Ignorar el número de línea y agregar el resto de los datos

    df = pd.DataFrame(data, columns=['ID', 'NAME', 'R', 'G', 'B', 'A'])
    if names_lower_case:
        df.iloc[:, 1] = df.iloc[:, 1].str.lower()
    return df


def segment_freesurfer_labels(img: np.ndarray, seg: np.ndarray, labels: Sequence[int | str]) -> np.ndarray:
    """Extract intensity values belonging to a set of FreeSurfer labels, zeroing everything else.

    Args:
        img: Intensity image, same shape as ``seg``.
        seg: Label map.
        labels: Sequence of label values (as ``int``) or label names (as
            ``str``, resolved via :func:`read_freesurfer_color_table`).

    Returns:
        An array the same shape as ``img`` with only the requested
        labels' voxels retained (all else zeroed).

    Raises:
        ValueError: If a string label is not found in the color table.
    """
    img_masked = np.zeros_like(img)
    color_table = read_freesurfer_color_table()
    for label in labels:
        if isinstance(label, str):
            label = color_table[color_table['NAME'] == str(label)].ID
            label = int(label.iloc[0])            
        if isinstance(label, int):
            img_masked[seg == label] = img[seg == label]
        else:
            raise ValueError(f'Label {label} is not valid')
    return img_masked


def seg_to_rgbseg(seg: np.ndarray, colors_dict: dict | None = None) -> np.ndarray:
    """Convert an integer label map to an RGB color image using the FreeSurfer LUT.

    Args:
        seg: Integer label map.
        colors_dict: Optional override mapping ``{label: (r, g, b)}`` for
            specific labels, taking priority over the FreeSurfer LUT.

    Returns:
        ``uint8`` array of shape ``seg.shape + (3,)``.

    Raises:
        ValueError: If a label present in ``seg`` is not found in the
            FreeSurfer color table (and not present in ``colors_dict``).
    """
    color_table = read_freesurfer_color_table()
    rgb_seg = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label in np.unique(seg):
        if colors_dict is not None and label in colors_dict:
            color = colors_dict[label]
        else:
            color = color_table[color_table['ID'] == str(int(label))][['R', 'G', 'B']].values
        if len(color) == 0:
            raise ValueError(f'Label {label} not found in the color table')
        rgb_seg[seg == label] = color
    return rgb_seg


def apply_seg_transparency(img: np.ndarray, seg: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend an RGB segmentation overlay on top of a (normalized) grayscale image.

    Args:
        img: Grayscale intensity image (normalized internally by its max value).
        seg: Either an already-RGB segmentation (4D array) or an integer
            label map (converted via :func:`seg_to_rgbseg`).
        alpha: Blend weight of the segmentation overlay (0 = image only, 1 = seg only).

    Returns:
        Blended RGB image (float, not clamped to ``uint8``).
    """
    if len(seg.shape) != 4:
        seg = seg_to_rgbseg(seg)

    img = (255 * (img/np.max(img))).astype(np.uint8)

    img_masked = seg * alpha
    img_masked = img_masked + (1 - alpha) * img
    return img_masked

def get_tissue_mask(seg: np.ndarray, tissues: Sequence[int]) -> np.ndarray:
    """Return a boolean mask selecting voxels whose label is in ``tissues``."""
    tissue_mask = np.isin(seg.astype(int), tissues)
    return tissue_mask

def get_cerebellum_brainstem_4vent_mask(seg: np.ndarray) -> np.ndarray:
    """Return a binary mask of cerebellum + brainstem + fourth ventricle."""
    tissues= CEREBELLUM + BRAINSTEM + FOURTH_VENTRICLE
    cerebellum_mask = merge_seg96_to_mask(seg, tissues)
    return cerebellum_mask


def remove_cerebellum_brainstem_4vent(seg: np.ndarray, seg_36: np.ndarray, deep_copy: bool = True, return_cerebellum_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Zero out cerebellum/brainstem/4th-ventricle voxels in a segmentation.

    Args:
        seg: Segmentation to modify (any label scheme; only masked by
            position using ``seg_36``).
        seg_36: 36-class segmentation used to compute the removal mask
            (see :func:`get_cerebellum_brainstem_4vent_mask`).
        deep_copy: If True, operate on a copy of ``seg`` (leaving the
            input untouched); if False, modify ``seg`` in place.
        return_cerebellum_mask: If True, also return the mask used.

    Returns:
        The masked segmentation, or ``(masked_seg, cerebellum_mask)`` if
        ``return_cerebellum_mask=True``.
    """
    cerebellum_mask = get_cerebellum_brainstem_4vent_mask(seg_36)
    if deep_copy:
        seg_masked = seg.copy()
    else:
        seg_masked = seg
        
    seg_masked[cerebellum_mask == 1] = 0

    if return_cerebellum_mask:
        return seg_masked, cerebellum_mask
    return seg_masked



def get_4_tissues_prob_mask(brain_mask: np.ndarray, tissues_3_seg: np.ndarray, weights: dict[str, float] = {"bg":0, "ext":1, "csf":1, "gm":1, "wm":1}, dtype=np.float32) -> np.ndarray:
    """Build a weighted 4-tissue (bg/external/csf/gm/wm) probability-style mask.

    Args:
        brain_mask: Binary brain (vs. background) mask.
        tissues_3_seg: 3-class tissue segmentation (0=bg/csf-like, up to
            3=wm-like, matching the ``brain_mask + tissues_3_seg`` sum
            categories used below).
        weights: Per-category scalar weight applied to that category's
            mask before summing.
        dtype: Output dtype.

    Returns:
        A single-channel array combining the 5 weighted one-hot channels
        (bg, ext, csf, gm, wm) via summation.
    """
    complete_mask = brain_mask + tissues_3_seg
    multi_channel_mask = np.stack([np.where(complete_mask==0, 1, 0), # bg
                                    np.where(complete_mask==1, 1, 0), # ext
                                    np.where(complete_mask==2, 1, 0), # csf
                                    np.where(complete_mask==3, 1, 0), # gm
                                    np.where(complete_mask==4, 1, 0)], # wm
                                    axis=-1).transpose(3, 0, 1, 2).astype(dtype)
    multi_channel_mask[0][multi_channel_mask[0]>0]=weights["bg"]
    multi_channel_mask[1][multi_channel_mask[1]>0]=weights["ext"]
    multi_channel_mask[2][multi_channel_mask[2]>0]=weights["csf"]
    multi_channel_mask[3][multi_channel_mask[3]>0]=weights["gm"]
    multi_channel_mask[4][multi_channel_mask[4]>0]=weights["wm"]
    return np.sum(multi_channel_mask, axis=0)



# for dice score using gaussian mixture model
def obtain_gmm_elements_using_seg(img: np.ndarray, seg: np.ndarray, remove_bg: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-tissue intensity Gaussian-mixture statistics from a segmentation.

    Args:
        img: Intensity image.
        seg: Label map, same shape as ``img``.
        remove_bg: If True, exclude the background label (0) from the
            returned statistics.

    Returns:
        ``(means, stds, class_weights, class_imbalance)``, one entry per
        unique label (excluding background if ``remove_bg``):
            - ``means``/``stds``: per-tissue intensity mean/std.
            - ``class_weights``: voxel-count fraction per tissue.
            - ``class_imbalance``: inverse of the voxel-count fraction
              relative to the largest class (usable as a loss weight).
    """
    tissue_index, num_elem = np.unique(seg, return_counts=True)
    if remove_bg and np.any(tissue_index == 0):
        first_index = np.argmax(tissue_index == 0)
        tissue_index = np.delete(tissue_index, first_index)
        num_elem = np.delete(num_elem, first_index)

    nb_tissues = tissue_index.shape[0]
    classs_weights = num_elem/ np.sum(num_elem)
    means = np.zeros(nb_tissues)
    stds = np.zeros(nb_tissues)

    for i in range(nb_tissues):
        tissue_elemnts = img[seg == tissue_index[i]]
        means[i] = np.mean(tissue_elemnts)
        stds[i] = np.std(tissue_elemnts)

    class_imbalance = num_elem / np.max(num_elem)
    class_imbalance = 1/class_imbalance

    return means, stds, classs_weights, class_imbalance

def obtain_multiple_gmm_elements_using_seg(imgs: Sequence[np.ndarray], segs: Sequence[np.ndarray], dtype=np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch version of :func:`obtain_gmm_elements_using_seg` over multiple (img, seg) pairs.

    Args:
        imgs: Sequence of intensity images.
        segs: Sequence of matching label maps.
        dtype: Output dtype for the stacked arrays.

    Returns:
        ``(means, stds, class_weights, class_imbalance)``, each stacked
        into an array of shape ``(len(imgs), nb_tissues)``.
    """
    means_list = []
    stds_list = []
    classs_weights_list = []
    class_imbalance_list = []

    for img, seg in zip(imgs, segs):
        means, stds, classs_weights, class_imbalance = obtain_gmm_elements_using_seg(img, seg)
        means_list.append(means)
        stds_list.append(stds)
        classs_weights_list.append(classs_weights)
        class_imbalance_list.append(class_imbalance)
    
    return (np.array(means_list, dtype=dtype), 
            np.array(stds_list, dtype=dtype), 
            np.array(classs_weights_list, dtype=dtype), 
            np.array(class_imbalance_list, dtype=dtype))
     

def merge_seg96_to_seg36(seg96: np.ndarray) -> np.ndarray:
    """Collapse the 96-class cortical parcellation into the coarser 36-class scheme.

    All left-hemisphere cortical parcels are merged to a single left
    cortex label (3), and likewise for the right hemisphere (42).

    Args:
        seg96: 96-class segmentation.

    Returns:
        36-class segmentation (a copy; ``seg96`` is not modified).
    """
    seg_36 = seg96.copy()
    for cortex in CEREBRAL_CORTEX_LEFT_96:
        seg_36[seg_36 == cortex] = 3
    for cortex in CEREBRAL_CORTEX_RIGHT_96:
        seg_36[seg_36 == cortex] = 42
    return seg_36

def merge_seg36_to_seg3(seg_36: np.ndarray) -> np.ndarray:
    """Collapse a 36-class segmentation into 4 coarse tissue classes (bg/csf/gm/wm).

    Args:
        seg_36: 36-class segmentation.

    Returns:
        ``uint8`` array with values ``{0: bg, 1: csf, 2: gm, 3: wm}``
        (via one-hot stacking + argmax over ``[BG, CSF, GM_36, WM]``).
    """
    seg_3 = []
    for tissues in [BG, CSF, GM_36, WM]:
        mask = np.isin(seg_36, tissues)
        mask = np.where(mask, 1, 0)
        seg_3.append(mask)

    seg_3 = np.stack(seg_3, axis=-1)
    seg_3 = np.argmax(seg_3, axis=-1).astype(np.uint8)

    return seg_3


def merge_seg96_to_seg3(seg96: np.ndarray) -> np.ndarray:
    """Collapse a 96-class segmentation directly into 4 coarse tissue classes.

    Convenience composition of :func:`merge_seg96_to_seg36` then
    :func:`merge_seg36_to_seg3`.
    """
    return merge_seg36_to_seg3(merge_seg96_to_seg36(seg96))

def merge_seg96_to_mask(seg96: np.ndarray, tissue_list: Sequence[int]) -> np.ndarray:
    """Build a binary mask selecting a set of label values.

    Works identically for both 36-class and 96-class segmentations, since
    only the cortical labels differ between the two schemes (and this
    function is typically called with non-cortical label lists).

    Args:
        seg96: Label map.
        tissue_list: Label values to include in the mask.

    Returns:
        Binary (0/1) mask, same shape as ``seg96``.
    """
    mask = np.isin(seg96, tissue_list)
    mask = np.where(mask, 1, 0)
    return mask


def find_bigger_vent_layer(seg: np.ndarray) -> int:
    """Find the axial layer (last axis index) with the most lateral-ventricle voxels.

    Args:
        seg: Segmentation containing lateral ventricle labels (4, 43).

    Returns:
        The index along the last axis with the largest lateral-ventricle
        voxel count -- useful for picking a representative slice for
        ventricle visualization.
    """
    mask_vent = np.zeros_like(seg)
    mask_vent[(seg == 4) | (seg == 43)] = 1
    bigger_layer = np.argmax(np.sum(mask_vent, axis=(0, 1)))
    return bigger_layer




def combine_synthseg_vols(df_synthseg: pd.DataFrame, structure_names: list[str], index_list: list[Sequence[int]], possible_index: Sequence[int] | None = None) -> pd.DataFrame:
    """
    Sum SynthSeg per-structure volume columns into coarser named groups.

    Args:
        df_synthseg: DataFrame with individual per-structure volumes
            obtained from SynthSeg (may contain extra non-structure
            columns, which are preserved and prepended to the result
            unchanged).
        structure_names: Names of the coarse output structures to compute.
        index_list: For each entry in ``structure_names``, the list of
            FreeSurfer label indices (columns in ``df_synthseg``, matched
            by name via the color LUT) to sum together.
        possible_index: Optional whitelist restricting which label
            indices are actually summed (columns not in this set are
            excluded even if listed in ``index_list``).

    Returns:
        A DataFrame with any preserved non-structure columns from
        ``df_synthseg`` followed by one integer-volume column per entry
        in ``structure_names``.
    """

    # load id and names matche
    df_free_surfer = read_freesurfer_color_table()
    structure_id_list = df_free_surfer["ID"].tolist()
    structure_name_list = df_free_surfer["NAME"].tolist()
    # replace ["-", "_"] with " " in structure names
    structure_name_list = [s.replace("-", " ").replace("_", " ").replace("*","").lower() for s in structure_name_list]
    structure_name_list = [s.replace("inf lat vent", "inferior lateral ventricle").replace("ventraldc", "ventral dc") for s in structure_name_list]


    cols_synthseg = df_synthseg.columns.tolist()

    # # replace the structures names with the correspding id
    new_columns = []
    org_columns = []
    for col in cols_synthseg:
        s = col.replace("-", " ").replace("_", " ").lower()
        if s in structure_name_list:
            structure_id = structure_id_list[structure_name_list.index(s)]
            new_columns.append(structure_id)
        else:
            new_columns.append(s)
            org_columns.append(col)

    # merge structure to compute the total volume of the structures in structure_names
    df_synthseg_new_cols = df_synthseg.copy()
    df_synthseg_new_cols.columns = new_columns
    new_df = pd.DataFrame()
    for my_structure_name, my_structure_index in zip(structure_names, index_list):
        valid_columns = [str(s) for s in my_structure_index if str(s) in df_synthseg_new_cols.columns]
        if possible_index is not None:
            valid_columns = [s for s in valid_columns if int(s) in possible_index]

        new_df[my_structure_name] = df_synthseg_new_cols[valid_columns].sum(axis=1).astype(int)

    if len(org_columns) > 0:
        new_df = pd.concat([df_synthseg[org_columns], new_df], axis=1)

    return new_df


def combine_left_and_right_seg(seg: np.ndarray) -> np.ndarray:
    """Merge left/right-hemisphere label pairs into their left-side label value.

    Applies this merge across cortical (96- and 36-class), white matter,
    subcortical, ventricle, and extra subcortical/white-matter structures.

    Args:
        seg: Segmentation to merge.

    Returns:
        A new segmentation array (copy of ``seg``) with all right-side
        labels relabeled to their corresponding left-side value.
    """
    seg_combined = seg.copy()
    # cortical structures
    for left, right in zip(CEREBRAL_CORTEX_LEFT_96, CEREBRAL_CORTEX_RIGHT_96):
        seg_combined[seg == right] = left
        
    # if seg 36 is used
    for left, right in zip(CEREBRAL_CORTEX_LEFT_36, CEREBRAL_CORTEX_RIGHT_36):
        seg_combined[seg == right] = left
    
    # withe matter
    for left, right in zip(CEREBRAL_WM_LEFT, CEREBRAL_WM_RIGHT):
        seg_combined[seg == right] = left
        
    # subcortical structures
    for left, right in zip(CEREBRAL_SUB_CORTICAL_GM_LEFT, CEREBRAL_SUB_CORTICAL_GM_RIGHT):
        seg_combined[seg == right] = left 
        
    # ventricles
    for left, right in zip(LATERAL_VENTRICLES_LEFT, LATERAL_VENTRICLES_RIGHT):
        seg_combined[seg == right] = left
        
    # inferior lateral ventricles
    for left, right in zip(INFERIOR_LATERAL_VENTRICLES[0:1], INFERIOR_LATERAL_VENTRICLES[1:2]):
        seg_combined[seg == right] = left

    # extra subcortical structures
    for left, right in zip(EXTRA_CEREBRAL_SUB_CORTICAL_GM_LEFT, EXTRA_CEREBRAL_SUB_CORTICAL_GM_RIGHT):
        seg_combined[seg == right] = left

    # extra white matter
    for left, right in zip(EXTRA_CEREBRAL_WM_LEFT, EXTRA_CEREBRAL_WM_RIGHT):
        seg_combined[seg == right] = left
    
    return seg_combined


def merge_segmentation(seg: np.ndarray, mapping_dict: dict[int, Sequence[int]]) -> np.ndarray:
    """Relabel a segmentation according to an arbitrary new-label -> old-labels mapping.

    Args:
        seg: Input segmentation array.
        mapping_dict: ``{new_label: [old_label, ...]}``, e.g.
            ``{1: [10, 11, 12], 2: [20, 21]}`` merges labels 10/11/12 into
            new label 1 and labels 20/21 into new label 2.

    Returns:
        A new array (same shape as ``seg``, initialized to all zeros)
        with only the labels named in ``mapping_dict`` set, to their new
        values.
    """
    merged_segmentation = np.zeros_like(seg) 
    
    for new_label, old_labels in mapping_dict.items():
        for old_label in old_labels:
            merged_segmentation[seg == old_label] = new_label
            
    return merged_segmentation


# not really sure it is true, this is from chat gpt
freesurfer_lobes = {
    "frontal": [
        # Left hemisphere
        1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1024, 1026, 1027, 1028, 1032,
        # Right hemisphere
        2002, 2003, 2012, 2014, 2017, 2018, 2019, 2020, 2024, 2026, 2027, 2028, 2032,
    ],
    "parietal": [
        # Left hemisphere
        1008, 1022, 1025, 1029, 1031, 1023, 1010,
        # Right hemisphere
        2008, 2022, 2025, 2029, 2031, 2023, 2010,
    ],
    "temporal": [
        # Left hemisphere
        1001, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034, 1035,
        # Right hemisphere
        2001, 2006, 2007, 2009, 2015, 2016, 2030, 2033, 2034, 2035,
    ],
    "occipital": [
        # Left hemisphere
        1005, 1011, 1013, 1021,
        # Right hemisphere
        2005, 2011, 2013, 2021,
    ]
}