
import numpy as np  

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
INTER_CSF = CEREBRAL_VENTRICLES + NO_CEREBRAL_VENTRICLES

SURROUNDING_CSF = [24]
CSF = INTER_CSF + SURROUNDING_CSF


# COMBINED
PARECHIMA_36 = CEREBRAL_GM_36 + CEREBRAL_WM
PARECHIMA_96 = CEREBRAL_GM_96 + CEREBRAL_WM

GM_36 = CEREBRAL_GM_36 + CEREBELLUM_GM + BRAINSTEM
GM_96 = CEREBRAL_GM_96 + CEREBELLUM_GM + BRAINSTEM

WM = CEREBRAL_WM + CEREBELLUM_WM

# TOTAL
TOTAL_96 = GM_96 + WM + CSF
TOTAL_36 = GM_36 + WM + CSF

TOTAL_96_NO_SURROUNDING_CSF = GM_96 + WM + INTER_CSF
TOTAL_36_NO_SURROUNDING_CSF = GM_36 + WM + INTER_CSF





structures_index_dict = {
    "total": None,

    "surrounding_csf": SURROUNDING_CSF,
    "lateral_ventricles": LATERAL_VENTRICLES,
    "fourth_ventricle": FOURTH_VENTRICLE,
    "third_ventricle": THIRD_VENTRICLE,


    "cortical_gm": CEREBRAL_CORTEX_96,
    "thalamus": THALAMUS_V0 + THALAMUS_V1,
    "hippocampus": HIPPOCAMPUS,
    "caudate": CAUDATE,
    "putamen": PUTAMEN,
    "pallidum": PALLIDUM,
    "amygdala": AMYGDALA,
    "accumbens_area": ACCUMBENS_AREA,
    "ventral_dc": VENTRAL_DC,

    "cerebral_wm": CEREBRAL_WM,

    "cerebellum_gm": CEREBELLUM_GM,
    "cerebellum_wm": CEREBELLUM_WM,
    "brainstem": BRAINSTEM,

}

STRUCTURE_NAME_LIST = [f"{vol}_vol" for vol in list(structures_index_dict.keys())]
STRUCTURE_INDEX_LIST = list(structures_index_dict.values())


def merge_seg_to_mask(seg, tissue_list):
    mask = np.isin(seg, tissue_list)
    mask = np.where(mask, 1, 0)
    return mask


def get_volumes(seg, structure_names_list):
    # 1. generate random volumes using the brainst_vol model
    # ---- 1.1 instantiate model
    vols_dict = {}
    # for i, index in enumerate(STRUCTURE_INDEX_LIST):
    for key in structure_names_list:
        if key not in STRUCTURE_NAME_LIST:
            raise ValueError(f"Structure name {key} not found in STRUCTURE_NAME_LIST")
        
        index = STRUCTURE_INDEX_LIST[STRUCTURE_NAME_LIST.index(key)]
        
        if index is None:
            mask_seg = np.where(seg > 0, 1, 0)
        else:
            mask_seg = merge_seg_to_mask(seg, index)

        vols_dict[key] = np.sum(mask_seg)
    return vols_dict




def normalize_by_icv(vols_dict, structure_names_list, icv_column="total_vol", percentage=False):
    norm_vols_dict = vols_dict.copy()
    if vols_dict[icv_column] <= 0:
        print(f"Warning: ICV volume is zero, cannot normalize by ICV. Returning original volumes.")
        return vols_dict
    
    for s in structure_names_list:
        if s != icv_column:
            norm_vols_dict[s] = (vols_dict[s] / vols_dict[icv_column]) * 100 if percentage else vols_dict[s] / vols_dict[icv_column]
    return norm_vols_dict
            
