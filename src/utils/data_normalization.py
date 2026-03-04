import json
import numpy as np

class ZScoreStandardizer:
    def __init__(self, robust=True):
        self.mean = None
        self.std = None
        self.robust = robust

    def fit(self, data):
        data = np.asarray(data)
        if self.robust:
            self.mean = np.median(data)
            # mad = np.median(np.abs(data - self.mean))
            # self.std = 1.4826 * mad  # Scale MAD to match standard deviation for normal distribution
            self.std = np.quantile(data, 0.75) - np.quantile(data, 0.25)
            # self.mean = np.mean(data)
            # self.std = np.std(data)  # Scale MAD to match standard deviation for normal distribution
        else:
            self.mean = np.mean(data)
            self.std = np.std(data)

        if self.std == 0:
            self.std = 1.0

    def transform(self, data):
        data = np.asarray(data)
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        data = np.asarray(data)
        return data * self.std + self.mean

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def load_params(self, mean, std):
        self.mean = mean
        self.std = std if std != 0 else 1.0



class ZScoreStandardizerBrainStructures:
    def __init__(self, structure_list, robust=True):
        self.structure_list = structure_list
        self.standarizers = {s: ZScoreStandardizer(robust=robust) for s in structure_list}

    def fit(self, data_df):
        for s in self.structure_list:
            self.standarizers[s].fit(data_df[s].values)
    
    def transform(self, data_df):
        data_df_copy = data_df.copy()
        for s in self.structure_list:
            if s in data_df_copy.columns:
                data_df_copy[s] = self.standarizers[s].transform(data_df_copy[s].values)
        return data_df_copy
    
    def inverse_transform(self, data_df):
        data_df_copy = data_df.copy()
        for s in self.structure_list:
            data_df_copy[s] = self.standarizers[s].inverse_transform(data_df_copy[s].values)
        return data_df_copy
    
    def transform_single(self, data, structure):
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.standarizers[structure].transform(data)

    def inverse_transform_single(self, data, structure):
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.standarizers[structure].inverse_transform(data)

    def fit_transform(self, data_df):
        self.fit(data_df)
        return self.transform(data_df)
    
    def load_params(self, json_path):
        # load from json file
        with open(json_path, "r") as f:
            params = json.load(f)
        # for s in self.structure_list:
        #     self.standarizers[s].load_params(params[s]["mean"], params[s]["std"])

        # obtain the keys in params to avoid issues if structure_list is different
        self.structure_list = list(params.keys())

        for s in params.keys():
            self.standarizers[s] = ZScoreStandardizer()
            self.standarizers[s].load_params(params[s]["mean"], params[s]["std"])

    def save_params(self, json_path):
        # save to json file
        params = {}
        for s in self.structure_list:
            params[s] = {
                "mean": self.standarizers[s].mean,
                "std": self.standarizers[s].std
            }
        with open(json_path, "w") as f:
            json.dump(params, f, indent=4)







class OutlierRobustNormalizer:
    """
    A robust min-max normalizer that excludes outliers using the IQR method.

    This class computes normalization parameters on training data by clipping outliers
    (based on the interquartile range), then scales the clipped data to a specified range.
    The learned parameters can be reused to normalize test data consistently.
    """

    def __init__(self, lower_percentile=25, upper_percentile=75, scale_min=0.0, scale_max=1.0, tukey_factor=1.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.tukey_factor = tukey_factor
        self.lower_bound = None
        self.upper_bound = None
        self.data_min = None
        self.data_max = None

    def fit(self, data):
        """
        Compute the normalization parameters excluding outliers.

        Parameters:
        - data (array-like): 1D array of training data.
        """
        data = np.asarray(data)
        q1 = np.percentile(data, self.lower_percentile)
        q3 = np.percentile(data, self.upper_percentile)
        iqr = q3 - q1

        self.lower_bound = q1 - self.tukey_factor * iqr
        self.upper_bound = q3 + self.tukey_factor * iqr

        # Clip the data to remove outliers
        clipped = np.clip(data, self.lower_bound, self.upper_bound)
        self.data_min = clipped.min()
        self.data_max = clipped.max()

        # Avoid division by zero
        if self.data_max == self.data_min:
            self.data_max += 1e-8

    def transform(self, data, clip_data=False, remove_data=False):
        """
        Apply the learned normalization to new data.

        Parameters:
        - data (array-like): 1D array to be normalized.

        Returns:
        - normalized (np.array): Normalized array.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("The normalizer must be fitted before calling transform.")

        data = np.asarray(data)
        if clip_data:
            # Clip the data to the learned bounds
            clipped = np.clip(data, self.lower_bound, self.upper_bound)
        elif remove_data:
            mask = (data >= self.lower_bound) & (data <= self.upper_bound)
            clipped = data[mask]
            if clipped.size == 0:
                raise ValueError("All data points are considered outliers.")
        else:
            clipped = data.copy()
        normalized = (clipped - self.data_min) / (self.data_max - self.data_min)
        normalized = normalized * (self.scale_max - self.scale_min) + self.scale_min

        return normalized

    def fit_transform(self, data, clip_data=True):
        """
        Fit the normalizer on training data and return the normalized result.

        Parameters:
        - data (array-like): 1D array of training data.

        Returns:
        - normalized (np.array): Normalized array.
        """
        self.fit(data)
        return self.transform(data, clip_data=clip_data)

    def inverse_transform(self, normalized_data, add_min=True):
        """
        Reverse the normalization transformation.

        Parameters:
        - normalized_data (array-like): Normalized data to be transformed back to original scale.

        Returns:
        - original_scale (np.array): Data in the original scale.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("The normalizer must be fitted before calling inverse_transform.")

        normalized_data = np.asarray(normalized_data)
        
        # Reverse the scaling to [0, 1] range
        scaled_back = (normalized_data - self.scale_min) / (self.scale_max - self.scale_min)
        
        # Reverse the min-max normalization
        original_scale = scaled_back * (self.data_max - self.data_min)
        if add_min:
            original_scale += self.data_min
        return original_scale

    def load_params(self, data_min, data_max, lower_bound, upper_bound):
        """
        Load normalization parameters.

        Parameters:
        - data_min (float): The minimum value of the training data.
        - data_max (float): The maximum value of the training data.
        - lower_bound (float): The lower bound for clipping [0-1].
        - upper_bound (float): The upper bound for clipping [0-1].

        """
        self.data_min = data_min
        self.data_max = data_max

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound



class OutlierRobustNormalizerBrainStructures:
    def __init__(self, structure_list, lower_percentile=25, upper_percentile=75, scale_min=0.0, scale_max=1.0, tukey_factor=1.5):
        self.structure_list = structure_list
        self.normalizer = {s: OutlierRobustNormalizer(lower_percentile=lower_percentile, upper_percentile=upper_percentile, scale_min=scale_min, scale_max=scale_max, tukey_factor=tukey_factor) for s in structure_list}
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.tukey_factor = tukey_factor

    def fit(self, data_df):
        for s in self.structure_list:
            self.normalizer[s].fit(data_df[s].values)
    
    def transform(self, data_df, clip_data=False, remove_data=False):
        data_df_copy = data_df.copy()
        for s in self.structure_list:
            if s in data_df_copy.columns:
                data_df_copy[s] = self.normalizer[s].transform(data_df_copy[s].values, clip_data=clip_data, remove_data=remove_data)
        return data_df_copy
    
    def inverse_transform(self, data_df, add_min=True):
        data_df_copy = data_df.copy()
        for s in self.structure_list:
            data_df_copy[s] = self.normalizer[s].inverse_transform(data_df_copy[s].values, add_min=add_min)
        return data_df_copy
    
    def transform_single(self, data, structure, clip_data=False, remove_data=False):
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.normalizer[structure].transform(data, clip_data=clip_data, remove_data=remove_data)

    def inverse_transform_single(self, data, structure, add_min=True):
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.normalizer[structure].inverse_transform(data, add_min=add_min)

    def fit_transform(self, data_df, clip_data=False, remove_data=False):
        self.fit(data_df)
        return self.transform(data_df, clip_data=clip_data, remove_data=remove_data)

    def load_params(self, json_path):
        # load from json file
        with open(json_path, "r") as f:
            params = json.load(f)
        # for s in self.structure_list:
        #     self.normalizer[s].load_params(params[s]["data_min"], params[s]["data_max"], params[s]["lower_bound"], params[s]["upper_bound"])
        # obtain the keys in params to avoid issues if structure_list is different
        self.structure_list = list(params.keys())
        for s in params.keys():
            self.normalizer[s] = OutlierRobustNormalizer(lower_percentile=self.lower_percentile, upper_percentile=self.upper_percentile, scale_min=self.scale_min, scale_max=self.scale_max, tukey_factor=self.tukey_factor)
            self.normalizer[s].load_params(params[s]["data_min"], params[s]["data_max"], params[s]["lower_bound"], params[s]["upper_bound"])


    def save_params(self, json_path):
        # save to json file
        params = {}
        for s in self.structure_list:
            params[s] = {
                "data_min": self.normalizer[s].data_min,
                "data_max": self.normalizer[s].data_max,
                "lower_bound": self.normalizer[s].lower_bound,
                "upper_bound": self.normalizer[s].upper_bound
            }
        with open(json_path, "w") as f:
            json.dump(params, f, indent=4)


class SavedNormalizerBrainStructures:
# load any normalizer from json
    def __init__(self, normalizer_params):
        # verify type of normalizer 
        # load json
        with open(normalizer_params, "r") as f:
            params = json.load(f)
        # verify if mean and std exists in params
        if "mean" in list(params.values())[0] and "std" in list(params.values())[0]:
            self.normalizer = ZScoreStandardizerBrainStructures(structure_list=list(params.keys()))
        elif "data_min" in list(params.values())[0] and "data_max" in list(params.values())[0] and "lower_bound" in list(params.values())[0] and "upper_bound" in list(params.values())[0]:
            self.normalizer = OutlierRobustNormalizerBrainStructures(structure_list=list(params.keys()))
        else:
            raise ValueError("Normalizer parameters not recognized.")
        self.normalizer.load_params(normalizer_params)


    def transform(self, data_df, clip_data=False, remove_data=False):
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.transform(data_df)
        elif isinstance(self.normalizer, OutlierRobustNormalizerBrainStructures):
            return self.normalizer.transform(data_df, clip_data=clip_data, remove_data=remove_data)        
    
    def inverse_transform(self, data_df, add_min=True):
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.inverse_transform(data_df)
        elif isinstance(self.normalizer, OutlierRobustNormalizerBrainStructures):
            return self.normalizer.inverse_transform(data_df, add_min=add_min)

    def transform_single(self, data, structure, clip_data=False, remove_data=False):
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.transform_single(data, structure)
        elif isinstance(self.normalizer, OutlierRobustNormalizerBrainStructures):
            return self.normalizer.transform_single(data, structure, clip_data=clip_data, remove_data=remove_data)
    
    def inverse_transform_single(self, data, structure, add_min=True):
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.inverse_transform_single(data, structure)
        elif isinstance(self.normalizer, OutlierRobustNormalizerBrainStructures):
            return self.normalizer.inverse_transform_single(data, structure, add_min=add_min)

    


def normalize_by_icv(brain_stats_df, structure_names, icv_column="total_vol", percentage=False):
    df = brain_stats_df.copy()
    
    for s in structure_names:
        if s != icv_column:
            df[s] = df.apply(
                lambda row: (row[s] / row[icv_column]) * 100 if row[icv_column] > 0 and percentage else row[s] / row[icv_column] if row[icv_column] > 0 else row[s],
                axis=1
            )
    
    return df