"""Normalization utilities for images and per-structure brain volumes.

Two families of normalizers are provided for ROI volumes:
    - ZScoreStandardizer(BrainStructures): classic (mean, std) standardization,
      optionally using robust (median, IQR) statistics.
    - OutlierRobustNormalizer(BrainStructures): min-max scaling to a target
      range after clipping outliers via the Tukey (IQR) rule.

`SavedNormalizerBrainStructures` is a thin dispatcher that loads either
family from a saved JSON file and exposes a uniform interface.

A separate `normalize_image` function normalizes voxel intensities (not
ROI volumes) to [0, 1] using robust percentiles.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd


class ZScoreStandardizer:
    """Standardizes a 1D array of scalar values to zero mean / unit scale.

    Supports two modes:
        - robust=True: center = median, scale = interquartile range (IQR).
        - robust=False: center = mean, scale = standard deviation.
    """

    def __init__(self, robust: bool = True) -> None:
        """Initialize the standardizer.

        Args:
            robust: If True, use median/IQR statistics (less sensitive to
                outliers). If False, use mean/std.
        """
        self.mean: float | None = None
        self.std: float | None = None
        self.robust = robust

    def fit(self, data: np.ndarray | Sequence[float]) -> None:
        """Compute and store the center/scale statistics from ``data``.

        Args:
            data: 1D array-like of values to fit on.

        Side Effects:
            Sets ``self.mean`` and ``self.std``. If the computed scale is
            zero (constant data), ``self.std`` is set to 1.0 to avoid
            division by zero in :meth:`transform`.
        """
        data = np.asarray(data)
        if self.robust:
            self.mean = np.median(data)
            self.std = np.quantile(data, 0.75) - np.quantile(data, 0.25)
        else:
            self.mean = np.mean(data)
            self.std = np.std(data)

        if self.std == 0:
            self.std = 1.0

    def transform(self, data: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply ``(data - mean) / std`` using previously fitted statistics."""
        data = np.asarray(data)
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray | Sequence[float]) -> np.ndarray:
        """Invert :meth:`transform`: ``data * std + mean``."""
        data = np.asarray(data)
        return data * self.std + self.mean

    def fit_transform(self, data: np.ndarray | Sequence[float]) -> np.ndarray:
        """Fit on ``data`` then transform it in one call."""
        self.fit(data)
        return self.transform(data)

    def load_params(self, mean: float, std: float) -> None:
        """Load previously fitted statistics directly (skipping `fit`).

        Args:
            mean: Center statistic (median or mean, depending on how the
                original standardizer was fit).
            std: Scale statistic. If 0, it is coerced to 1.0.
        """
        self.mean = mean
        self.std = std if std != 0 else 1.0


class ZScoreStandardizerBrainStructures:
    """Per-structure z-score standardizer for a table of brain-ROI volumes.

    Wraps one :class:`ZScoreStandardizer` per structure (column) so that
    each ROI is standardized independently using its own statistics.
    """

    def __init__(self, structure_list: list[str], robust: bool = True) -> None:
        """Initialize one standardizer per structure.

        Args:
            structure_list: Column names (structure names) to standardize.
            robust: Passed through to each :class:`ZScoreStandardizer`.
        """
        self.structure_list = structure_list
        self.standarizers = {name: ZScoreStandardizer(robust=robust) for name in structure_list}

    def fit(self, data_df: pd.DataFrame) -> None:
        """Fit each structure's standardizer on the corresponding column."""
        for structure_name in self.structure_list:
            self.standarizers[structure_name].fit(data_df[structure_name].values)

    def transform(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize every structure column present in ``data_df``.

        Args:
            data_df: DataFrame containing one or more structure columns.

        Returns:
            A copy of ``data_df`` with the matching columns standardized.
            Columns not in ``self.structure_list`` are left untouched.
        """
        data_df_copy = data_df.copy()
        for structure_name in self.structure_list:
            if structure_name in data_df_copy.columns:
                data_df_copy[structure_name] = self.standarizers[structure_name].transform(
                    data_df_copy[structure_name].values
                )
        return data_df_copy

    def inverse_transform(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Invert :meth:`transform` for every structure in ``self.structure_list``."""
        data_df_copy = data_df.copy()
        for structure_name in self.structure_list:
            data_df_copy[structure_name] = self.standarizers[structure_name].inverse_transform(
                data_df_copy[structure_name].values
            )
        return data_df_copy

    def transform_single(self, data: float | np.ndarray, structure: str) -> float | np.ndarray:
        """Standardize a single value (or array) for one named structure.

        Raises:
            ValueError: If ``structure`` is not in ``self.structure_list``.
        """
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.standarizers[structure].transform(data)

    def inverse_transform_single(self, data: float | np.ndarray, structure: str) -> float | np.ndarray:
        """Invert :meth:`transform_single` for one named structure.

        Raises:
            ValueError: If ``structure`` is not in ``self.structure_list``.
        """
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.standarizers[structure].inverse_transform(data)

    def fit_transform(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``data_df`` then transform it in one call."""
        self.fit(data_df)
        return self.transform(data_df)

    def load_params(self, json_path: str) -> None:
        """Load per-structure (mean, std) parameters from a JSON file.

        Note:
            This replaces ``self.structure_list`` with whatever structure
            names are present as keys in the JSON file, so the resulting
            object may cover a different set of structures than the one
            passed to ``__init__``.

        Args:
            json_path: Path to a JSON file mapping structure name ->
                ``{"mean": float, "std": float}``.
        """
        with open(json_path, "r", encoding="utf-8") as params_file:
            params = json.load(params_file)

        self.structure_list = list(params.keys())
        for structure_name in params:
            self.standarizers[structure_name] = ZScoreStandardizer()
            self.standarizers[structure_name].load_params(
                params[structure_name]["mean"], params[structure_name]["std"]
            )

    def save_params(self, json_path: str) -> None:
        """Save each structure's (mean, std) parameters to a JSON file."""
        params = {
            structure_name: {
                "mean": self.standarizers[structure_name].mean,
                "std": self.standarizers[structure_name].std,
            }
            for structure_name in self.structure_list
        }
        with open(json_path, "w", encoding="utf-8") as params_file:
            json.dump(params, params_file, indent=4)


class OutlierRobustNormalizer:
    """Robust min-max normalizer that excludes outliers via the IQR (Tukey) rule.

    Computes normalization parameters on training data by clipping
    outliers (based on the interquartile range), then scales the clipped
    data to a target ``[scale_min, scale_max]`` range. Learned parameters
    can be reused to normalize held-out data consistently.
    """

    def __init__(
        self,
        lower_percentile: float = 25,
        upper_percentile: float = 75,
        scale_min: float = 0.0,
        scale_max: float = 1.0,
        tukey_factor: float = 1.5,
    ) -> None:
        """Initialize the normalizer.

        Args:
            lower_percentile: Lower percentile (0-100) defining Q1.
            upper_percentile: Upper percentile (0-100) defining Q3.
            scale_min: Minimum of the output range.
            scale_max: Maximum of the output range.
            tukey_factor: Multiplier on the IQR used to define the
                outlier-clipping bounds (1.5 is the classic Tukey fence).
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.tukey_factor = tukey_factor
        self.lower_bound: float | None = None
        self.upper_bound: float | None = None
        self.data_min: float | None = None
        self.data_max: float | None = None

    def fit(self, data: np.ndarray | Sequence[float]) -> None:
        """Compute outlier-clipping bounds and the resulting data range.

        Args:
            data: 1D array-like of training data.

        Side Effects:
            Sets ``self.lower_bound``, ``self.upper_bound``,
            ``self.data_min``, ``self.data_max``. If the clipped data is
            constant, ``self.data_max`` is nudged by ``1e-8`` to avoid a
            zero-division in :meth:`transform`.
        """
        data = np.asarray(data)
        first_quartile = np.percentile(data, self.lower_percentile)
        third_quartile = np.percentile(data, self.upper_percentile)
        interquartile_range = third_quartile - first_quartile

        self.lower_bound = first_quartile - self.tukey_factor * interquartile_range
        self.upper_bound = third_quartile + self.tukey_factor * interquartile_range

        clipped_data = np.clip(data, self.lower_bound, self.upper_bound)
        self.data_min = clipped_data.min()
        self.data_max = clipped_data.max()

        if self.data_max == self.data_min:
            self.data_max += 1e-8

    def transform(
        self,
        data: np.ndarray | Sequence[float],
        clip_data: bool = False,
        remove_data: bool = False,
    ) -> np.ndarray:
        """Apply the learned min-max normalization to new data.

        Args:
            data: 1D array-like to normalize.
            clip_data: If True, clip values to ``[lower_bound, upper_bound]``
                before scaling.
            remove_data: If True, drop values outside
                ``[lower_bound, upper_bound]`` instead of clipping them.
                Mutually exclusive in effect with ``clip_data`` (if both
                are True, ``clip_data`` takes precedence, matching the
                original ``if/elif`` behavior).

        Returns:
            Normalized array in ``[scale_min, scale_max]`` (values outside
            the fitted range may fall slightly outside this interval
            unless ``clip_data`` or ``remove_data`` is used).

        Raises:
            RuntimeError: If called before :meth:`fit` or :meth:`load_params`.
            ValueError: If ``remove_data=True`` and every point is an outlier.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("The normalizer must be fitted before calling transform.")

        data = np.asarray(data)
        if clip_data:
            clipped_data = np.clip(data, self.lower_bound, self.upper_bound)
        elif remove_data:
            valid_mask = (data >= self.lower_bound) & (data <= self.upper_bound)
            clipped_data = data[valid_mask]
            if clipped_data.size == 0:
                raise ValueError("All data points are considered outliers.")
        else:
            clipped_data = data.copy()

        normalized = (clipped_data - self.data_min) / (self.data_max - self.data_min)
        normalized = normalized * (self.scale_max - self.scale_min) + self.scale_min
        return normalized

    def fit_transform(self, data: np.ndarray | Sequence[float], clip_data: bool = True) -> np.ndarray:
        """Fit on ``data`` then transform it in one call.

        Note:
            Defaults to ``clip_data=True`` here (unlike :meth:`transform`,
            whose default is False) — this mirrors the original API.
        """
        self.fit(data)
        return self.transform(data, clip_data=clip_data)

    def inverse_transform(self, normalized_data: np.ndarray | Sequence[float], add_min: bool = True) -> np.ndarray:
        """Invert the min-max normalization back to the original data scale.

        Args:
            normalized_data: Data in the ``[scale_min, scale_max]`` range.
            add_min: If True (default), add back ``data_min`` to recover
                absolute values. If False, returns the value relative to
                ``data_min`` (i.e., the scaled range only).

        Raises:
            RuntimeError: If called before :meth:`fit` or :meth:`load_params`.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("The normalizer must be fitted before calling inverse_transform.")

        normalized_data = np.asarray(normalized_data)
        scaled_to_unit_range = (normalized_data - self.scale_min) / (self.scale_max - self.scale_min)
        original_scale = scaled_to_unit_range * (self.data_max - self.data_min)
        if add_min:
            original_scale += self.data_min
        return original_scale

    def load_params(self, data_min: float, data_max: float, lower_bound: float, upper_bound: float) -> None:
        """Load previously fitted parameters directly (skipping `fit`).

        Args:
            data_min: Minimum of the clipped training data.
            data_max: Maximum of the clipped training data.
            lower_bound: Lower outlier-clipping bound.
            upper_bound: Upper outlier-clipping bound.
        """
        self.data_min = data_min
        self.data_max = data_max
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class OutlierRobustNormalizerBrainStructures:
    """Per-structure :class:`OutlierRobustNormalizer` for a volumes table."""

    def __init__(
        self,
        structure_list: list[str],
        lower_percentile: float = 25,
        upper_percentile: float = 75,
        scale_min: float = 0.0,
        scale_max: float = 1.0,
        tukey_factor: float = 1.5,
    ) -> None:
        """Initialize one normalizer per structure with shared hyperparameters."""
        self.structure_list = structure_list
        self.normalizer = {
            name: OutlierRobustNormalizer(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                scale_min=scale_min,
                scale_max=scale_max,
                tukey_factor=tukey_factor,
            )
            for name in structure_list
        }
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.tukey_factor = tukey_factor

    def fit(self, data_df: pd.DataFrame) -> None:
        """Fit each structure's normalizer on the corresponding column."""
        for structure_name in self.structure_list:
            self.normalizer[structure_name].fit(data_df[structure_name].values)

    def transform(self, data_df: pd.DataFrame, clip_data: bool = False, remove_data: bool = False) -> pd.DataFrame:
        """Normalize every structure column present in ``data_df``. See
        :meth:`OutlierRobustNormalizer.transform` for ``clip_data``/``remove_data``.
        """
        data_df_copy = data_df.copy()
        for structure_name in self.structure_list:
            if structure_name in data_df_copy.columns:
                data_df_copy[structure_name] = self.normalizer[structure_name].transform(
                    data_df_copy[structure_name].values, clip_data=clip_data, remove_data=remove_data
                )
        return data_df_copy

    def inverse_transform(self, data_df: pd.DataFrame, add_min: bool = True) -> pd.DataFrame:
        """Invert :meth:`transform` for every structure in ``self.structure_list``."""
        data_df_copy = data_df.copy()
        for structure_name in self.structure_list:
            data_df_copy[structure_name] = self.normalizer[structure_name].inverse_transform(
                data_df_copy[structure_name].values, add_min=add_min
            )
        return data_df_copy

    def transform_single(
        self, data: float | np.ndarray, structure: str, clip_data: bool = False, remove_data: bool = False
    ) -> float | np.ndarray:
        """Normalize a single value (or array) for one named structure.

        Raises:
            ValueError: If ``structure`` is not in ``self.structure_list``.
        """
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.normalizer[structure].transform(data, clip_data=clip_data, remove_data=remove_data)

    def inverse_transform_single(
        self, data: float | np.ndarray, structure: str, add_min: bool = True
    ) -> float | np.ndarray:
        """Invert :meth:`transform_single` for one named structure.

        Raises:
            ValueError: If ``structure`` is not in ``self.structure_list``.
        """
        if structure not in self.structure_list:
            raise ValueError(f"Structure {structure} not in structure list.")
        return self.normalizer[structure].inverse_transform(data, add_min=add_min)

    def fit_transform(
        self, data_df: pd.DataFrame, clip_data: bool = False, remove_data: bool = False
    ) -> pd.DataFrame:
        """Fit on ``data_df`` then transform it in one call."""
        self.fit(data_df)
        return self.transform(data_df, clip_data=clip_data, remove_data=remove_data)

    def load_params(self, json_path: str) -> None:
        """Load per-structure min-max parameters from a JSON file.

        Note:
            As with :meth:`ZScoreStandardizerBrainStructures.load_params`,
            this replaces ``self.structure_list`` with the keys found in
            the JSON file.

        Args:
            json_path: Path to a JSON file mapping structure name ->
                ``{"data_min", "data_max", "lower_bound", "upper_bound"}``.
        """
        with open(json_path, "r", encoding="utf-8") as params_file:
            params = json.load(params_file)

        self.structure_list = list(params.keys())
        for structure_name in params:
            self.normalizer[structure_name] = OutlierRobustNormalizer(
                lower_percentile=self.lower_percentile,
                upper_percentile=self.upper_percentile,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                tukey_factor=self.tukey_factor,
            )
            self.normalizer[structure_name].load_params(
                params[structure_name]["data_min"],
                params[structure_name]["data_max"],
                params[structure_name]["lower_bound"],
                params[structure_name]["upper_bound"],
            )

    def save_params(self, json_path: str) -> None:
        """Save each structure's min-max parameters to a JSON file."""
        params = {
            structure_name: {
                "data_min": self.normalizer[structure_name].data_min,
                "data_max": self.normalizer[structure_name].data_max,
                "lower_bound": self.normalizer[structure_name].lower_bound,
                "upper_bound": self.normalizer[structure_name].upper_bound,
            }
            for structure_name in self.structure_list
        }
        with open(json_path, "w", encoding="utf-8") as params_file:
            json.dump(params, params_file, indent=4)


class SavedNormalizerBrainStructures:
    """Loads either normalizer family from a saved JSON file, auto-detecting the type.

    Inspects the JSON keys to decide whether the file was produced by a
    :class:`ZScoreStandardizerBrainStructures` (keys ``mean``/``std``) or
    an :class:`OutlierRobustNormalizerBrainStructures`
    (keys ``data_min``/``data_max``/``lower_bound``/``upper_bound``), then
    wraps the appropriate class and exposes a uniform interface.
    """

    def __init__(self, normalizer_params: str) -> None:
        """Load and auto-detect the normalizer type from a JSON params file.

        Args:
            normalizer_params: Path to the saved normalizer JSON file.

        Raises:
            ValueError: If the JSON structure matches neither known
                normalizer parameter schema.
        """
        with open(normalizer_params, "r", encoding="utf-8") as params_file:
            params = json.load(params_file)

        first_structure_params = list(params.values())[0]
        if "mean" in first_structure_params and "std" in first_structure_params:
            self.normalizer = ZScoreStandardizerBrainStructures(structure_list=list(params.keys()))
        elif all(
            key in first_structure_params for key in ("data_min", "data_max", "lower_bound", "upper_bound")
        ):
            self.normalizer = OutlierRobustNormalizerBrainStructures(structure_list=list(params.keys()))
        else:
            raise ValueError("Normalizer parameters not recognized.")
        self.normalizer.load_params(normalizer_params)

    def transform(self, data_df: pd.DataFrame, clip_data: bool = False, remove_data: bool = False) -> pd.DataFrame:
        """Normalize ``data_df`` using the wrapped normalizer.

        Note:
            ``clip_data``/``remove_data`` are ignored when the wrapped
            normalizer is a :class:`ZScoreStandardizerBrainStructures`,
            since that normalizer has no such options.
        """
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.transform(data_df)
        return self.normalizer.transform(data_df, clip_data=clip_data, remove_data=remove_data)

    def inverse_transform(self, data_df: pd.DataFrame, add_min: bool = True) -> pd.DataFrame:
        """Invert normalization using the wrapped normalizer."""
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.inverse_transform(data_df)
        return self.normalizer.inverse_transform(data_df, add_min=add_min)

    def transform_single(
        self, data: float | np.ndarray, structure: str, clip_data: bool = False, remove_data: bool = False
    ) -> float | np.ndarray:
        """Normalize a single value for one structure using the wrapped normalizer."""
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.transform_single(data, structure)
        return self.normalizer.transform_single(data, structure, clip_data=clip_data, remove_data=remove_data)

    def inverse_transform_single(
        self, data: float | np.ndarray, structure: str, add_min: bool = True
    ) -> float | np.ndarray:
        """Invert normalization for a single value using the wrapped normalizer."""
        if isinstance(self.normalizer, ZScoreStandardizerBrainStructures):
            return self.normalizer.inverse_transform_single(data, structure)
        return self.normalizer.inverse_transform_single(data, structure, add_min=add_min)


def _normalize_single_volume_by_icv(
    volume: float,
    icv: float,
    percentage: bool,
) -> float:
    """Normalize a single structure volume by intracranial volume (ICV).

    Args:
        volume: Raw volume of the structure.
        icv: Intracranial volume for the same subject/session.
        percentage: If True, express the result as a percentage of ICV.

    Returns:
        ``volume`` normalized by ``icv`` (or the unmodified ``volume`` if
        ``icv <= 0``, since division would be invalid).
    """
    if icv <= 0:
        return volume
    normalized = volume / icv
    return normalized * 100 if percentage else normalized


def normalize_by_icv(
    brain_stats_df: pd.DataFrame,
    structure_names: list[str],
    icv_column: str = "total_vol",
    percentage: bool = False,
) -> pd.DataFrame:
    """Normalize each structure's volume by intracranial volume (ICV).

    Args:
        brain_stats_df: DataFrame with one row per subject/session and one
            column per structure volume.
        structure_names: Columns to normalize (``icv_column`` itself is
            skipped, i.e. left as an absolute volume).
        icv_column: Column used as the intracranial volume reference.
        percentage: If True, express the normalized volume as a percentage
            of ICV (multiply by 100) instead of a raw fraction.

    Returns:
        A copy of ``brain_stats_df`` with the specified structures
        normalized. Rows where ``icv_column <= 0`` are left unmodified for
        that structure (division by a non-positive ICV is skipped).
    """
    normalized_df = brain_stats_df.copy()

    for structure_name in structure_names:
        if structure_name == icv_column:
            continue
        normalized_df[structure_name] = normalized_df.apply(
            lambda row: _normalize_single_volume_by_icv(
                row[structure_name], row[icv_column], percentage
            ),
            axis=1,
        )

    return normalized_df


def normalize_image(
    image: np.ndarray,
    percentile: tuple[float, float] = (0, 100),
    mask: np.ndarray | None = None,
    reference_tensor: np.ndarray | None = None,
    strictly_positive: bool = True,
    clip_values: bool = True,
) -> np.ndarray:
    """Normalize image intensities to ``[0, 1]`` using robust percentiles.

    Args:
        image: Image (or tensor) to normalize.
        percentile: ``(p_min, p_max)`` percentiles (0-100) used to compute
            the normalization range.
        mask: Optional boolean/binary mask; if given, percentiles are
            computed only over voxels where ``mask > 0``.
        reference_tensor: Optional array to compute percentiles from
            instead of ``image`` itself (``image`` is still what gets
            clipped/scaled). Defaults to ``image``.
        strictly_positive: If True, clamp the lower percentile to 0 when
            it would otherwise be negative.
        clip_values: If True, clip ``image`` to ``[p_min, p_max]`` before
            scaling.

    Returns:
        Image normalized to ``[0, 1]`` (assuming default percentile/clip
        settings). If ``p_max <= p_min`` (e.g., a constant image), returns
        an all-zeros array of the same shape as ``image``.
    """
    reference = reference_tensor if reference_tensor is not None else image

    if mask is not None:
        reference = reference[mask > 0]

    lower_bound, upper_bound = np.percentile(reference, percentile)

    if strictly_positive and lower_bound < 0:
        lower_bound = 0

    if clip_values:
        clipped_image = np.clip(image, lower_bound, upper_bound)
    else:
        clipped_image = image

    if upper_bound > lower_bound:
        normalized_image = (clipped_image - lower_bound) / (upper_bound - lower_bound)
    else:
        normalized_image = np.zeros_like(image)

    return normalized_image