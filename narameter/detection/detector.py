"""
Narrative shift detection for Narameter pipeline.

Detects statistically significant shifts in the Wasserstein distance
time series using z-score, IQR, or fixed threshold methods.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from narameter.config import DetectionConfig

logger = logging.getLogger(__name__)


class NarrativeShiftDetector:
    """
    Detects narrative shifts from a time series of Wasserstein distances.

    Four detection methods:
    - "zscore": Flag distances > μ + z·σ (global or rolling)
    - "iqr": Flag outliers using interquartile range (Q3 + 1.5·IQR)
    - "otsu": Otsu's method — finds optimal threshold that minimizes
              intra-class variance. Best for bimodal distance distributions.
    - "threshold": Flag distances above a fixed threshold value
    """

    VALID_METHODS = {"zscore", "iqr", "otsu", "threshold"}

    # Shift magnitude categories
    MAGNITUDE_THRESHOLDS = {
        "minor": 1.0,    # 1.0 - 1.5 std devs
        "moderate": 1.5,  # 1.5 - 2.5 std devs
        "major": 2.5,     # > 2.5 std devs
    }

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the detector.

        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self.config = config or DetectionConfig()

        if self.config.method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{self.config.method}'. "
                f"Choose from: {self.VALID_METHODS}"
            )

    def detect(
        self,
        distances_df: pd.DataFrame,
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Detect narrative shifts from distance data.

        Args:
            distances_df: DataFrame with columns ["period_a", "period_b", "distance"].
            method: Detection method. Uses config default if None.

        Returns:
            DataFrame with added columns:
            - "is_shift": bool — whether this transition is a significant shift
            - "z_score": float — how many std devs above mean
            - "shift_magnitude": str — "none", "minor", "moderate", "major"
            - "threshold": float — the threshold value used
            - "rolling_mean": float — rolling average of distances
            - "rolling_std": float — rolling std of distances
        """
        method = method or self.config.method
        df = distances_df.copy()

        if "distance" not in df.columns:
            raise ValueError("DataFrame must contain a 'distance' column")

        distances = df["distance"].values

        if len(distances) < 2:
            logger.warning("Too few data points for meaningful shift detection")
            df["is_shift"] = False
            df["z_score"] = 0.0
            df["shift_magnitude"] = "none"
            df["threshold"] = 0.0
            df["rolling_mean"] = distances
            df["rolling_std"] = 0.0
            return df

        if method == "zscore":
            df = self._detect_zscore(df)
        elif method == "iqr":
            df = self._detect_iqr(df)
        elif method == "otsu":
            df = self._detect_otsu(df)
        elif method == "threshold":
            df = self._detect_threshold(df)

        # Classify magnitude
        df["shift_magnitude"] = df["z_score"].apply(self._classify_magnitude)

        # Summary log
        n_shifts = df["is_shift"].sum()
        logger.info(
            f"Detected {n_shifts}/{len(df)} narrative shifts "
            f"(method={method})"
        )

        if n_shifts > 0:
            shifts = df[df["is_shift"]]
            for _, row in shifts.iterrows():
                logger.info(
                    f"  ⚡ {row['period_a']} → {row['period_b']}: "
                    f"WD={row['distance']:.4f}, z={row['z_score']:.2f}, "
                    f"magnitude={row['shift_magnitude']}"
                )

        return df

    def _detect_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect shifts using z-score method."""
        distances = df["distance"].values
        window = min(self.config.window_size, len(distances))

        # Compute rolling statistics
        dist_series = df["distance"]

        if len(distances) <= window:
            # Not enough data for rolling — use global stats
            mean = distances.mean()
            std = distances.std()
            df["rolling_mean"] = mean
            df["rolling_std"] = std
        else:
            df["rolling_mean"] = dist_series.rolling(
                window=window, min_periods=1, center=True
            ).mean()
            df["rolling_std"] = dist_series.rolling(
                window=window, min_periods=1, center=True
            ).std().fillna(0)

        # Global statistics for z-score
        global_mean = distances.mean()
        global_std = distances.std()

        if global_std == 0:
            df["z_score"] = 0.0
            df["threshold"] = global_mean
            df["is_shift"] = False
        else:
            df["z_score"] = (distances - global_mean) / global_std
            df["threshold"] = global_mean + self.config.zscore_threshold * global_std
            df["is_shift"] = df["distance"] > df["threshold"]

        return df

    def _detect_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect shifts using IQR (Interquartile Range) method."""
        distances = df["distance"].values

        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr

        global_mean = distances.mean()
        global_std = distances.std() if distances.std() > 0 else 1.0

        df["z_score"] = (distances - global_mean) / global_std
        df["threshold"] = upper_fence
        df["is_shift"] = df["distance"] > upper_fence
        df["rolling_mean"] = global_mean
        df["rolling_std"] = global_std

        return df

    def _detect_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect shifts using a fixed threshold."""
        if self.config.fixed_threshold is None:
            raise ValueError(
                "fixed_threshold must be set in config when using 'threshold' method"
            )

        distances = df["distance"].values
        threshold = self.config.fixed_threshold

        global_mean = distances.mean()
        global_std = distances.std() if distances.std() > 0 else 1.0

        df["z_score"] = (distances - global_mean) / global_std
        df["threshold"] = threshold
        df["is_shift"] = df["distance"] > threshold
        df["rolling_mean"] = global_mean
        df["rolling_std"] = global_std

        return df

    def _detect_otsu(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect shifts using Otsu's method (1D binarization).

        Finds the threshold that minimizes the weighted intra-class variance
        of the two groups (baseline vs shift). Works well when distances
        form a bimodal distribution.
        """
        distances = df["distance"].values

        # Find optimal Otsu threshold
        sorted_vals = np.sort(distances)
        best_threshold = sorted_vals[0]
        best_variance = float("inf")

        for i in range(1, len(sorted_vals)):
            # Split into two classes
            class_low = sorted_vals[:i]
            class_high = sorted_vals[i:]

            w_low = len(class_low) / len(sorted_vals)
            w_high = len(class_high) / len(sorted_vals)

            var_low = class_low.var() if len(class_low) > 1 else 0
            var_high = class_high.var() if len(class_high) > 1 else 0

            # Weighted intra-class variance
            within_var = w_low * var_low + w_high * var_high

            if within_var < best_variance:
                best_variance = within_var
                # Threshold is the midpoint between the two classes
                best_threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2

        global_mean = distances.mean()
        global_std = distances.std() if distances.std() > 0 else 1.0

        df["z_score"] = (distances - global_mean) / global_std
        df["threshold"] = best_threshold
        df["is_shift"] = df["distance"] > best_threshold
        df["rolling_mean"] = global_mean
        df["rolling_std"] = global_std

        logger.info(f"Otsu threshold: {best_threshold:.6f}")

        return df

    def _classify_magnitude(self, z_score: float) -> str:
        """Classify the magnitude of a shift based on z-score."""
        z = abs(z_score)
        if z >= self.MAGNITUDE_THRESHOLDS["major"]:
            return "major"
        elif z >= self.MAGNITUDE_THRESHOLDS["moderate"]:
            return "moderate"
        elif z >= self.MAGNITUDE_THRESHOLDS["minor"]:
            return "minor"
        return "none"

    def summary(self, results: pd.DataFrame) -> dict:
        """
        Generate a summary of detection results.

        Args:
            results: DataFrame from detect().

        Returns:
            Dictionary with summary statistics.
        """
        distances = results["distance"].values

        summary = {
            "total_transitions": len(results),
            "total_shifts": int(results["is_shift"].sum()),
            "shift_rate": float(results["is_shift"].mean()),
            "distance_mean": float(distances.mean()),
            "distance_std": float(distances.std()),
            "distance_min": float(distances.min()),
            "distance_max": float(distances.max()),
            "distance_median": float(np.median(distances)),
            "shifts_by_magnitude": {
                mag: int((results["shift_magnitude"] == mag).sum())
                for mag in ["minor", "moderate", "major"]
            },
        }

        if results["is_shift"].any():
            shifts = results[results["is_shift"]]
            summary["largest_shift"] = {
                "period_a": shifts.loc[shifts["distance"].idxmax(), "period_a"],
                "period_b": shifts.loc[shifts["distance"].idxmax(), "period_b"],
                "distance": float(shifts["distance"].max()),
                "z_score": float(shifts.loc[shifts["distance"].idxmax(), "z_score"]),
            }

        return summary
