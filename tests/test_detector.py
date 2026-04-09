"""Tests for narrative shift detection."""

import numpy as np
import pandas as pd
import pytest

from narameter.detection.detector import NarrativeShiftDetector
from narameter.config import DetectionConfig


@pytest.fixture
def detector():
    """Create a detector with z-score method."""
    config = DetectionConfig(method="zscore", zscore_threshold=2.0)
    return NarrativeShiftDetector(config)


@pytest.fixture
def flat_distances():
    """Distances with no shifts — all roughly equal."""
    return pd.DataFrame({
        "period_a": ["W1", "W2", "W3", "W4", "W5"],
        "period_b": ["W2", "W3", "W4", "W5", "W6"],
        "distance": [0.10, 0.11, 0.09, 0.10, 0.12],
    })


@pytest.fixture
def spike_distances():
    """Distances with a clear spike at W3→W4."""
    return pd.DataFrame({
        "period_a": ["W1", "W2", "W3", "W4", "W5", "W6"],
        "period_b": ["W2", "W3", "W4", "W5", "W6", "W7"],
        "distance": [0.10, 0.11, 0.55, 0.09, 0.10, 0.12],
    })


@pytest.fixture
def multi_spike_distances():
    """Distances with multiple spikes."""
    return pd.DataFrame({
        "period_a": [f"W{i}" for i in range(1, 11)],
        "period_b": [f"W{i}" for i in range(2, 12)],
        "distance": [0.10, 0.11, 0.60, 0.10, 0.09, 0.11, 0.10, 0.55, 0.12, 0.10],
    })


class TestZScoreDetection:
    def test_no_shifts_in_flat_data(self, detector, flat_distances):
        results = detector.detect(flat_distances)
        assert not results["is_shift"].any(), "Should find no shifts in flat data"

    def test_detects_spike(self, detector, spike_distances):
        results = detector.detect(spike_distances)
        assert results["is_shift"].any(), "Should detect the spike"

        # The spike should be at W3→W4 (index 2)
        spike_rows = results[results["is_shift"]]
        assert "W3" in spike_rows["period_a"].values

    def test_detects_multiple_spikes(self, multi_spike_distances):
        config = DetectionConfig(method="zscore", zscore_threshold=1.5)
        sensitive_detector = NarrativeShiftDetector(config)
        results = sensitive_detector.detect(multi_spike_distances)
        n_shifts = results["is_shift"].sum()
        assert n_shifts >= 2, f"Should detect at least 2 spikes, got {n_shifts}"

    def test_z_score_positive_for_spikes(self, detector, spike_distances):
        results = detector.detect(spike_distances)
        spike_rows = results[results["is_shift"]]
        assert (spike_rows["z_score"] > 0).all(), "Z-scores should be positive for spikes"


class TestIQRDetection:
    def test_detects_outlier(self, spike_distances):
        config = DetectionConfig(method="iqr")
        detector = NarrativeShiftDetector(config)
        results = detector.detect(spike_distances)
        assert results["is_shift"].any(), "IQR should detect the outlier"


class TestThresholdDetection:
    def test_detects_above_threshold(self, spike_distances):
        config = DetectionConfig(method="threshold", fixed_threshold=0.30)
        detector = NarrativeShiftDetector(config)
        results = detector.detect(spike_distances)

        above = results[results["distance"] > 0.30]
        assert results["is_shift"].any(), "Should detect values above threshold"
        assert len(results[results["is_shift"]]) == len(above)

    def test_no_detection_with_high_threshold(self, spike_distances):
        config = DetectionConfig(method="threshold", fixed_threshold=1.0)
        detector = NarrativeShiftDetector(config)
        results = detector.detect(spike_distances)
        assert not results["is_shift"].any(), "Nothing should exceed threshold of 1.0"


class TestMagnitudeClassification:
    def test_magnitude_categories(self, detector, spike_distances):
        results = detector.detect(spike_distances)
        valid_magnitudes = {"none", "minor", "moderate", "major"}
        assert set(results["shift_magnitude"].unique()).issubset(valid_magnitudes)

    def test_major_shift_gets_major_label(self, detector, spike_distances):
        results = detector.detect(spike_distances)
        spike_rows = results[results["is_shift"]]
        # The spike is very large compared to baseline, should be major
        if not spike_rows.empty:
            magnitudes = spike_rows["shift_magnitude"].values
            assert "major" in magnitudes or "moderate" in magnitudes


class TestOutputSchema:
    def test_has_required_columns(self, detector, flat_distances):
        results = detector.detect(flat_distances)
        required = ["is_shift", "z_score", "shift_magnitude", "threshold"]
        for col in required:
            assert col in results.columns, f"Missing column: {col}"


class TestSummary:
    def test_summary_structure(self, detector, spike_distances):
        results = detector.detect(spike_distances)
        summary = detector.summary(results)
        assert "total_transitions" in summary
        assert "total_shifts" in summary
        assert "distance_mean" in summary
        assert "shifts_by_magnitude" in summary

    def test_summary_counts(self, detector, flat_distances):
        results = detector.detect(flat_distances)
        summary = detector.summary(results)
        assert summary["total_transitions"] == len(flat_distances)
        assert summary["total_shifts"] == 0


class TestEdgeCases:
    def test_two_data_points(self, detector):
        df = pd.DataFrame({
            "period_a": ["W1"],
            "period_b": ["W2"],
            "distance": [0.5],
        })
        results = detector.detect(df)
        assert len(results) == 1

    def test_handles_zero_std(self, detector):
        """All distances identical — std = 0."""
        df = pd.DataFrame({
            "period_a": ["W1", "W2", "W3"],
            "period_b": ["W2", "W3", "W4"],
            "distance": [0.1, 0.1, 0.1],
        })
        results = detector.detect(df)
        assert not results["is_shift"].any()
