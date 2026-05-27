from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _compute_feature_drift(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Detect drift using mean and std comparison."""
    drift_report: dict[str, Any] = {
        "threshold": threshold,
        "features_with_drift": [],
        "drift_metrics": {},
    }

    for col in numeric_cols:
        if col not in train_frame.columns or col not in test_frame.columns:
            continue

        train_mean = float(train_frame[col].mean())
        test_mean = float(test_frame[col].mean())
        train_std = float(train_frame[col].std())

        if train_std > 0:
            normalized_drift = abs(test_mean - train_mean) / train_std
        else:
            normalized_drift = 0.0

        drift_report["drift_metrics"][col] = {
            "train_mean": train_mean,
            "test_mean": test_mean,
            "normalized_drift": normalized_drift,
        }

        if normalized_drift > threshold:
            drift_report["features_with_drift"].append(col)

    return drift_report


def _compute_prediction_drift(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    """Detect drift in prediction distributions."""
    return {
        "train_mean_score": float(train_scores.mean()),
        "test_mean_score": float(test_scores.mean()),
        "train_std_score": float(train_scores.std()),
        "test_std_score": float(test_scores.std()),
        "score_drift_ratio": float(test_scores.mean() / max(train_scores.mean(), 0.01)),
    }


def _compute_calibration_drift(
    train_y: np.ndarray,
    train_prob: np.ndarray,
    test_y: np.ndarray,
    test_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Detect calibration drift by comparing bins."""
    metrics = {}

    for split_name, y, prob in [("train", train_y, train_prob), ("test", test_y, test_prob)]:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []

        for i in range(n_bins):
            mask = (prob >= bins[i]) & (prob < bins[i + 1])
            if mask.sum() > 0:
                bin_accs.append(float(y[mask].mean()))
                bin_confs.append(float(prob[mask].mean()))

        if bin_accs:
            metrics[f"{split_name}_calibration_error"] = float(
                np.mean(np.abs(np.array(bin_accs) - np.array(bin_confs)))
            )

    return metrics


def compute_monitoring_report(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    train_prob: np.ndarray,
    test_frame: pd.DataFrame,
    test_y: np.ndarray,
    test_prob: np.ndarray,
    numeric_cols: list[str],
) -> dict[str, Any]:
    """Generate comprehensive monitoring report."""
    return {
        "feature_drift": _compute_feature_drift(train_frame, test_frame, numeric_cols),
        "prediction_drift": _compute_prediction_drift(train_prob, test_prob),
        "calibration_drift": _compute_calibration_drift(train_y, train_prob, test_y, test_prob),
    }

