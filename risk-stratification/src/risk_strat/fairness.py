from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(metric_fn(y_true, y_score))


def _group_summary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predicted = (y_score >= threshold).astype(int)
    return {
        "n": int(y_true.size),
        "positive_rate": float(np.mean(y_true)),
        "mean_score": float(np.mean(y_score)),
        "predicted_positive_rate": float(np.mean(predicted)),
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": _safe_metric(average_precision_score, y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def _disparity_for_metric(
    groups: list[dict[str, Any]],
    metric_name: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any] | None:
    metric_values = [
        (group["group"], float(group[metric_name]))
        for group in groups
        if group.get(metric_name) is not None
    ]
    if len(metric_values) < 2:
        return None

    sorted_values = sorted(metric_values, key=lambda item: item[1])
    min_group, min_value = sorted_values[0]
    max_group, max_value = sorted_values[-1]
    gap = max_value - min_value

    favored_group = min_group if lower_is_better else max_group
    harmed_group = max_group if lower_is_better else min_group

    return {
        "gap": float(gap),
        "favored_group": favored_group,
        "harmed_group": harmed_group,
        "min_value": float(min_value),
        "max_value": float(max_value),
    }


def _compute_slice_disparities(groups: list[dict[str, Any]]) -> dict[str, Any]:
    metric_config = {
        "positive_rate": {"lower_is_better": False},
        "mean_score": {"lower_is_better": False},
        "predicted_positive_rate": {"lower_is_better": False},
        "roc_auc": {"lower_is_better": False},
        "average_precision": {"lower_is_better": False},
        "brier_score": {"lower_is_better": True},
    }
    disparities: dict[str, Any] = {}

    for metric_name, config in metric_config.items():
        disparity = _disparity_for_metric(
            groups,
            metric_name,
            lower_is_better=bool(config["lower_is_better"]),
        )
        if disparity is not None:
            disparities[metric_name] = disparity

    return disparities


def compute_fairness_report(
    y_true: pd.Series,
    y_score: np.ndarray,
    group_frame: pd.DataFrame,
    min_group_size: int = 100,
) -> dict[str, Any]:
    """Compute subgroup metrics for clinically meaningful slices."""
    report: dict[str, Any] = {
        "min_group_size": int(min_group_size),
        "slices": {},
        "disparities": {},
    }

    aligned_y = pd.Series(y_true).reset_index(drop=True)
    aligned_score = np.asarray(y_score)
    aligned_groups = group_frame.reset_index(drop=True).copy()

    for column in aligned_groups.columns:
        series = aligned_groups[column].astype("string").fillna("missing")
        column_report: list[dict[str, Any]] = []

        for group_name in sorted(series.unique().tolist()):
            mask = series == group_name
            group_size = int(mask.sum())
            if group_size < min_group_size:
                continue

            group_y = aligned_y.loc[mask].astype(int).to_numpy()
            group_score = aligned_score[mask.to_numpy()]
            column_report.append(
                {
                    "group": group_name,
                    **_group_summary(group_y, group_score),
                }
            )

        if column_report:
            sorted_report = sorted(column_report, key=lambda item: item["n"], reverse=True)
            report["slices"][column] = sorted_report
            report["disparities"][column] = _compute_slice_disparities(sorted_report)

    return report

