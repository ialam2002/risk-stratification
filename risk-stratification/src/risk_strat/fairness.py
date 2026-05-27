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
            report["slices"][column] = sorted(column_report, key=lambda item: item["n"], reverse=True)

    return report

