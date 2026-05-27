from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET_COLUMN, load_dataset
from .explain import build_shap_report
from .fairness import compute_fairness_report


@dataclass
class TrainingResult:
    metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path
    fairness_path: Path
    shap_report_path: Path
    shap_background_path: Path
    calibration_path: Path
    threshold_report_path: Path
    model_card_path: Path


def _build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )


def _validate_binary_target(y: pd.Series, target_column: str) -> None:
    values = sorted(pd.Series(y).dropna().unique().tolist())
    if set(values) - {0, 1}:
        raise ValueError(
            f"Target column '{target_column}' must be binary (0/1). "
            f"Found values: {values}"
        )


def _dataset_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    return {
        "dataset": "uci_diabetes_130_us_hospitals",
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "positive_rate": float(df[target_column].mean()),
    }


def _sample_frame(frame: pd.DataFrame, size: int, random_state: int) -> pd.DataFrame:
    sample_size = min(size, len(frame))
    if sample_size <= 0:
        raise ValueError("Cannot sample from an empty frame.")
    return frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _selected_group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [column for column in ["age", "gender", "race"] if column in frame.columns]
    return frame[candidate_columns].copy() if candidate_columns else pd.DataFrame(index=frame.index)


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _build_calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    points = [
        {
            "mean_predicted_probability": float(pred),
            "observed_positive_rate": float(obs),
        }
        for pred, obs in zip(mean_pred, frac_pos)
    ]
    calibration_error = float(np.mean(np.abs(np.asarray(mean_pred) - np.asarray(frac_pos)))) if points else 0.0
    return {
        "n_bins": int(n_bins),
        "mean_absolute_calibration_error": calibration_error,
        "points": points,
    }


def _build_threshold_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    evaluations = [_metrics_at_threshold(y_true, y_prob, float(threshold)) for threshold in candidate_thresholds]

    best_by_f1 = max(evaluations, key=lambda item: item["f1"])
    best_by_youden = max(evaluations, key=lambda item: item["sensitivity"] + item["specificity"] - 1.0)

    return {
        "default_threshold": _metrics_at_threshold(y_true, y_prob, threshold=0.5),
        "best_f1_threshold": best_by_f1,
        "best_youden_threshold": best_by_youden,
        "evaluations": evaluations,
    }


def _build_model_card(metrics: dict[str, Any]) -> str:
    fairness_slices = metrics.get("fairness", {}).get("slices", {})
    fairness_disparities = metrics.get("fairness", {}).get("disparities", {})
    shap_features = metrics.get("shap", {}).get("top_features", [])
    threshold = metrics.get("thresholds", {}).get("best_f1_threshold", {}).get("threshold", 0.5)

    top_feature_lines = [
        f"- `{item['feature']}`: {item['mean_abs_shap']:.4f}"
        for item in shap_features[:5]
    ]
    if not top_feature_lines:
        top_feature_lines = ["- Not available"]

    disparity_lines: list[str] = []
    for slice_name, slice_metrics in fairness_disparities.items():
        score_gap = slice_metrics.get("mean_score", {}).get("gap")
        positive_gap = slice_metrics.get("positive_rate", {}).get("gap")
        if score_gap is not None or positive_gap is not None:
            disparity_lines.append(
                f"- `{slice_name}` score_gap={score_gap if score_gap is not None else 0.0:.4f}, "
                f"positive_rate_gap={positive_gap if positive_gap is not None else 0.0:.4f}"
            )
    if not disparity_lines:
        disparity_lines = ["- No subgroup disparities were computed (insufficient group sizes)."]

    return "\n".join(
        [
            "# Model Card: UCI Diabetes Readmission Risk",
            "",
            "## Intended use",
            "- Educational portfolio project for readmission risk stratification benchmarking.",
            "- Not for clinical decision making.",
            "",
            "## Dataset",
            f"- Source: UCI Diabetes 130-US hospitals (n={metrics.get('n_rows', 'unknown')}).",
            "- Target mapping: `<30` to 1, `>30`/`NO` to 0.",
            "",
            "## Performance",
            f"- ROC AUC: {metrics.get('roc_auc', 0.0):.4f}",
            f"- Average precision: {metrics.get('average_precision', 0.0):.4f}",
            f"- Brier score: {metrics.get('brier_score', 0.0):.4f}",
            f"- Suggested threshold (best F1): {threshold:.2f}",
            "",
            "## Explainability (SHAP top features)",
            *top_feature_lines,
            "",
            "## Fairness snapshot",
            f"- Slices evaluated: {', '.join(sorted(fairness_slices.keys())) if fairness_slices else 'none'}",
            *disparity_lines,
            "",
            "## Safety notes",
            "- Use only de-identified data.",
            "- Validate calibration, fairness, and temporal drift before any deployment.",
            "- Keep human oversight for any clinical workflow integration.",
            "",
        ]
    )


def run_training(
    input_csv: str | None = None,
    target_column: str = TARGET_COLUMN,
    output_dir: str = "artifacts",
    random_state: int = 42,
) -> TrainingResult:
    df = load_dataset(input_csv=input_csv)

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    y = df[target_column].astype(int)
    _validate_binary_target(y=y, target_column=target_column)
    X = df.drop(columns=[target_column])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found after dropping target.")

    model = _build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_np = y_test.to_numpy()

    fairness_report = compute_fairness_report(
        y_true=y_test,
        y_score=y_prob,
        group_frame=_selected_group_frame(X_test),
        min_group_size=max(50, int(len(X_test) * 0.02)),
    )

    calibration_report = _build_calibration_report(y_true=y_test_np, y_prob=y_prob)
    threshold_report = _build_threshold_report(y_true=y_test_np, y_prob=y_prob)

    shap_background = _sample_frame(X_train, size=200, random_state=random_state)
    shap_sample = _sample_frame(X_test, size=100, random_state=random_state)
    shap_report = build_shap_report(model=model, background_frame=shap_background, sample_frame=shap_sample)

    metrics = {
        **_dataset_summary(df, target_column),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "test_prevalence": float(y_test.mean()),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "fairness": fairness_report,
        "calibration": calibration_report,
        "thresholds": threshold_report,
        "shap": shap_report,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "risk_model.joblib"
    metrics_path = output_path / "metrics.json"
    fairness_path = output_path / "fairness.json"
    shap_report_path = output_path / "shap_report.json"
    shap_background_path = output_path / "shap_background.csv"
    calibration_path = output_path / "calibration.json"
    threshold_report_path = output_path / "thresholds.json"
    model_card_path = output_path / "model_card.md"

    metrics["fairness_path"] = str(fairness_path)
    metrics["shap_report_path"] = str(shap_report_path)
    metrics["shap_background_path"] = str(shap_background_path)
    metrics["calibration_path"] = str(calibration_path)
    metrics["threshold_report_path"] = str(threshold_report_path)
    metrics["model_card_path"] = str(model_card_path)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fairness_path.write_text(json.dumps(fairness_report, indent=2), encoding="utf-8")
    shap_report_path.write_text(json.dumps(shap_report, indent=2), encoding="utf-8")
    shap_background.to_csv(shap_background_path, index=False)
    calibration_path.write_text(json.dumps(calibration_report, indent=2), encoding="utf-8")
    threshold_report_path.write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")
    model_card_path.write_text(_build_model_card(metrics), encoding="utf-8")

    return TrainingResult(
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        fairness_path=fairness_path,
        shap_report_path=shap_report_path,
        shap_background_path=shap_background_path,
        calibration_path=calibration_path,
        threshold_report_path=threshold_report_path,
        model_card_path=model_card_path,
    )


def format_metrics(metrics: dict[str, Any]) -> str:
    ordered = [
        "dataset",
        "n_rows",
        "n_features",
        "positive_rate",
        "roc_auc",
        "average_precision",
        "brier_score",
        "test_prevalence",
        "n_train",
        "n_test",
        "fairness_path",
        "shap_report_path",
        "calibration_path",
        "threshold_report_path",
        "model_card_path",
    ]
    lines = ["Training complete. Key metrics:"]
    for key in ordered:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
    return "\n".join(lines)

