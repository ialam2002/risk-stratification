from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET_COLUMN, load_dataset


@dataclass
class TrainingResult:
    metrics: dict[str, Any]
    model_path: Path
    metrics_path: Path


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

    metrics = {
        **_dataset_summary(df, target_column),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "test_prevalence": float(y_test.mean()),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "risk_model.joblib"
    metrics_path = output_path / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingResult(metrics=metrics, model_path=model_path, metrics_path=metrics_path)


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

