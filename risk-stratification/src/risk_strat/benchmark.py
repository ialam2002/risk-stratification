from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .baselines import compare_baselines
from .data import TARGET_COLUMN, load_dataset, temporal_split
from .model import _build_pipeline


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)]
    )


def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 300) -> list[float]:
    rng = np.random.default_rng(42)
    values: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        values.append(float(roc_auc_score(sample_y, y_score[idx])))

    if not values:
        return [0.0, 0.0]

    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def generate_benchmark_metrics(output_dir: str = "artifacts") -> dict[str, Any]:
    """Run a benchmark pass with train/validation/test splits and model comparison."""
    df = load_dataset()
    train_df, val_df, test_df = temporal_split(df)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = _build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    baseline_results = compare_baselines(
        X_train_transformed,
        y_train.to_numpy(),
        X_val_transformed,
        y_val.to_numpy(),
    )

    logistic_pipeline = _build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    logistic_pipeline.fit(X_train, y_train)
    y_test_prob = logistic_pipeline.predict_proba(X_test)[:, 1]

    benchmark = {
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "baseline_validation": baseline_results,
        "logistic_test": {
            "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
            "average_precision": float(average_precision_score(y_test, y_test_prob)),
            "brier_score": float(brier_score_loss(y_test, y_test_prob)),
            "roc_auc_ci_95": _bootstrap_auc_ci(y_test.to_numpy(), y_test_prob),
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "benchmark_metrics.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    return benchmark


if __name__ == "__main__":
    print(json.dumps(generate_benchmark_metrics(), indent=2))

