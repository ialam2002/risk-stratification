from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

UCI_DIABETES_DATASET_ID = 296
RAW_TARGET_COLUMN = "readmitted"
TARGET_COLUMN = "readmitted_binary"
IDENTIFIER_COLUMNS = {"encounter_id", "patient_nbr"}


def _fetch_ucirepo() -> Any:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - exercised in environments missing dependency
        raise ImportError(
            "ucimlrepo is required to fetch the UCI diabetes dataset. "
            "Install it with `python -m pip install -r requirements.txt`."
        ) from exc
    return fetch_ucirepo


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].map(
                lambda value: np.nan
                if value in {"?", ""}
                else value.strip()
                if isinstance(value, str)
                else value
            )

    return cleaned


def _coerce_readmission_target(values: pd.Series) -> pd.Series:
    series = values.copy()
    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(pd.Series(series).dropna().astype(int).unique().tolist())
        if unique_values <= {0, 1}:
            return series.astype(int)

    normalized = series.astype("string").str.strip().str.upper()
    mapped = normalized.map({"<30": 1, ">30": 0, "NO": 0, "0": 0, "1": 1})

    if mapped.isna().any():
        unknown_values = sorted(normalized[mapped.isna()].dropna().unique().tolist())
        raise ValueError(
            "Unrecognized readmission labels found in target column. "
            f"Expected values like '<30', '>30', or 'NO'. Got: {unknown_values}"
        )

    return mapped.astype(int)


def prepare_diabetes_dataframe(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    features_df = _clean_text_columns(features)
    targets_df = _clean_text_columns(targets)

    if targets_df.shape[1] != 1:
        if RAW_TARGET_COLUMN not in targets_df.columns:
            raise ValueError(
                "The UCI diabetes target data must contain a single target column "
                f"or a '{RAW_TARGET_COLUMN}' column."
            )
        target_series = targets_df[RAW_TARGET_COLUMN]
    else:
        target_series = targets_df.iloc[:, 0]

    prepared = features_df.copy()
    for column in IDENTIFIER_COLUMNS:
        if column in prepared.columns:
            prepared = prepared.drop(columns=[column])

    if RAW_TARGET_COLUMN in prepared.columns:
        prepared = prepared.drop(columns=[RAW_TARGET_COLUMN])

    prepared[TARGET_COLUMN] = _coerce_readmission_target(target_series)
    return prepared


def load_dataset(input_csv: str | None = None) -> pd.DataFrame:
    if input_csv is not None:
        csv_path = Path(input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        df = _clean_text_columns(pd.read_csv(csv_path))
        if TARGET_COLUMN not in df.columns:
            if RAW_TARGET_COLUMN in df.columns:
                df[TARGET_COLUMN] = _coerce_readmission_target(df[RAW_TARGET_COLUMN])
            else:
                raise ValueError(
                    f"CSV dataset must include either '{TARGET_COLUMN}' or '{RAW_TARGET_COLUMN}'."
                )
        return df

    fetch_ucirepo = _fetch_ucirepo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
        dataset = fetch_ucirepo(id=UCI_DIABETES_DATASET_ID)
    return prepare_diabetes_dataframe(dataset.data.features, dataset.data.targets)


def temporal_split(
    df: pd.DataFrame,
    date_column: str | None = None,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset temporally if date_column is provided, else random split."""
    if train_ratio + test_ratio + val_ratio != 1.0:
        raise ValueError("train_ratio + test_ratio + val_ratio must equal 1.0")

    if date_column and date_column in df.columns:
        df_sorted = df.sort_values(by=date_column).reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = df_sorted.iloc[:train_end]
        val = df_sorted.iloc[train_end:val_end]
        test = df_sorted.iloc[val_end:]
    else:
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = df_shuffled.iloc[:train_end]
        val = df_shuffled.iloc[train_end:val_end]
        test = df_shuffled.iloc[val_end:]

    return train, val, test


