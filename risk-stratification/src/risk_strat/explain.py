from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
def _require_shap():
    import shap
    return shap
def _to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
def _transform_for_shap(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = model.named_steps['preprocessor']
    return _to_dense(preprocessor.transform(frame)), list(preprocessor.get_feature_names_out())
def build_shap_report(model: Any, background_frame: pd.DataFrame, sample_frame: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    shap = _require_shap()
    classifier = model.named_steps['classifier']
    background_matrix, feature_names = _transform_for_shap(model, background_frame)
    sample_matrix, _ = _transform_for_shap(model, sample_frame)
    explainer = shap.LinearExplainer(classifier, background_matrix, feature_names=feature_names)
    explanation = explainer(sample_matrix)
    values = np.asarray(explanation.values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mean_abs_shap = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    return {
        'background_size': int(background_frame.shape[0]),
        'sample_size': int(sample_frame.shape[0]),
        'expected_value': float(np.asarray(explanation.base_values).mean()),
        'top_features': [{'feature': feature_names[i], 'mean_abs_shap': float(mean_abs_shap[i])} for i in top_indices],
    }
def explain_instance(model: Any, background_frame: pd.DataFrame, instance_frame: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    shap = _require_shap()
    classifier = model.named_steps['classifier']
    background_matrix, feature_names = _transform_for_shap(model, background_frame)
    instance_matrix, _ = _transform_for_shap(model, instance_frame)
    explainer = shap.LinearExplainer(classifier, background_matrix, feature_names=feature_names)
    explanation = explainer(instance_matrix)
    values = np.asarray(explanation.values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    contributions = values[0]
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_n]
    return {
        'expected_value': float(np.asarray(explanation.base_values).reshape(-1)[0]),
        'top_contributions': [{'feature': feature_names[i], 'shap_value': float(contributions[i])} for i in top_indices],
    }
