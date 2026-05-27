from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .explain import explain_instance
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / 'artifacts' / 'risk_model.joblib'
DEFAULT_BACKGROUND_PATH = Path(__file__).resolve().parents[2] / 'artifacts' / 'shap_background.csv'
DEFAULT_THRESHOLD = 0.5
class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description='Flat feature map for a single patient record.')
class PredictionResponse(BaseModel):
    risk_score: float
    risk_label: int
    threshold: float
    missing_features: list[str]
    extra_features: list[str]
class ExplainResponse(PredictionResponse):
    expected_value: float
    top_contributions: list[dict[str, Any]]
class ModelService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, background_path: Path = DEFAULT_BACKGROUND_PATH, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.model_path = Path(model_path)
        self.background_path = Path(background_path)
        self.threshold = float(threshold)
        self.model = joblib.load(self.model_path)
        self.expected_columns = list(getattr(self.model, 'feature_names_in_', []))
        self.numeric_columns, self.categorical_columns = self._extract_schema()
        self.background_frame = pd.read_csv(self.background_path) if self.background_path.exists() else None
    def _extract_schema(self) -> tuple[list[str], list[str]]:
        preprocessor = self.model.named_steps['preprocessor']
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for name, _, columns in preprocessor.transformers_:
            if name == 'num':
                numeric_cols = list(columns)
            elif name == 'cat':
                categorical_cols = list(columns)
        return numeric_cols, categorical_cols
    def _align_payload(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
        missing = [column for column in self.expected_columns if column not in payload]
        extra = sorted([column for column in payload if column not in self.expected_columns])
        row: dict[str, Any] = {}
        for column in self.expected_columns:
            value = payload.get(column, np.nan if column in self.numeric_columns else None)
            if column in self.numeric_columns:
                row[column] = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
            else:
                row[column] = value
        frame = pd.DataFrame([row], columns=self.expected_columns)
        return frame, missing, extra
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame, missing, extra = self._align_payload(payload)
        risk_score = float(self.model.predict_proba(frame)[0, 1])
        return {
            'risk_score': risk_score,
            'risk_label': int(risk_score >= self.threshold),
            'threshold': self.threshold,
            'missing_features': missing,
            'extra_features': extra,
        }
    def explain(self, payload: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
        if self.background_frame is None or self.background_frame.empty:
            raise ValueError('SHAP background data is unavailable.')
        frame, missing, extra = self._align_payload(payload)
        prediction = self.predict(payload)
        explanation = explain_instance(self.model, self.background_frame, frame, top_n=top_n)
        return {
            **prediction,
            'expected_value': explanation['expected_value'],
            'top_contributions': explanation['top_contributions'],
        }
def create_app(model_path: Path = DEFAULT_MODEL_PATH, background_path: Path = DEFAULT_BACKGROUND_PATH) -> FastAPI:
    app = FastAPI(title='Healthcare Readmission Risk API', version='1.0.0')
    service = ModelService(model_path=model_path, background_path=background_path)
    app.state.service = service
    @app.get('/health')
    def health() -> dict[str, Any]:
        return {
            'status': 'ok',
            'model_path': str(service.model_path),
            'feature_count': len(service.expected_columns),
            'background_available': service.background_frame is not None,
        }
    @app.post('/predict', response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> dict[str, Any]:
        return service.predict(request.features)
    @app.post('/explain', response_model=ExplainResponse)
    def explain(request: PredictionRequest) -> dict[str, Any]:
        try:
            return service.explain(request.features)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return app


def _placeholder_app() -> FastAPI:
    app = FastAPI(title='Healthcare Readmission Risk API', version='1.0.0')

    @app.get('/health')
    def health() -> dict[str, Any]:
        return {
            'status': 'not_ready',
            'detail': f'Train the model first so {DEFAULT_MODEL_PATH} exists.',
        }

    return app


try:
    app = create_app()
except FileNotFoundError:
    app = _placeholder_app()
