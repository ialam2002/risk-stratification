from pathlib import Path
import json

import pandas as pd
from fastapi.testclient import TestClient

from risk_strat.api import create_app
from risk_strat.data import TARGET_COLUMN, prepare_diabetes_dataframe
from risk_strat.model import run_training


def test_prepare_diabetes_dataframe_maps_targets_and_cleans_placeholders() -> None:
    features = pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [11, 12, 13, 14],
            "race": ["Caucasian", "?", "AfricanAmerican", "Hispanic"],
            "gender": ["Female", "Male", "Female", "Male"],
            "age": ["[60-70)", "[70-80)", "[50-60)", "[80-90)"],
            "weight": ["?", "?", "?", "?"],
            "num_medications": [10, 12, 8, 14],
        }
    )
    targets = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})

    prepared = prepare_diabetes_dataframe(features, targets)

    assert TARGET_COLUMN in prepared.columns
    assert "encounter_id" not in prepared.columns
    assert "patient_nbr" not in prepared.columns
    assert prepared[TARGET_COLUMN].tolist() == [1, 0, 0, 1]
    assert pd.isna(prepared.loc[1, "race"])
    assert pd.isna(prepared.loc[0, "weight"])


def test_training_pipeline_smoke(tmp_path: Path, monkeypatch) -> None:
    rows = 60
    synthetic_like = pd.DataFrame(
        {
            "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"] * 10,
            "gender": ["Female", "Male"] * 30,
            "age": ["[60-70)", "[70-80)", "[50-60)", "[80-90)", "[40-50)", "[30-40)"] * 10,
            "time_in_hospital": [3, 4, 2, 5, 6, 7] * 10,
            "num_medications": [10, 12, 8, 14, 9, 11] * 10,
            "num_lab_procedures": [40, 45, 30, 50, 35, 42] * 10,
            "diag_1": ["250.13", "414", "428", "?", "401", "276"] * 10,
            "diag_2": ["276", "250.13", "428", "414", "?", "401"] * 10,
            TARGET_COLUMN: [1, 0, 1, 0, 1, 0] * 10,
        }
    )

    monkeypatch.setattr("risk_strat.model.load_dataset", lambda input_csv=None: synthetic_like)

    out_dir = tmp_path / "artifacts"
    result = run_training(output_dir=str(out_dir), random_state=7)

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert result.fairness_path.exists()
    assert result.shap_report_path.exists()
    assert result.shap_background_path.exists()

    assert result.metrics["dataset"] == "uci_diabetes_130_us_hospitals"
    assert result.metrics["n_rows"] == rows
    assert "fairness" in result.metrics
    assert "shap" in result.metrics
    assert 0.5 <= result.metrics["roc_auc"] <= 1.0
    assert 0.0 <= result.metrics["average_precision"] <= 1.0
    assert 0.0 <= result.metrics["brier_score"] <= 1.0

    app = create_app(model_path=result.model_path, background_path=result.shap_background_path)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["background_available"] is True

    payload = json.loads(synthetic_like.drop(columns=[TARGET_COLUMN]).iloc[0].to_json())
    predict_response = client.post("/predict", json={"features": payload})
    assert predict_response.status_code == 200
    predict_body = predict_response.json()
    assert 0.0 <= predict_body["risk_score"] <= 1.0
    assert predict_body["risk_label"] in {0, 1}

    explain_response = client.post("/explain", json={"features": payload})
    assert explain_response.status_code == 200
    explain_body = explain_response.json()
    assert "top_contributions" in explain_body
    assert len(explain_body["top_contributions"]) > 0

