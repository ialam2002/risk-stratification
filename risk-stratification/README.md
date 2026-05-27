# Healthcare Readmission Risk Stratification

This repository contains a tabular machine learning pipeline for predicting short-term hospital readmission risk using the **UCI Diabetes 130-US Hospitals** dataset. The project includes model training, evaluation, explainability, fairness reporting, benchmark comparisons, and a small FastAPI service for inference.

## What the project covers

- Real healthcare dataset ingestion through `ucimlrepo`
- Binary readmission target mapping (`<30` vs. `>30` / `NO`)
- Training pipeline with preprocessing, calibration reporting, and threshold analysis
- SHAP-based explainability outputs
- Subgroup fairness reporting for age, gender, and race
- Benchmark comparison against tree-based baselines
- FastAPI inference endpoints
- Container and CI configuration for local deployment workflows

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Diabetes 130-US hospitals for years 1999-2008
- **Identifier:** `296`
- **Prediction target:** readmission within 30 days

Target mapping used in this project:

- `<30` -> `1`
- `>30` -> `0`
- `NO` -> `0`

## Repository layout

```text
src/risk_strat/
  api.py                  FastAPI inference service
  baselines.py            Baseline model comparison
  benchmark.py            Reproducible benchmark run
  cli.py                  Command-line entrypoint
  clinical_analysis.py    Threshold analysis utilities
  compliance.py           Governance and deployment documentation helpers
  data.py                 Data loading, cleaning, and split utilities
  etl.py                  Example ETL and Spark-oriented helpers
  explain.py              SHAP reporting utilities
  fairness.py             Subgroup evaluation utilities
  model.py                Training pipeline and artifact generation
  monitoring.py           Drift and monitoring helpers

tests/
  test_pipeline.py        End-to-end smoke tests

artifacts/                Generated during training or benchmarking
```

## Setup

The commands below are written for PowerShell on Windows.

```powershell
cd "C:\Users\Iftekhar Alam\PycharmProjects\Health-DS\risk-stratification"
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Train the model

Run the default training pipeline:

```powershell
python -m risk_strat
```

Train from a local CSV instead of fetching the UCI dataset:

```powershell
python -m risk_strat --input-csv "data\diabetes.csv"
```

The CSV must contain either `readmitted_binary` or the raw `readmitted` column.

## Run the benchmark comparison

The benchmark module trains additional baselines and writes a summary to `artifacts\benchmark_metrics.json`.

```powershell
python -m risk_strat.benchmark
```

## Run the API

After training once, start the inference API:

```powershell
uvicorn risk_strat.api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `POST /explain`

Example request body:

```json
{
  "features": {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "time_in_hospital": 3,
    "num_medications": 10
  }
}
```

## Run tests

```powershell
python -m pytest -q
```

## Outputs

The main training run writes these artifacts under `artifacts/`:

- `risk_model.joblib`
- `metrics.json`
- `fairness.json`
- `calibration.json`
- `thresholds.json`
- `shap_report.json`
- `shap_background.csv`
- `model_card.md`

The benchmark run also writes:

- `benchmark_metrics.json`

## Results summary

### Main training run

Held-out test set (`n=20,354`):

- **ROC-AUC:** `0.6521`
- **Average Precision:** `0.2028`
- **Brier Score:** `0.2298`
- **Best F1 threshold:** `0.55`

Top features from the SHAP summary:

- `number_inpatient`
- `payer_code_missing`
- `discharge_disposition_id`
- `age_[70-80)`

### Benchmark comparison

Validation split (`n=15,265`):

| Model | ROC-AUC | Average Precision | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.6447 | 0.2010 | 0.2308 |
| XGBoost | **0.6710** | 0.2249 | **0.2167** |
| LightGBM | 0.6702 | **0.2260** | 0.2170 |

Additional details are available in `RESULTS.md` and the JSON artifacts under `artifacts/`.

## Notes and limitations

- The project uses de-identified public data.
- The current default training path uses a stratified train/test split.
- The `temporal_split` utility supports chronological splitting when a date-like column is available, but the public UCI dataset used here does not provide a production-ready event timestamp in the current pipeline.
- This repository is intended for portfolio, experimentation, and engineering demonstration purposes rather than clinical use.

