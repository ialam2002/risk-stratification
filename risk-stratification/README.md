# Healthcare Readmission Risk Stratification

This project predicts 30-day readmission risk using the real **UCI Diabetes 130-US hospitals for years 1999-2008** dataset.

The default workflow fetches the dataset directly through `ucimlrepo`, cleans the cohort, maps the target to a binary label, and trains a baseline model.

## Dataset

- **Source:** UCI ML Repository
- **Dataset ID:** `296`
- **Problem:** readmission prediction
- **Binary target definition:**
  - `<30` → 1
  - `>30` → 0
  - `NO` → 0

## What this demonstrates

- End-to-end real-data ML workflow
- Healthcare-style target definition and cohort cleaning
- Model evaluation with `roc_auc`, `average_precision`, and `brier_score`
- Reproducible training pipeline and test harness
- Strong portfolio story for healthcare AI interviews

## Project structure

```text
risk-stratification/
  src/risk_strat/
    __init__.py
    __main__.py
    cli.py
    data.py
    model.py
  tests/
    test_pipeline.py
  artifacts/               # generated at runtime
  requirements.txt
  README.md
```

## Quick start

```powershell
cd "C:\Users\Iftekhar Alam\PycharmProjects\Health-DS\risk-stratification"
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m risk_strat
```

If you want to train from a local CSV export of the same cohort:

```powershell
$env:PYTHONPATH = "src"
python -m risk_strat --input-csv "data\diabetes_uci_export.csv"
```

## Expected CSV format

If you export the cohort yourself, keep either:

- `readmitted_binary` as the binary target, or
- the raw `readmitted` column, which will be converted automatically.

## Run tests

```powershell
cd "C:\Users\Iftekhar Alam\PycharmProjects\Health-DS\risk-stratification"
python -m pytest -q
```

## Suggested next steps for interview strength

1. Add subgroup fairness analysis by age, gender, or race.
2. Add probability calibration and compare Brier score / calibration curves.
3. Add SHAP explanations for top features.
4. Add a FastAPI inference service and audit logging.
5. Later upgrade the same pipeline to MIMIC-IV once access is approved.

## Notes on healthcare data

- Use de-identified data only.
- Document label mapping and missing-value handling.
- Track dataset version, model version, and decision threshold.

