# Results Snapshot

This file captures the headline metrics that are most useful in a portfolio, resume, or interview discussion.

## Primary training run

The default pipeline trains an interpretable logistic regression model on the UCI Diabetes 130-US Hospitals dataset.

**Held-out test set (`n=20,354`)**
- ROC-AUC: **0.6521**
- Average Precision: **0.2028**
- Brier Score: **0.2298**
- Positive class prevalence: **11.16%**

## Operating points

These thresholds are useful when discussing how model behavior changes based on clinical priorities.

| Strategy | Threshold | Precision | Recall / Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.1706 | 0.5561 | 0.6604 | 0.2611 |
| Best F1 | 0.55 | 0.1949 | 0.4100 | 0.7874 | 0.2642 |
| High specificity leaning | 0.60 | 0.2216 | 0.2823 | 0.8755 | 0.2483 |
| High sensitivity leaning | 0.35 | 0.1225 | 0.9313 | 0.1620 | 0.2165 |

## Fairness snapshot

### Gender
- ROC-AUC gap: **0.0041**
- Average precision gap: **0.0143**
- Brier score gap: **0.0006**

### Age
- ROC-AUC gap: **0.1270**
- Average precision gap: **0.0787**
- Brier score gap: **0.0520**

### Race
- ROC-AUC gap: **0.0118**
- Average precision gap: **0.0607**
- Brier score gap: **0.0557**

Interpretation: subgroup performance is fairly stable by gender, while age shows the largest variation and is the slice worth discussing first in interviews.

## Explainability

Top global drivers from the SHAP summary:
1. `number_inpatient`
2. `payer_code_missing`
3. `discharge_disposition_id`
4. `age_[70-80)`
5. `medical_specialty_missing`

These features are useful talking points because they connect to utilization history, discharge planning, and patient complexity.

## Benchmark pass

A second benchmark run used a 70/15/15 train/validation/test split to compare stronger baselines.

### Validation split model comparison (`n=15,265`)

| Model | ROC-AUC | Average Precision | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.6447 | 0.2010 | 0.2308 |
| XGBoost | **0.6710** | 0.2249 | **0.2167** |
| LightGBM | 0.6702 | **0.2260** | 0.2170 |

### Logistic regression test check (`n=15,265`)
- ROC-AUC: **0.6390**
- 95% bootstrap CI: **0.6251 to 0.6529**
- Average Precision: **0.1873**
- Brier Score: **0.2307**

## Suggested interview framing

A concise way to present the results:

> I built a readmission risk pipeline on the UCI Diabetes 130-US Hospitals dataset with about 102k encounters. The interpretable logistic baseline reached a held-out ROC-AUC of 0.65 and average precision of 0.20. I added threshold analysis, subgroup fairness checks, calibration reporting, and SHAP explanations. In a separate benchmark pass, XGBoost and LightGBM improved validation ROC-AUC to about 0.67, which gave me a good tradeoff discussion around interpretability versus predictive lift.

## Source artifacts

- `artifacts/metrics.json`
- `artifacts/fairness.json`
- `artifacts/thresholds.json`
- `artifacts/shap_report.json`
- `artifacts/benchmark_metrics.json`

