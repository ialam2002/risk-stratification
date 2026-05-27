from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix
def _compute_clinical_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {"threshold": float(threshold), "sensitivity": float(sensitivity), "specificity": float(specificity), "ppv": float(ppv), "npv": float(npv), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
def _optimize_for_use_case(y_true, y_prob, use_case):
    thresholds = np.linspace(0.01, 0.99, 99)
    evaluations = [_compute_clinical_metrics_at_threshold(y_true, y_prob, float(t)) for t in thresholds]
    if use_case == "high_sensitivity":
        best = max(evaluations, key=lambda x: x["sensitivity"])
        constraint = "Minimize false negatives"
    elif use_case == "high_specificity":
        best = max(evaluations, key=lambda x: x["specificity"])
        constraint = "Minimize false positives"
    else:
        best = max(evaluations, key=lambda x: x["sensitivity"] + x["specificity"])
        constraint = "Balance sensitivity/specificity"
    return {"use_case": use_case, "constraint": constraint, "recommended_threshold": best, "all_evaluations": evaluations}
def clinical_threshold_analysis(y_true, y_prob):
    return {use_case: _optimize_for_use_case(y_true, y_prob, use_case) for use_case in ["high_sensitivity", "high_specificity", "balanced"]}
