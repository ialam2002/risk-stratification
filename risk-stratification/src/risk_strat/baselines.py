from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
def compare_baselines(X_train, y_train, X_val, y_val):
    results = {}
    lr = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_val)[:, 1]
    results["logistic_regression"] = {"name": "logistic_regression", "roc_auc": float(roc_auc_score(y_val, y_prob)), "average_precision": float(average_precision_score(y_val, y_prob)), "brier_score": float(brier_score_loss(y_val, y_prob))}
    if HAS_XGB:
        try:
            xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()))
            xgb_model.fit(X_train, y_train)
            y_prob = xgb_model.predict_proba(X_val)[:, 1]
            results["xgboost"] = {"name": "xgboost", "roc_auc": float(roc_auc_score(y_val, y_prob)), "average_precision": float(average_precision_score(y_val, y_prob)), "brier_score": float(brier_score_loss(y_val, y_prob))}
        except Exception as e:
            results["xgboost"] = {"error": str(e)}
    if HAS_LGB:
        try:
            lgb_model = lgb.LGBMClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()))
            lgb_model.fit(X_train, y_train)
            y_prob = lgb_model.predict_proba(X_val)[:, 1]
            results["lightgbm"] = {"name": "lightgbm", "roc_auc": float(roc_auc_score(y_val, y_prob)), "average_precision": float(average_precision_score(y_val, y_prob)), "brier_score": float(brier_score_loss(y_val, y_prob))}
        except Exception as e:
            results["lightgbm"] = {"error": str(e)}
    return results
