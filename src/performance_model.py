import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing   import StandardScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "study_hrs_day", "consistency", "difficulty", "days_left",
    "past_score", "completion_pct", "urgency_score", "performance_gap",
    "study_efficiency", "completion_rate", "remaining_topics_ratio",
    "productive_hours", "hard_work_remaining",
]


def prepare_data(df, target, test_size=0.2, seed=42):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[target].values
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_regression(df, save_dir="models"):
    X_tr, X_te, y_tr, y_te = prepare_data(df, "expected_score")
    results = {}

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)
    results["linear_regression"] = {
        "model": lr,
        "MAE":   round(mean_absolute_error(y_te, y_pred_lr), 3),
        "R2":    round(r2_score(y_te, y_pred_lr), 3),
        "y_test": y_te,
        "y_pred": y_pred_lr,
    }
    print(f"LinearRegression  MAE: {results['linear_regression']['MAE']}  R2: {results['linear_regression']['R2']}")

    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    results["random_forest"] = {
        "model": rf,
        "MAE":   round(mean_absolute_error(y_te, y_pred_rf), 3),
        "R2":    round(r2_score(y_te, y_pred_rf), 3),
        "y_test": y_te,
        "y_pred": y_pred_rf,
        "feature_importances": rf.feature_importances_,
    }
    print(f"RandomForest      MAE: {results['random_forest']['MAE']}  R2: {results['random_forest']['R2']}")

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/regression_lr.pkl",  "wb") as f:
        pickle.dump(lr, f)
    with open(f"{save_dir}/regression_rf.pkl",  "wb") as f:
        pickle.dump(rf, f)
    print(f"Models saved to {save_dir}/")
    return results


def train_classification(df, save_dir="models"):
    X_tr, X_te, y_tr, y_te = prepare_data(df, "at_risk")
    results = {}

    logr = LogisticRegression(max_iter=500, random_state=42)
    logr.fit(X_tr, y_tr)
    y_pred_log = logr.predict(X_te)
    results["logistic_regression"] = {
        "model":  logr,
        "report": classification_report(y_te, y_pred_log, output_dict=True),
        "y_test": y_te,
        "y_pred": y_pred_log,
    }
    print("LogisticRegression done")

    rfc = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rfc.fit(X_tr, y_tr)
    y_pred_rfc = rfc.predict(X_te)
    results["random_forest"] = {
        "model":  rfc,
        "report": classification_report(y_te, y_pred_rfc, output_dict=True),
        "y_test": y_te,
        "y_pred": y_pred_rfc,
        "feature_importances": rfc.feature_importances_,
    }
    print("RandomForestClassifier done")

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/classification_rf.pkl", "wb") as f:
        pickle.dump(rfc, f)
    return results


def predict_student(student_row, reg_model=None, clf_model=None, model_dir="models"):
    if reg_model is None:
        with open(f"{model_dir}/regression_rf.pkl", "rb") as f:
            reg_model = pickle.load(f)
    if clf_model is None:
        with open(f"{model_dir}/classification_rf.pkl", "rb") as f:
            clf_model = pickle.load(f)
    available = [c for c in FEATURE_COLS if c in student_row]
    X = np.array([[student_row[c] for c in available]])
    score      = float(np.clip(reg_model.predict(X)[0], 0, 100))
    risk_label = int(clf_model.predict(X)[0])
    confidence = float(clf_model.predict_proba(X)[0][0])
    return {
        "predicted_score": round(score, 1),
        "at_risk":         risk_label,
        "confidence":      round(confidence * 100, 1),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline       import run_pipeline
    from feature_engineering import engineer_features
    pipe  = run_pipeline()
    df_fe = engineer_features(pipe["clean_df"])
    train_regression(df_fe)
    train_classification(df_fe)
    print("All models trained and saved.")