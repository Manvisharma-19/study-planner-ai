import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def add_urgency_score(df):
    df = df.copy()
    df["urgency_score"] = df["difficulty"] / (df["days_left"] + 1)
    max_u = df["urgency_score"].max()
    df["urgency_score"] = df["urgency_score"] / max_u if max_u > 0 else 0
    return df


def add_performance_gap(df, target_score=75.0):
    df = df.copy()
    df["performance_gap"] = target_score - df["past_score"]
    return df


def add_study_efficiency(df):
    df = df.copy()
    numerator   = df["past_score"] * df["consistency"]
    denominator = df["difficulty"] * df["study_hrs_day"] + 1
    df["study_efficiency"] = numerator / denominator
    mn, mx = df["study_efficiency"].min(), df["study_efficiency"].max()
    df["study_efficiency"] = (df["study_efficiency"] - mn) / (mx - mn + 1e-9)
    return df


def add_completion_rate(df):
    df = df.copy()
    df["completion_rate"]         = df["completion_pct"] / 100
    df["remaining_topics_ratio"]  = 1 - df["completion_rate"]
    return df


def add_risk_label(df, pass_mark=50.0, target_col="expected_score"):
    df = df.copy()
    if target_col in df.columns:
        df["at_risk"] = (df[target_col] < pass_mark).astype(int)
    return df


def add_interaction_features(df):
    df = df.copy()
    df["productive_hours"]    = df["consistency"] * df["study_hrs_day"]
    df["hard_work_remaining"] = df["difficulty"]  * df["remaining_topics_ratio"]
    if "urgency_score" in df.columns and "performance_gap" in df.columns:
        gap_clipped = df["performance_gap"].clip(lower=0)
        df["critical_deficit"] = df["urgency_score"] * gap_clipped
    return df


def engineer_features(df, target_score=75.0, pass_mark=50.0):
    df = add_urgency_score(df)
    df = add_performance_gap(df, target_score=target_score)
    df = add_study_efficiency(df)
    df = add_completion_rate(df)
    df = add_risk_label(df, pass_mark=pass_mark)
    df = add_interaction_features(df)
    print("Feature engineering complete. Shape:", df.shape)
    return df


FEATURE_COLS = [
    "study_hrs_day",
    "consistency",
    "difficulty",
    "days_left",
    "past_score",
    "completion_pct",
    "urgency_score",
    "performance_gap",
    "study_efficiency",
    "completion_rate",
    "remaining_topics_ratio",
    "productive_hours",
    "hard_work_remaining",
]

TARGET_REGRESSION     = "expected_score"
TARGET_CLASSIFICATION = "at_risk"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import run_pipeline
    out = run_pipeline()
    df  = engineer_features(out["clean_df"])
    print(df[FEATURE_COLS].head())