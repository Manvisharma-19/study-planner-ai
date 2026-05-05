import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def generate_sample_data(n_students=200, seed=42):
    np.random.seed(seed)
    subjects = ["Mathematics", "Physics", "Chemistry", "Biology",
                "Computer Science", "English", "History"]
    difficulty = {"Mathematics": 0.85, "Physics": 0.80, "Chemistry": 0.75,
                  "Biology": 0.60, "Computer Science": 0.70,
                  "English": 0.45, "History": 0.50}
    records = []
    for student_id in range(1, n_students + 1):
        n_subjects = np.random.randint(3, 6)
        chosen = np.random.choice(subjects, n_subjects, replace=False)
        for subj in chosen:
            days_left    = np.random.randint(7, 60)
            study_hrs    = np.random.uniform(0.5, 6.0)
            consistency  = np.random.uniform(0.3, 1.0)
            past_score   = float(np.clip(np.random.normal(60, 15), 20, 100))
            completion   = np.random.uniform(10, 100)
            base = (past_score * 0.5 + study_hrs * 4 + consistency * 10
                    - difficulty[subj] * 15 + (completion / 100) * 8
                    + (days_left / 60) * 5)
            expected = float(np.clip(base + np.random.normal(0, 5), 20, 100))
            records.append({
                "student_id":     student_id,
                "subject":        subj,
                "difficulty":     difficulty[subj],
                "days_left":      days_left,
                "study_hrs_day":  round(study_hrs, 2),
                "consistency":    round(consistency, 2),
                "past_score":     round(past_score, 1),
                "completion_pct": round(completion, 1),
                "expected_score": round(expected, 1),
            })
    return pd.DataFrame(records)


def load_data(filepath=None):
    if filepath:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
    else:
        df = generate_sample_data()
        print(f"Generated {len(df)} synthetic rows")
    return df


def clean_data(df):
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["study_hrs_day"]  = df["study_hrs_day"].clip(0, 16)
    df["consistency"]    = df["consistency"].clip(0, 1)
    df["past_score"]     = df["past_score"].clip(0, 100)
    df["completion_pct"] = df["completion_pct"].clip(0, 100)
    df["difficulty"]     = df["difficulty"].clip(0, 1)
    df["days_left"]      = df["days_left"].clip(1, 365)
    if "expected_score" in df.columns:
        df["expected_score"] = df["expected_score"].clip(0, 100)
    return df.reset_index(drop=True)


def encode_and_normalize(df, scaler=None, encoder=None, fit=True):
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
    df["subject_enc"] = encoder.fit_transform(df["subject"]) if fit else encoder.transform(df["subject"])
    scale_cols = ["study_hrs_day", "days_left", "past_score",
                  "completion_pct", "difficulty", "consistency"]
    if scaler is None:
        scaler = MinMaxScaler()
    if fit:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])
    return df, scaler, encoder


def run_pipeline(filepath=None):
    raw_df   = load_data(filepath)
    clean_df = clean_data(raw_df.copy())
    proc_df, scaler, encoder = encode_and_normalize(clean_df)
    print(f"Pipeline complete. Shape: {proc_df.shape}")
    return {"raw_df": raw_df, "clean_df": clean_df,
            "processed_df": proc_df, "scaler": scaler, "encoder": encoder}


if __name__ == "__main__":
    result = run_pipeline()
    print(result["processed_df"].head())