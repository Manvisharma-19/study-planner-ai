import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.data_pipeline        import run_pipeline
from src.feature_engineering  import engineer_features, FEATURE_COLS
from src.planner              import Subject, StudyPlanner
from src.performance_model    import train_regression, train_classification

st.set_page_config(
    page_title="Smart Study Planner AI",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:800;color:#1a1a2e}
.sub-title{font-size:1rem;color:#666;margin-bottom:1.5rem}
.section-header{font-size:1.2rem;font-weight:700;border-left:4px solid
#4F6EF5;padding-left:0.6rem;margin:1rem 0 0.5rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training ML models...")
def get_models():
    pipe  = run_pipeline()
    df_fe = engineer_features(pipe["clean_df"])
    reg   = train_regression(df_fe,    save_dir="models")
    clf   = train_classification(df_fe, save_dir="models")
    return {
        "reg_rf":  reg["random_forest"]["model"],
        "clf_rf":  clf["random_forest"]["model"],
        "reg_imp": reg["random_forest"]["feature_importances"],
        "df_fe":   df_fe,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Setup")
    student_name  = st.text_input("Your Name", value="Rahul Sharma")
    hours_per_day = st.slider("Study Hours / Day", 1.0, 12.0, 5.0, 0.5)
    target_score  = st.slider("Target Score (%)", 50, 100, 75)

    st.markdown("---")
    st.markdown("### Add Subjects")

    PRESETS = {
        "Mathematics":      0.85,
        "Physics":          0.80,
        "Chemistry":        0.75,
        "Computer Science": 0.70,
        "Biology":          0.60,
        "English":          0.45,
        "History":          0.50,
    }

    if "subjects" not in st.session_state:
        st.session_state.subjects = []

    with st.form("add_subject", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            s_name = st.selectbox("Subject", list(PRESETS.keys()))
        with col2:
            s_days = st.number_input("Days Left", 1, 180, 21)
        s_score = st.slider("Past Score (%)", 0, 100, 60)
        s_comp  = st.slider("Topics Done (%)", 0, 100, 40)
        if st.form_submit_button("➕ Add Subject"):
            st.session_state.subjects.append({
                "name":           s_name,
                "difficulty":     PRESETS[s_name],
                "days_left":      int(s_days),
                "past_score":     float(s_score),
                "completion_pct": float(s_comp),
                "target_score":   float(target_score),
            })

    if st.session_state.subjects:
        st.success(f"{len(st.session_state.subjects)} subject(s) added")
        if st.button("🗑 Clear All"):
            st.session_state.subjects = []
            st.rerun()

    if not st.session_state.subjects:
        st.info("Using demo subjects")
        st.session_state.subjects = [
            {"name":"Mathematics",    "difficulty":0.85,"days_left":14,"past_score":55,"completion_pct":40,"target_score":float(target_score)},
            {"name":"Physics",        "difficulty":0.80,"days_left":21,"past_score":62,"completion_pct":55,"target_score":float(target_score)},
            {"name":"Computer Science","difficulty":0.70,"days_left":10,"past_score":70,"completion_pct":60,"target_score":float(target_score)},
            {"name":"English",        "difficulty":0.45,"days_left":30,"past_score":78,"completion_pct":80,"target_score":float(target_score)},
        ]


# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📚 Smart Study Planner AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML-powered performance prediction · dynamic scheduling · adaptive re-planning</p>', unsafe_allow_html=True)

models      = get_models()
reg_model   = models["reg_rf"]
clf_model   = models["clf_rf"]
subj_objs   = [Subject(**s) for s in st.session_state.subjects]
planner     = StudyPlanner(subj_objs, hours_per_day=hours_per_day)
schedule    = planner.generate_schedule()
summary_df  = planner.summary()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard","📅 Schedule","🤖 ML Predictions","📈 Analytics"])


# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### Welcome, {student_name}!")
    predictions = []
    for s in subj_objs:
        feat = {
            "study_hrs_day":          hours_per_day,
            "consistency":            0.75,
            "difficulty":             s.difficulty,
            "days_left":              s.days_left,
            "past_score":             s.past_score,
            "completion_pct":         s.completion_pct,
            "urgency_score":          s.urgency,
            "performance_gap":        s.performance_gap,
            "study_efficiency":       s.past_score * 0.75 / (s.difficulty * hours_per_day + 1),
            "completion_rate":        s.completion_pct / 100,
            "remaining_topics_ratio": 1 - s.completion_pct / 100,
            "productive_hours":       0.75 * hours_per_day,
            "hard_work_remaining":    s.difficulty * (1 - s.completion_pct / 100),
        }
        avail = [c for c in FEATURE_COLS if c in feat]
        X     = np.array([[feat[c] for c in avail]])
        pred  = float(np.clip(reg_model.predict(X)[0], 0, 100))
        risk  = int(clf_model.predict(X)[0])
        conf  = float(clf_model.predict_proba(X)[0][0] * 100)
        predictions.append({
            "Subject":    s.name,
            "Past Score": s.past_score,
            "Predicted":  round(pred, 1),
            "At Risk":    "⚠ Yes" if risk else "✅ No",
            "Confidence": f"{round(conf,1)}%",
            "Days Left":  s.days_left,
        })

    pred_df = pd.DataFrame(predictions)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects",            len(pred_df))
    c2.metric("Avg Predicted Score", f"{pred_df['Predicted'].mean():.1f}%")
    c3.metric("At Risk",             pred_df['At Risk'].str.contains('Yes').sum())
    c4.metric("Avg Days to Exam",    f"{pred_df['Days Left'].mean():.0f}d")
    st.markdown("---")
    st.dataframe(pred_df, use_container_width=True)


# ── Tab 2: