import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import smtplib
import json
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
warnings.filterwarnings("ignore")

from src.data_pipeline        import run_pipeline
from src.feature_engineering  import engineer_features, FEATURE_COLS
from src.planner              import Subject, StudyPlanner
from src.performance_model    import train_regression, train_classification

st.set_page_config(
    page_title="StudyAI — Smart Planner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(79,110,245,0.3);
}
.hero-section {
    background: linear-gradient(135deg, rgba(79,110,245,0.15), rgba(108,62,232,0.15));
    border: 1px solid rgba(79,110,245,0.3);
    border-radius: 20px; padding: 2.5rem; margin-bottom: 2rem; text-align: center;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #4F6EF5, #a78bfa, #6EE7B7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: rgba(255,255,255,0.6); font-size: 1rem; }
.kpi-card {
    background: linear-gradient(135deg, rgba(79,110,245,0.1), rgba(108,62,232,0.1));
    border: 1px solid rgba(79,110,245,0.25); border-radius: 16px;
    padding: 1.5rem; text-align: center;
}
.kpi-number { font-size: 2.4rem; font-weight: 800; color: #4F6EF5; line-height: 1; }
.kpi-label { font-size: 0.75rem; color: rgba(255,255,255,0.5);
             text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; }
.subject-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
}
.subject-card:hover { border-color: rgba(79,110,245,0.4); }
.badge-risk {
    background: rgba(239,68,68,0.2); color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    padding: 3px 12px; border-radius: 99px; font-size: 0.75rem; font-weight: 600;
}
.badge-safe {
    background: rgba(110,231,183,0.15); color: #6EE7B7;
    border: 1px solid rgba(110,231,183,0.3);
    padding: 3px 12px; border-radius: 99px; font-size: 0.75rem; font-weight: 600;
}
.section-title {
    font-size: 1.3rem; font-weight: 700; color: white;
    margin: 1.5rem 0 1rem; padding-left: 0.75rem;
    border-left: 4px solid #4F6EF5;
}
.schedule-day {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
}
.pred-result {
    background: linear-gradient(135deg, rgba(79,110,245,0.15), rgba(110,231,183,0.1));
    border: 1px solid rgba(79,110,245,0.3);
    border-radius: 20px; padding: 2rem; text-align: center; margin: 1rem 0;
}
.pred-score-big {
    font-size: 4rem; font-weight: 800;
    background: linear-gradient(135deg, #4F6EF5, #6EE7B7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.tip-box {
    background: rgba(110,231,183,0.08); border: 1px solid rgba(110,231,183,0.25);
    border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0;
    color: #6EE7B7; font-size: 0.9rem;
}
.warning-box {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0;
    color: #f87171; font-size: 0.9rem;
}
.quiz-card {
    background: rgba(79,110,245,0.08); border: 1px solid rgba(79,110,245,0.2);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
}
.syllabus-topic {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.75rem;
}
.stButton > button {
    background: linear-gradient(135deg, #4F6EF5, #6C3EE8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important; padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.08) !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(255,255,255,0.5) !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F6EF5, #6C3EE8) !important;
    color: white !important;
}
div[data-testid="stMetricValue"] { color: #4F6EF5 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "logged_in": False, "username": "", "email": "",
    "page": "login", "subjects": [], "syllabus": {},
    "quiz_score": 0, "quiz_total": 0, "reminder_sent": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── AUTH ──────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:2rem 0 1rem'>
            <div style='font-size:3rem'>🎓</div>
            <h1 style='color:white;font-weight:800;margin:0'>StudyAI</h1>
            <p style='color:rgba(255,255,255,0.4);margin:0'>Your intelligent study companion</p>
        </div>""", unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.page = "login"; st.rerun()
        with t2:
            if st.button("✨ Sign Up", use_container_width=True):
                st.session_state.page = "signup"; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.page == "login":
            st.markdown("<h3 style='color:white;text-align:center'>Welcome back 👋</h3>",
                        unsafe_allow_html=True)
            email = st.text_input("📧 Email", placeholder="you@example.com")
            passw = st.text_input("🔒 Password", type="password")
            if st.button("Login →", use_container_width=True):
                if email and passw:
                    st.session_state.logged_in = True
                    st.session_state.username  = email.split("@")[0].capitalize()
                    st.session_state.email     = email
                    st.rerun()
                else:
                    st.error("Please fill all fields")
            st.caption("Demo: any email + password works")

        else:
            st.markdown("<h3 style='color:white;text-align:center'>Create Account ✨</h3>",
                        unsafe_allow_html=True)
            name  = st.text_input("👤 Full Name")
            email = st.text_input("📧 Email")
            passw = st.text_input("🔒 Password (min 6 chars)", type="password")
            goal  = st.selectbox("🎯 Goal", ["Score above 75%","Score above 85%",
                                              "Just pass","Top of class"])
            if st.button("Create Account →", use_container_width=True):
                if name and email and passw and len(passw) >= 6:
                    st.session_state.logged_in = True
                    st.session_state.username  = name
                    st.session_state.email     = email
                    st.rerun()
                else:
                    st.error("Fill all fields. Password min 6 chars.")
    st.stop()


# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Training AI models...")
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


# ── Quiz bank ─────────────────────────────────────────────────────────────────
QUIZ_BANK = {
    "Mathematics": [
        {"q": "What is the derivative of x²?",
         "options": ["2x", "x²", "2", "x"], "answer": "2x"},
        {"q": "What is ∫2x dx?",
         "options": ["x²+C", "2x²+C", "x+C", "2+C"], "answer": "x²+C"},
        {"q": "If a² + b² = c², this is called?",
         "options": ["Pythagorean theorem","Euler's formula","Binomial theorem","Fermat's theorem"],
         "answer": "Pythagorean theorem"},
        {"q": "What is log(1)?",
         "options": ["0","1","-1","undefined"], "answer": "0"},
    ],
    "Physics": [
        {"q": "What is Newton's second law?",
         "options": ["F=ma","E=mc²","F=mv","P=mv"], "answer": "F=ma"},
        {"q": "Unit of electric current?",
         "options": ["Ampere","Volt","Ohm","Watt"], "answer": "Ampere"},
        {"q": "Speed of light in vacuum?",
         "options": ["3×10⁸ m/s","3×10⁶ m/s","3×10¹⁰ m/s","3×10⁴ m/s"],
         "answer": "3×10⁸ m/s"},
        {"q": "Which law states energy cannot be created or destroyed?",
         "options": ["1st law of thermodynamics","Newton's 1st law",
                     "Ohm's law","Faraday's law"],
         "answer": "1st law of thermodynamics"},
    ],
    "Computer Science": [
        {"q": "What does CPU stand for?",
         "options": ["Central Processing Unit","Computer Personal Unit",
                     "Core Processing Utility","Central Program Unit"],
         "answer": "Central Processing Unit"},
        {"q": "Which data structure uses LIFO?",
         "options": ["Stack","Queue","Array","Tree"], "answer": "Stack"},
        {"q": "What is the time complexity of binary search?",
         "options": ["O(log n)","O(n)","O(n²)","O(1)"], "answer": "O(log n)"},
        {"q": "HTML stands for?",
         "options": ["HyperText Markup Language","High Tech Modern Language",
                     "HyperText Modern Links","High Text Markup Language"],
         "answer": "HyperText Markup Language"},
    ],
    "Chemistry": [
        {"q": "What is the atomic number of Carbon?",
         "options": ["6","12","8","14"], "answer": "6"},
        {"q": "What is the chemical formula of water?",
         "options": ["H₂O","HO₂","H₂O₂","H₃O"], "answer": "H₂O"},
        {"q": "Which gas is most abundant in Earth's atmosphere?",
         "options": ["Nitrogen","Oxygen","Carbon dioxide","Argon"],
         "answer": "Nitrogen"},
        {"q": "pH of pure water?",
         "options": ["7","0","14","5"], "answer": "7"},
    ],
    "Biology": [
        {"q": "What is the powerhouse of the cell?",
         "options": ["Mitochondria","Nucleus","Ribosome","Golgi body"],
         "answer": "Mitochondria"},
        {"q": "DNA stands for?",
         "options": ["Deoxyribonucleic Acid","Diribonucleic Acid",
                     "Deoxyribose Nucleic Acid","Dynamic Nucleic Acid"],
         "answer": "Deoxyribonucleic Acid"},
        {"q": "How many chromosomes do humans have?",
         "options": ["46","23","48","44"], "answer": "46"},
        {"q": "Which organ produces insulin?",
         "options": ["Pancreas","Liver","Kidney","Heart"], "answer": "Pancreas"},
    ],
    "English": [
        {"q": "What is a synonym for 'happy'?",
         "options": ["Joyful","Sad","Angry","Tired"], "answer": "Joyful"},
        {"q": "Identify the noun: 'The cat sat on the mat'",
         "options": ["cat","sat","on","the"], "answer": "cat"},
        {"q": "What is an antonym of 'ancient'?",
         "options": ["Modern","Old","Historic","Classic"], "answer": "Modern"},
        {"q": "Which is a conjunction?",
         "options": ["and","run","happy","quickly"], "answer": "and"},
    ],
    "History": [
        {"q": "In which year did World War 2 end?",
         "options": ["1945","1939","1942","1950"], "answer": "1945"},
        {"q": "Who was the first President of the United States?",
         "options": ["George Washington","Abraham Lincoln",
                     "Thomas Jefferson","John Adams"],
         "answer": "George Washington"},
        {"q": "The French Revolution began in which year?",
         "options": ["1789","1776","1804","1815"], "answer": "1789"},
        {"q": "Mahatma Gandhi led independence movement of which country?",
         "options": ["India","Pakistan","Bangladesh","Sri Lanka"],
         "answer": "India"},
    ],
}


# ── Email reminder ────────────────────────────────────────────────────────────
def send_reminder_email(to_email, student_name, subjects_info):
    """
    Sends a study reminder email using Gmail SMTP.
    Requires SMTP credentials in Streamlit secrets.
    """
    try:
        sender   = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
    except Exception:
        return False, "Email credentials not configured in Streamlit secrets."

    subject_lines = "\n".join([
        f"  • {s['name']} — {s['days_left']} days left (Past score: {s['past_score']}%)"
        for s in subjects_info
    ])

    body = f"""
Hi {student_name}! 👋

This is your StudyAI reminder to keep up your study momentum!

📚 YOUR SUBJECTS:
{subject_lines}

💡 TIPS:
- Study consistently every day — even 1 hour helps
- Focus on weak subjects first
- Complete pending topics before your exam
- Take practice quizzes on StudyAI to test yourself

Keep going — you've got this! 💪

Best,
StudyAI Team
    """

    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = to_email
    msg["Subject"] = f"📚 StudyAI Reminder — {student_name}, your exams are coming up!"
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1rem 0; border-bottom:1px solid rgba(79,110,245,0.2); margin-bottom:1rem'>
        <div style='font-size:1.5rem'>🎓</div>
        <div style='color:white;font-weight:700;font-size:1.1rem'>{st.session_state.username}</div>
        <div style='color:rgba(255,255,255,0.4);font-size:0.8rem'>{st.session_state.email}</div>
    </div>""", unsafe_allow_html=True)

    hours_per_day = st.slider("⏰ Study Hours / Day", 1.0, 12.0, 5.0, 0.5)
    target_score  = st.slider("🎯 Target Score (%)",  50, 100, 75)

    st.markdown("---")
    st.markdown("### ➕ Add Subject")

    PRESETS = {
        "Mathematics": 0.85, "Physics": 0.80, "Chemistry": 0.75,
        "Computer Science": 0.70, "Biology": 0.60,
        "English": 0.45, "History": 0.50,
    }

    with st.form("add_subject", clear_on_submit=True):
        s_name  = st.selectbox("Subject", list(PRESETS.keys()))
        s_days  = st.number_input("Days to Exam", 1, 180, 21)
        s_score = st.slider("Past Score (%)", 0, 100, 60)
        s_comp  = st.slider("Topics Done (%)", 0, 100, 40)
        if st.form_submit_button("➕ Add", use_container_width=True):
            # avoid duplicates
            existing = [s["name"] for s in st.session_state.subjects]
            if s_name in existing:
                st.warning(f"{s_name} already added!")
            else:
                st.session_state.subjects.append({
                    "name": s_name, "difficulty": PRESETS[s_name],
                    "days_left": int(s_days), "past_score": float(s_score),
                    "completion_pct": float(s_comp),
                    "target_score": float(target_score),
                })
                st.success(f"Added {s_name}!")

    if st.session_state.subjects:
        st.markdown(f"**{len(st.session_state.subjects)} subject(s) added**")
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.subjects = []
            st.rerun()

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","subjects","syllabus","quiz_score","quiz_total"]:
            st.session_state[k] = False if k == "logged_in" else [] if k == "subjects" else {} if k == "syllabus" else 0
        st.rerun()

    # Demo subjects fallback
    if not st.session_state.subjects:
        st.session_state.subjects = [
            {"name":"Mathematics",     "difficulty":0.85,"days_left":14,
             "past_score":55,"completion_pct":40,"target_score":float(target_score)},
            {"name":"Physics",         "difficulty":0.80,"days_left":21,
             "past_score":62,"completion_pct":55,"target_score":float(target_score)},
            {"name":"Computer Science","difficulty":0.70,"days_left":10,
             "past_score":70,"completion_pct":60,"target_score":float(target_score)},
            {"name":"English",         "difficulty":0.45,"days_left":30,
             "past_score":78,"completion_pct":80,"target_score":float(target_score)},
        ]


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-section'>
    <div class='hero-title'>Welcome back, {st.session_state.username}! 👋</div>
    <div class='hero-sub'>AI-powered study planning · ML predictions · Smart scheduling</div>
</div>""", unsafe_allow_html=True)

# ── Load models & build planner ───────────────────────────────────────────────
models    = get_models()
reg_model = models["reg_rf"]
clf_model = models["clf_rf"]
subj_objs = [Subject(**s) for s in st.session_state.subjects]
planner   = StudyPlanner(subj_objs, hours_per_day=hours_per_day)
schedule  = planner.generate_schedule()
summary_df= planner.summary()

# Compute predictions for all subjects
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
        "Subject": s.name, "Past Score": s.past_score,
        "Predicted": round(pred, 1), "At Risk": risk,
        "Confidence": round(conf, 1), "Days Left": s.days_left,
        "Priority": round(s.priority_score, 3),
    })

pred_df  = pd.DataFrame(predictions)
n_risk   = int(pred_df["At Risk"].sum())
avg_pred = float(pred_df["Predicted"].mean())

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
for col, num, label, color in [
    (k1, len(pred_df),                    "📚 Subjects",       "#4F6EF5"),
    (k2, f"{avg_pred:.1f}%",              "🎯 Avg Predicted",  "#6EE7B7"),
    (k3, n_risk,                          "⚠️ At Risk",        "#f87171" if n_risk else "#6EE7B7"),
    (k4, f"{pred_df['Days Left'].mean():.0f}d", "📅 Avg Days", "#a78bfa"),
]:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='color:{color}'>{num}</div>
            <div class='kpi-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "📅 Schedule", "🤖 ML Predictions",
    "📈 Analytics", "📝 Syllabus", "🧠 Quiz"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Subject Overview</div>",
                unsafe_allow_html=True)
    for p in predictions:
        badge     = "<span class='badge-risk'>⚠ At Risk</span>" if p["At Risk"] \
                    else "<span class='badge-safe'>✅ Safe</span>"
        bar_color = "#6EE7B7" if p["Predicted"] >= target_score else "#f87171"
        st.markdown(f"""
        <div class='subject-card'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <div style='font-size:1.1rem;font-weight:600;color:white'>{p['Subject']}</div>
                    <div style='color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:2px'>
                        {p['Days Left']} days left &nbsp;·&nbsp; Past: {p['Past Score']}%
                        &nbsp;·&nbsp; Priority: {p['Priority']}
                    </div>
                </div>
                <div style='text-align:right'>
                    <div style='font-size:1.8rem;font-weight:800;color:{bar_color}'>
                        {p['Predicted']}%
                    </div>
                    {badge}
                </div>
            </div>
            <div style='background:rgba(255,255,255,0.08);border-radius:99px;
                        height:8px;margin-top:10px'>
                <div style='width:{int(p["Predicted"])}%;height:8px;border-radius:99px;
                     background:linear-gradient(90deg,{bar_color},{bar_color}88)'></div>
            </div>
        </div>""", unsafe_allow_html=True)

    if n_risk > 0:
        names = [p["Subject"] for p in predictions if p["At Risk"]]
        st.markdown(f"""
        <div class='warning-box'>⚠️ <strong>{n_risk} subject(s) at risk:</strong>
        {', '.join(names)}. Increase study hours or topic completion.</div>""",
        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='tip-box'>✅ All subjects on track! Keep up the momentum.</div>""",
        unsafe_allow_html=True)

    # Email reminder section
    st.markdown("<div class='section-title'>📧 Email Reminder</div>",
                unsafe_allow_html=True)
    reminder_email = st.text_input("Send reminder to email",
                                    value=st.session_state.email,
                                    placeholder="student@example.com")
    if st.button("📨 Send Study Reminder Email"):
        ok, msg = send_reminder_email(
            reminder_email,
            st.session_state.username,
            st.session_state.subjects
        )
        if ok:
            st.success(f"✅ Reminder sent to {reminder_email}!")
        else:
            st.info(f"ℹ️ Email not configured yet. To enable: add Gmail credentials "
                    f"to Streamlit secrets. Error: {msg}")
            st.markdown("""
            <div class='tip-box'>
            📌 <strong>To enable real emails:</strong><br>
            1. Go to your Streamlit Cloud app → Settings → Secrets<br>
            2. Add this:<br><br>
            <code>[email]<br>
            sender = "yourgmail@gmail.com"<br>
            password = "your_app_password"</code><br><br>
            3. Enable 2FA on Gmail → Generate App Password → paste above
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Your Personalized Study Schedule</div>",
                unsafe_allow_html=True)

    if not schedule:
        st.warning("No schedule yet. Add subjects in the sidebar.")
    else:
        show_days = st.slider("Days to preview", 3, min(30, len(schedule)), 7)
        COLORS_LIST = ["#4F6EF5","#6EE7B7","#f59e0b","#a78bfa","#f87171","#34d399"]
        all_subjs   = list({s for d in schedule[:show_days] for s in d.sessions})
        color_map   = {s: COLORS_LIST[i % len(COLORS_LIST)]
                       for i, s in enumerate(all_subjs)}

        for day in schedule[:show_days]:
            blocks = "".join([
                f"""<span style='background:{color_map.get(s,"#4F6EF5")}22;
                    color:{color_map.get(s,"#4F6EF5")};
                    border:1px solid {color_map.get(s,"#4F6EF5")}44;
                    border-radius:8px;padding:4px 12px;
                    font-size:0.8rem;font-weight:600;margin-right:6px'>
                    {s.split()[0]} {h:.1f}h</span>"""
                for s, h in day.sessions.items()
            ])
            st.markdown(f"""
            <div class='schedule-day'>
                <div style='font-size:0.8rem;color:rgba(255,255,255,0.4);
                            font-weight:500;margin-bottom:6px'>
                    {day.date.strftime('%A, %d %B')}
                </div>
                <div>{blocks}</div>
            </div>""", unsafe_allow_html=True)

        # Stacked bar chart
        st.markdown("<div class='section-title'>Hours Distribution Chart</div>",
                    unsafe_allow_html=True)
        plot_days = schedule[:min(14, len(schedule))]
        dates     = [d.date.strftime("%d/%m") for d in plot_days]
        data_mat  = np.array([[d.sessions.get(s, 0) for s in all_subjs]
                               for d in plot_days])
        fig, ax   = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        bottom = np.zeros(len(plot_days))
        for i, subj in enumerate(all_subjs):
            c = color_map.get(subj, "#4F6EF5")
            ax.bar(dates, data_mat[:, i], bottom=bottom,
                   label=subj, color=c, alpha=0.85)
            bottom += data_mat[:, i]
        ax.set_ylabel("Hours", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("rgba(255,255,255,0.1)")
        plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
        ax.legend(fontsize=8, labelcolor="white", facecolor="#16213e",
                  edgecolor="rgba(255,255,255,0.1)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Priority table
        st.markdown("<div class='section-title'>Priority Summary</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            summary_df[["Subject","Daily Hours","Priority Score","Days Left"]].style
            .background_gradient(cmap="Blues", subset=["Priority Score"]),
            use_container_width=True
        )

    # Adaptive re-planning
    st.markdown("<div class='section-title'>🔁 Update Progress & Re-Plan</div>",
                unsafe_allow_html=True)
    subj_names = [s["name"] for s in st.session_state.subjects]
    if subj_names:
        up_subj  = st.selectbox("Subject to update", subj_names)
        ca, cb   = st.columns(2)
        with ca:
            new_comp = st.slider("New Completion %", 0, 100, 60)
        with cb:
            new_sc = st.number_input("New Score (0 = skip)", 0, 100, 0)
        if st.button("🔄 Update & Re-Plan", use_container_width=True):
            planner.update_progress(up_subj, new_comp,
                                     float(new_sc) if new_sc > 0 else None)
            st.success(f"✅ Schedule updated for {up_subj}!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Live Performance Predictor</div>",
                unsafe_allow_html=True)
    st.caption("Adjust sliders — prediction updates instantly")

    cl, cr = st.columns(2)
    with cl:
        i_study = st.slider("📖 Study hrs/day",     0.5, 12.0, 4.0, 0.5, key="p_study")
        i_cons  = st.slider("💪 Consistency",       0.0,  1.0, 0.7, 0.05, key="p_cons")
        i_diff  = st.slider("🧠 Difficulty",        0.1,  1.0, 0.7, 0.05, key="p_diff")
    with cr:
        i_days  = st.number_input("📅 Days to Exam", 1, 180, 20, key="p_days")
        i_past  = st.slider("📝 Past Score (%)",     0, 100, 60, key="p_past")
        i_comp  = st.slider("✅ Topics Done (%)",    0, 100, 50, key="p_comp")

    feat_in = {
        "study_hrs_day":          i_study,
        "consistency":            i_cons,
        "difficulty":             i_diff,
        "days_left":              i_days,
        "past_score":             i_past,
        "completion_pct":         i_comp,
        "urgency_score":          i_diff / (i_days + 1),
        "performance_gap":        max(0, target_score - i_past),
        "study_efficiency":       i_past * i_cons / (i_diff * i_study + 1),
        "completion_rate":        i_comp / 100,
        "remaining_topics_ratio": 1 - i_comp / 100,
        "productive_hours":       i_cons * i_study,
        "hard_work_remaining":    i_diff * (1 - i_comp / 100),
    }
    avail = [c for c in FEATURE_COLS if c in feat_in]
    X_in  = np.array([[feat_in[c] for c in avail]])
    pred  = float(np.clip(reg_model.predict(X_in)[0], 0, 100))
    risk  = int(clf_model.predict(X_in)[0])
    conf  = float(clf_model.predict_proba(X_in)[0][0] * 100)
    rc    = "#f87171" if risk else "#6EE7B7"

    st.markdown(f"""
    <div class='pred-result'>
        <div style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-bottom:0.5rem'>
            Predicted Score
        </div>
        <div class='pred-score-big'>{pred:.1f}%</div>
        <div style='margin-top:1rem;display:flex;justify-content:center;gap:3rem'>
            <div style='text-align:center'>
                <div style='color:{rc};font-weight:700;font-size:1.1rem'>
                    {"⚠ At Risk" if risk else "✅ Safe"}
                </div>
                <div style='color:rgba(255,255,255,0.4);font-size:0.8rem'>Risk Level</div>
            </div>
            <div style='text-align:center'>
                <div style='color:#a78bfa;font-weight:700;font-size:1.1rem'>{conf:.1f}%</div>
                <div style='color:rgba(255,255,255,0.4);font-size:0.8rem'>Confidence</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    msg_html = f"""<div class='{"warning-box" if risk else "tip-box"}'>
        {"⚠️ At risk — increase study hours and topic completion." if risk
         else "✅ On track! Maintain your current study habits."}
    </div>"""
    st.markdown(msg_html, unsafe_allow_html=True)

    # Feature importance chart
    st.markdown("<div class='section-title'>Feature Importance</div>",
                unsafe_allow_html=True)
    avail_feats = [c for c in FEATURE_COLS if c in models["df_fe"].columns]
    imp = models["reg_imp"]
    idx = np.argsort(imp)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#1a1a2e")
    ax2.barh(np.array(avail_feats)[idx], imp[idx],
             color=["#4F6EF5" if v > 0.1 else "#6C3EE8" for v in imp[idx]])
    ax2.set_xlabel("Importance", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("rgba(255,255,255,0.1)")
    ax2.set_title("What drives predicted score?", color="white", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(subj_objs) < 2:
        st.info("Add at least 2 subjects to see analytics.")
    else:
        names   = [s.name for s in subj_objs]
        past    = [s.past_score for s in subj_objs]
        pred_sc = [p["Predicted"] for p in predictions]
        x, w    = np.arange(len(names)), 0.35

        # Chart 1: Past vs Predicted
        st.markdown("<div class='section-title'>Past vs Predicted Scores</div>",
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fig3.patch.set_facecolor("#1a1a2e")
        ax3.set_facecolor("#1a1a2e")
        ax3.bar(x-w/2, past,    w, label="Past Score", color="#4F6EF5", alpha=0.85)
        ax3.bar(x+w/2, pred_sc, w, label="Predicted",  color="#6EE7B7", alpha=0.85)
        ax3.axhline(target_score, color="#f87171", linestyle="--",
                    label=f"Target {target_score}%", linewidth=1.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=20, ha="right",
                             fontsize=9, color="white")
        ax3.set_ylim(0, 110)
        ax3.tick_params(colors="white")
        ax3.spines[:].set_color("rgba(255,255,255,0.1)")
        ax3.legend(labelcolor="white", facecolor="#16213e",
                   edgecolor="rgba(255,255,255,0.1)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Chart 2: Topic completion
        st.markdown("<div class='section-title'>Topic Completion</div>",
                    unsafe_allow_html=True)
        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
        fig4.patch.set_facecolor("#1a1a2e")
        for ax in axes4:
            ax.set_facecolor("#1a1a2e")

        comp_vals    = [s.completion_pct for s in subj_objs]
        wedge_colors = ["#4F6EF5","#6EE7B7","#f59e0b","#a78bfa",
                        "#f87171","#34d399","#60a5fa"]
        axes4[0].pie(comp_vals, labels=names, autopct="%1.0f%%",
                     startangle=90,
                     colors=wedge_colors[:len(names)],
                     textprops={"color":"white","fontsize":9})
        axes4[0].set_title("Completion Distribution", color="white", fontweight="bold")

        # Priority bar
        priorities = [s.priority_score for s in subj_objs]
        bars = axes4[1].bar(names, priorities,
                             color=wedge_colors[:len(names)], alpha=0.85)
        axes4[1].set_title("Priority Score (higher = needs more focus)",
                            color="white", fontweight="bold")
        axes4[1].tick_params(colors="white")
        axes4[1].spines[:].set_color("rgba(255,255,255,0.1)")
        plt.xticks(rotation=20, ha="right", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # Hours impact simulator
        st.markdown("<div class='section-title'>Study Hours Impact Simulator</div>",
                    unsafe_allow_html=True)
        sel = st.selectbox("Select subject", names, key="analytics_sel")
        s_s = next(s for s in subj_objs if s.name == sel)
        hrs_r     = np.linspace(0.5, 12, 60)
        sim_preds = []
        for h in hrs_r:
            f = {
                "study_hrs_day":          h,
                "consistency":            0.75,
                "difficulty":             s_s.difficulty,
                "days_left":              s_s.days_left,
                "past_score":             s_s.past_score,
                "completion_pct":         s_s.completion_pct,
                "urgency_score":          s_s.urgency,
                "performance_gap":        s_s.performance_gap,
                "study_efficiency":       s_s.past_score*0.75/(s_s.difficulty*h+1),
                "completion_rate":        s_s.completion_pct/100,
                "remaining_topics_ratio": 1-s_s.completion_pct/100,
                "productive_hours":       0.75*h,
                "hard_work_remaining":    s_s.difficulty*(1-s_s.completion_pct/100),
            }
            av = [c for c in FEATURE_COLS if c in f]
            X_ = np.array([[f[c] for c in av]])
            sim_preds.append(float(np.clip(reg_model.predict(X_)[0], 0, 100)))

        fig5, ax5 = plt.subplots(figsize=(10, 3.5))
        fig5.patch.set_facecolor("#1a1a2e")
        ax5.set_facecolor("#1a1a2e")
        ax5.plot(hrs_r, sim_preds, color="#4F6EF5", linewidth=2.5)
        ax5.fill_between(hrs_r, sim_preds, alpha=0.15, color="#4F6EF5")
        ax5.axhline(target_score,  color="#f87171", linestyle="--",
                    label=f"Target {target_score}%", linewidth=1.5)
        ax5.axvline(hours_per_day, color="#f59e0b", linestyle=":",
                    label=f"Current {hours_per_day}h", linewidth=1.5)
        ax5.set_xlabel("Study Hours/Day", color="white")
        ax5.set_ylabel("Predicted Score", color="white")
        ax5.set_title(f"Score vs Hours — {sel}", color="white", fontweight="bold")
        ax5.tick_params(colors="white")
        ax5.spines[:].set_color("rgba(255,255,255,0.1)")
        ax5.legend(labelcolor="white", facecolor="#16213e",
                   edgecolor="rgba(255,255,255,0.1)")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SYLLABUS PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>📝 Syllabus & Study Plan Generator</div>",
                unsafe_allow_html=True)
    st.caption("Add your syllabus topics — AI generates a focused study plan")

    subj_names = [s["name"] for s in st.session_state.subjects]
    if not subj_names:
        st.warning("Add subjects in the sidebar first.")
    else:
        sel_subj = st.selectbox("Select Subject", subj_names, key="syl_subj")

        # Add topics
        with st.form("add_topic", clear_on_submit=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                topic_name = st.text_input("Topic Name",
                                            placeholder="e.g. Differentiation, Vectors")
            with c2:
                topic_hrs  = st.number_input("Est. Hours", 0.5, 20.0, 2.0, 0.5)
            with c3:
                topic_done = st.selectbox("Status", ["Not Started","In Progress","Done"])
            if st.form_submit_button("➕ Add Topic", use_container_width=True):
                if topic_name:
                    if sel_subj not in st.session_state.syllabus:
                        st.session_state.syllabus[sel_subj] = []
                    st.session_state.syllabus[sel_subj].append({
                        "topic": topic_name,
                        "hours": topic_hrs,
                        "status": topic_done,
                    })
                    st.success(f"Added: {topic_name}")

        # Show topics
        if sel_subj in st.session_state.syllabus and st.session_state.syllabus[sel_subj]:
            topics = st.session_state.syllabus[sel_subj]
            st.markdown(f"**{len(topics)} topics for {sel_subj}**")

            total_hrs  = sum(t["hours"] for t in topics)
            done_hrs   = sum(t["hours"] for t in topics if t["status"] == "Done")
            progress   = done_hrs / total_hrs if total_hrs > 0 else 0

            p1, p2, p3 = st.columns(3)
            p1.metric("Total Topics",    len(topics))
            p2.metric("Total Hours",     f"{total_hrs:.1f}h")
            p3.metric("Completion",      f"{progress*100:.0f}%")

            st.progress(progress)

            status_colors = {
                "Done":        ("#6EE7B7", "✅"),
                "In Progress": ("#f59e0b", "🔄"),
                "Not Started": ("#f87171", "⭕"),
            }

            for t in topics:
                color, icon = status_colors.get(t["status"], ("#fff","•"))
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);
                     border:1px solid rgba(255,255,255,0.08);
                     border-left:4px solid {color};
                     border-radius:10px;padding:0.75rem 1rem;
                     margin-bottom:0.5rem;display:flex;
                     justify-content:space-between;align-items:center'>
                    <div>
                        <span style='color:white;font-weight:500'>{icon} {t['topic']}</span>
                        <span style='color:rgba(255,255,255,0.4);
                              font-size:0.8rem;margin-left:1rem'>
                            {t['hours']}h estimated
                        </span>
                    </div>
                    <span style='color:{color};font-size:0.8rem;font-weight:600'>
                        {t['status']}
                    </span>
                </div>""", unsafe_allow_html=True)

            # Generate AI study plan
            if st.button("🤖 Generate AI Study Plan", use_container_width=True):
                pending = [t for t in topics if t["status"] != "Done"]
                if not pending:
                    st.balloons()
                    st.success("🎉 All topics completed! You're ready for the exam!")
                else:
                    subj_data = next(
                        (s for s in st.session_state.subjects if s["name"] == sel_subj),
                        None
                    )
                    days_left = subj_data["days_left"] if subj_data else 14
                    hrs_day   = hours_per_day
                    total_needed = sum(t["hours"] for t in pending)
                    days_needed  = total_needed / hrs_day

                    st.markdown("<div class='section-title'>📋 Your AI Study Plan</div>",
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='tip-box'>
                        📊 <strong>Plan Summary:</strong> {len(pending)} topics remaining ·
                        {total_needed:.1f}h total work ·
                        {hrs_day}h/day ·
                        Need ~{days_needed:.0f} days ·
                        You have {days_left} days
                        {'✅ Enough time!' if days_left >= days_needed else '⚠️ Tight schedule — increase daily hours!'}
                    </div>""", unsafe_allow_html=True)

                    # Sort by priority: not started first, then in progress
                    sorted_topics = sorted(pending,
                                           key=lambda t: 0 if t["status"]=="Not Started" else 1)
                    day = 1
                    hrs_today = 0
                    st.markdown("**Day-by-day breakdown:**")
                    for t in sorted_topics:
                        remaining_t = t["hours"]
                        while remaining_t > 0:
                            available = hrs_day - hrs_today
                            if available <= 0:
                                day += 1
                                hrs_today = 0
                                available = hrs_day
                            chunk = min(remaining_t, available)
                            st.markdown(f"""
                            <div style='background:rgba(79,110,245,0.08);
                                 border:1px solid rgba(79,110,245,0.2);
                                 border-radius:8px;padding:0.6rem 1rem;
                                 margin-bottom:4px;display:flex;
                                 justify-content:space-between'>
                                <span style='color:white'>
                                    📅 Day {day} — <strong>{t['topic']}</strong>
                                </span>
                                <span style='color:#4F6EF5;font-weight:600'>
                                    {chunk:.1f}h
                                </span>
                            </div>""", unsafe_allow_html=True)
                            hrs_today    += chunk
                            remaining_t  -= chunk

            if st.button("🗑 Clear Topics", use_container_width=True):
                st.session_state.syllabus[sel_subj] = []
                st.rerun()
        else:
            st.info("No topics added yet. Add your syllabus topics above.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-title'>🧠 Practice Quiz</div>",
                unsafe_allow_html=True)
    st.caption("Test your knowledge before the exam")

    available_quiz_subjects = [
        s["name"] for s in st.session_state.subjects
        if s["name"] in QUIZ_BANK
    ]

    if not available_quiz_subjects:
        st.info("Add subjects like Mathematics, Physics, Computer Science etc. to unlock quizzes.")
    else:
        quiz_subj = st.selectbox("Select Subject for Quiz", available_quiz_subjects)

        if "quiz_state" not in st.session_state:
            st.session_state.quiz_state = {
                "questions": [], "answers": {}, "submitted": False, "score": 0
            }

        qs = st.session_state.quiz_state

        # Start / reset quiz
        if st.button("🔀 Start New Quiz", use_container_width=True):
            pool = QUIZ_BANK.get(quiz_subj, [])
            st.session_state.quiz_state = {
                "questions": random.sample(pool, min(4, len(pool))),
                "answers":   {},
                "submitted": False,
                "score":     0,
                "subject":   quiz_subj,
            }
            st.rerun()

        qs = st.session_state.quiz_state

        if qs["questions"]:
            if not qs["submitted"]:
                st.markdown(f"**Subject: {qs.get('subject','')} · {len(qs['questions'])} questions**")
                with st.form("quiz_form"):
                    for i, q in enumerate(qs["questions"]):
                        st.markdown(f"""
                        <div class='quiz-card'>
                            <div style='color:white;font-weight:600;margin-bottom:0.75rem'>
                                Q{i+1}. {q['q']}
                            </div>
                        </div>""", unsafe_allow_html=True)
                        ans = st.radio(f"",
                                        q["options"],
                                        key=f"q_{i}",
                                        label_visibility="collapsed")
                        qs["answers"][i] = ans

                    if st.form_submit_button("✅ Submit Quiz", use_container_width=True):
                        score = sum(
                            1 for i, q in enumerate(qs["questions"])
                            if qs["answers"].get(i) == q["answer"]
                        )
                        st.session_state.quiz_state["score"]     = score
                        st.session_state.quiz_state["submitted"] = True
                        st.rerun()

            else:
                # Results
                score = qs["score"]
                total = len(qs["questions"])
                pct   = score / total * 100

                color = "#6EE7B7" if pct >= 75 else "#f59e0b" if pct >= 50 else "#f87171"
                emoji = "🎉" if pct >= 75 else "👍" if pct >= 50 else "📚"

                st.markdown(f"""
                <div class='pred-result'>
                    <div style='font-size:3rem'>{emoji}</div>
                    <div style='color:rgba(255,255,255,0.5);margin:0.5rem 0'>Quiz Score</div>
                    <div style='font-size:3.5rem;font-weight:800;color:{color}'>
                        {score}/{total}
                    </div>
                    <div style='color:rgba(255,255,255,0.6);margin-top:0.5rem'>
                        {pct:.0f}% · {"Excellent!" if pct>=75 else "Good effort!" if pct>=50 else "Keep studying!"}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Answer review
                st.markdown("<div class='section-title'>Answer Review</div>",
                            unsafe_allow_html=True)
                for i, q in enumerate(qs["questions"]):
                    user_ans    = qs["answers"].get(i, "")
                    correct_ans = q["answer"]
                    is_correct  = user_ans == correct_ans
                    icon  = "✅" if is_correct else "❌"
                    color2 = "#6EE7B7" if is_correct else "#f87171"
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03);
                         border:1px solid {color2}44;border-radius:12px;
                         padding:1rem;margin-bottom:0.75rem'>
                        <div style='color:white;font-weight:600;margin-bottom:6px'>
                            {icon} Q{i+1}. {q['q']}
                        </div>
                        <div style='font-size:0.85rem'>
                            <span style='color:rgba(255,255,255,0.5)'>Your answer: </span>
                            <span style='color:{color2};font-weight:500'>{user_ans}</span>
                        </div>
                        {"" if is_correct else f"<div style='font-size:0.85rem;margin-top:4px'><span style='color:rgba(255,255,255,0.5)'>Correct: </span><span style='color:#6EE7B7;font-weight:500'>{correct_ans}</span></div>"}
                    </div>""", unsafe_allow_html=True)

                # Update session totals
                st.session_state.quiz_score += score
                st.session_state.quiz_total += total

                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state.quiz_state["submitted"] = False
                    st.rerun()
        else:
            st.info("Click 'Start New Quiz' to begin!")

        # Overall quiz stats
        if st.session_state.quiz_total > 0:
            overall = st.session_state.quiz_score / st.session_state.quiz_total * 100
            st.markdown("---")
            st.markdown(f"""
            <div class='tip-box'>
                📊 Overall Quiz Performance: <strong>{overall:.0f}%</strong>
                ({st.session_state.quiz_score}/{st.session_state.quiz_total} correct)
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:rgba(255,255,255,0.2);font-size:0.8rem'>
    StudyAI · Python · scikit-learn · Streamlit · Built for students 🎓
</div>""", unsafe_allow_html=True)