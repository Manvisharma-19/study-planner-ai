import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import PyPDF2
import json
import re
from datetime import date, datetime
import google.generativeai as genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Gemini setup ──────────────────────────────────────────────────────────────
try:
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%) !important;
    border-right: 1px solid rgba(79,110,245,0.2);
}

.hero {
    background: linear-gradient(135deg, rgba(79,110,245,0.12), rgba(108,62,232,0.12));
    border: 1px solid rgba(79,110,245,0.25);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4F6EF5, #a78bfa, #6EE7B7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-sub {
    color: rgba(255,255,255,0.5);
    font-size: 1.1rem;
}

.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
}

.card-blue {
    background: rgba(79,110,245,0.06);
    border: 1px solid rgba(79,110,245,0.2);
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: white;
    margin: 1.5rem 0 1rem;
    padding-left: 0.75rem;
    border-left: 4px solid #4F6EF5;
}

.tip {
    background: rgba(110,231,183,0.08);
    border: 1px solid rgba(110,231,183,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #6EE7B7;
    font-size: 0.9rem;
    margin: 0.75rem 0;
}

.warn {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #f87171;
    font-size: 0.9rem;
    margin: 0.75rem 0;
}

.info-box {
    background: rgba(79,110,245,0.08);
    border: 1px solid rgba(79,110,245,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #a78bfa;
    font-size: 0.9rem;
    margin: 0.75rem 0;
}

.day-card {
    background: rgba(79,110,245,0.06);
    border: 1px solid rgba(79,110,245,0.15);
    border-left: 4px solid #4F6EF5;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}

.mcq-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
}

.score-big {
    font-size: 4rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #4F6EF5, #6EE7B7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.upload-box {
    background: rgba(79,110,245,0.04);
    border: 2px dashed rgba(79,110,245,0.3);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #4F6EF5, #6C3EE8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,110,245,0.35) !important;
}

.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(79,110,245,0.25) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: rgba(255,255,255,0.45) !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F6EF5, #6C3EE8) !important;
    color: white !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #4F6EF5, #6EE7B7) !important;
    border-radius: 99px !important;
}

div[data-testid="stMetricValue"] {
    color: #4F6EF5 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "logged_in":    False,
    "username":     "",
    "email":        "",
    "page":         "login",
    "study_plans":  {},   # {subject: plan_text}
    "quiz_state":   {},   # quiz state
    "summary":      "",   # last PDF summary
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
        <div style='text-align:center;padding:2.5rem 0 1.5rem'>
            <div style='font-size:3.5rem'>🎓</div>
            <h1 style='color:white;font-weight:800;font-size:2.2rem;margin:0.25rem 0'>
                StudyAI
            </h1>
            <p style='color:rgba(255,255,255,0.35);margin:0;font-size:0.95rem'>
                AI-powered exam preparation
            </p>
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
            st.markdown(
                "<h3 style='color:white;text-align:center;font-size:1.4rem'>"
                "Welcome back 👋</h3>",
                unsafe_allow_html=True)
            email = st.text_input("📧 Email", placeholder="you@example.com",
                                   key="li_email")
            passw = st.text_input("🔒 Password", type="password",
                                   key="li_pass")
            st.markdown("<br>", unsafe_allow_html=True)
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
            st.markdown(
                "<h3 style='color:white;text-align:center;font-size:1.4rem'>"
                "Create Account ✨</h3>",
                unsafe_allow_html=True)
            name  = st.text_input("👤 Full Name",  key="su_name")
            email = st.text_input("📧 Email",      key="su_email")
            passw = st.text_input("🔒 Password",   type="password", key="su_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True):
                if name and email and passw and len(passw) >= 6:
                    st.session_state.logged_in = True
                    st.session_state.username  = name
                    st.session_state.email     = email
                    st.rerun()
                else:
                    st.error("Fill all fields. Password min 6 chars.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(uploaded_file) -> tuple[str, str | None]:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text   = "\n".join(
            p.extract_text() or "" for p in reader.pages
        ).strip()
        if len(text) < 30:
            return "", "PDF seems empty or image-only. Please use a text-based PDF."
        return text, None
    except Exception as e:
        return "", str(e)


def call_gemini(prompt: str) -> tuple[str, str | None]:
    if not GEMINI_OK:
        return "", "Gemini API key not set. Add it to Streamlit secrets."
    try:
        model    = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return "", str(e)


def generate_study_plan(subject: str, exam_date: date,
                         hours_per_day: float,
                         syllabus_text: str) -> tuple[str, str | None]:
    today     = date.today()
    days_left = (exam_date - today).days

    if days_left <= 0:
        return "", "Exam date must be in the future."

    prompt = f"""
You are an expert study coach. Create a detailed day-by-day study plan.

Student details:
- Subject: {subject}
- Exam Date: {exam_date.strftime('%d %B %Y')}
- Days remaining: {days_left} days
- Study hours per day: {hours_per_day} hours
- Syllabus / Topics:
{syllabus_text[:2000]}

Create a complete day-by-day study plan from today until the exam.
Format each day exactly like this:

**Day 1 — [Date] — [Topic Name]**
- What to study: [specific topics]
- Focus: [key concepts]
- Time allocation: [X hours]
- Tips: [1 specific tip]

After all days, add:

**REVISION STRATEGY**
[2-3 bullet points on how to revise]

**EXAM DAY TIPS**
[3 bullet points for exam day]

Be specific, practical and motivating. Total plan must cover all days until exam.
"""
    return call_gemini(prompt)


def summarize_pdf(pdf_text: str, subject: str) -> tuple[str, str | None]:
    prompt = f"""
You are an expert teacher. Summarize the following study material for {subject}.

STUDY MATERIAL:
{pdf_text[:4000]}

Create a clear, structured summary with:

## 📌 Key Topics Covered
[List main topics]

## 🧠 Important Concepts
[Explain key concepts in simple language, 3-5 sentences each]

## ⚡ Quick Revision Points
[10-15 bullet points of most important facts]

## 🔑 Key Formulas / Definitions
[Any important formulas, definitions or dates]

## ❓ Likely Exam Areas
[3-5 topics most likely to appear in exam based on this content]

Make it student-friendly, clear and easy to revise from.
"""
    return call_gemini(prompt)


def generate_mcq_paper(pdf_text: str, subject: str,
                        n_questions: int) -> tuple[list, str | None]:
    prompt = f"""
You are an expert examiner. Create a proper MCQ question paper for {subject}.

STUDY MATERIAL:
{pdf_text[:4000]}

Generate exactly {n_questions} multiple choice questions based on this material.

Return ONLY a valid JSON array. No explanation. No markdown. No code blocks.
Exactly this format:
[
  {{
    "q": "Full question text?",
    "options": ["A. option one", "B. option two", "C. option three", "D. option four"],
    "answer": "A. option one",
    "explanation": "Brief explanation of why this is correct",
    "topic": "topic this question is from",
    "difficulty": "Easy/Medium/Hard"
  }}
]

Rules:
- Base ALL questions on the provided material
- Mix Easy, Medium and Hard questions
- Options must start with A. B. C. D.
- answer must exactly match one of the options
- Return ONLY the JSON array
"""
    text, err = call_gemini(prompt)
    if err:
        return [], err
    try:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*",     "", text)
        return json.loads(text.strip()), None
    except Exception as e:
        return [], f"Could not parse AI response. Try again. ({e})"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1.25rem 0;
         border-bottom:1px solid rgba(79,110,245,0.2);
         margin-bottom:1.25rem'>
        <div style='font-size:2rem'>🎓</div>
        <div style='color:white;font-weight:700;font-size:1.1rem;margin-top:4px'>
            {st.session_state.username}
        </div>
        <div style='color:rgba(255,255,255,0.35);font-size:0.8rem'>
            {st.session_state.email}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='color:rgba(255,255,255,0.5);font-size:0.8rem;
         text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem'>
        How to use
    </div>""", unsafe_allow_html=True)

    for step, desc in [
        ("1️⃣ Study Plan",  "Enter subject + exam date → get AI plan"),
        ("2️⃣ PDF Summary", "Upload PDF → get smart summary"),
        ("3️⃣ MCQ Quiz",    "Upload PDF → get question paper"),
    ]:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);
             border:1px solid rgba(255,255,255,0.07);
             border-radius:10px;padding:0.6rem 0.75rem;
             margin-bottom:0.5rem'>
            <div style='color:white;font-size:0.85rem;font-weight:500'>
                {step}
            </div>
            <div style='color:rgba(255,255,255,0.4);font-size:0.75rem'>
                {desc}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='hero'>
    <div class='hero-title'>StudyAI 🎓</div>
    <div class='hero-sub'>
        AI-powered study plans · PDF summaries · Smart MCQ papers
    </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📅  Study Plan",
    "📄  PDF Summary",
    "🧠  MCQ Quiz Paper",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDY PLAN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>📅 AI Study Plan Generator</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        💡 Tell the AI your subject, exam date and syllabus —
        it will create a complete day-by-day study plan just for you.
    </div>""", unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            subject = st.text_input("📚 Subject Name",
                                     placeholder="e.g. Physics, History...")
        with c2:
            exam_date = st.date_input("📅 Exam Date",
                                       min_value=date.today(),
                                       value=date.today())
        with c3:
            hours_day = st.slider("⏰ Study Hours/Day", 1.0, 12.0, 4.0, 0.5)

        days_left = (exam_date - date.today()).days
        if days_left > 0:
            st.markdown(f"""
            <div class='tip'>
                ✅ You have <strong>{days_left} days</strong> until your exam ·
                Total study time: <strong>{days_left * hours_day:.0f} hours</strong>
            </div>""", unsafe_allow_html=True)
        elif days_left == 0:
            st.markdown("""
            <div class='warn'>⚠️ Exam is today! Good luck! 🍀</div>""",
            unsafe_allow_html=True)

        st.markdown("**📝 Your Syllabus / Topics**")
        st.caption("Paste your syllabus, topics list, or anything you need to study")
        syllabus_input = st.text_area(
            "Syllabus",
            placeholder="""Example:
Chapter 1: Laws of Motion - Newton's laws, friction, circular motion
Chapter 2: Work, Energy & Power - kinetic energy, potential energy, conservation
Chapter 3: Gravitation - universal law, escape velocity, satellites
Chapter 4: Thermodynamics - laws, heat engines, entropy""",
            height=200,
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🤖 Generate My Study Plan", use_container_width=True):
            if not subject.strip():
                st.error("Please enter a subject name.")
            elif days_left <= 0:
                st.error("Please select a future exam date.")
            elif not syllabus_input.strip():
                st.error("Please enter your syllabus or topics.")
            elif not GEMINI_OK:
                st.error("Gemini API key not configured. Add it to Streamlit secrets.")
            else:
                with st.spinner(f"🧠 AI is creating your {days_left}-day study plan..."):
                    plan, err = generate_study_plan(
                        subject, exam_date, hours_day, syllabus_input
                    )
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.session_state.study_plans[subject] = plan
                    st.success("✅ Study plan generated!")

    # Show plan
    if st.session_state.study_plans:
        latest_subj = list(st.session_state.study_plans.keys())[-1]
        latest_plan = st.session_state.study_plans[latest_subj]

        st.markdown(
            f"<div class='section-title'>📋 Your Study Plan — {latest_subj}</div>",
            unsafe_allow_html=True)

        st.markdown(latest_plan)

        # Download button
        st.download_button(
            label="⬇️ Download Study Plan",
            data=latest_plan,
            file_name=f"study_plan_{latest_subj.replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PDF SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>📄 PDF Summarizer</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        💡 Upload any PDF — notes, textbook chapter, syllabus —
        and AI will create a smart summary with key points and likely exam areas.
    </div>""", unsafe_allow_html=True)

    s_subject = st.text_input("📚 Subject",
                               placeholder="e.g. Organic Chemistry",
                               key="sum_subj")

    uploaded = st.file_uploader("📄 Upload PDF",
                                 type=["pdf"],
                                 key="sum_pdf")

    if uploaded:
        fsize = len(uploaded.getvalue()) / 1024
        st.markdown(f"""
        <div class='tip'>
            📄 <strong>{uploaded.name}</strong> · {fsize:.1f} KB uploaded
        </div>""", unsafe_allow_html=True)

        if st.button("🤖 Summarize This PDF", use_container_width=True):
            if not s_subject.strip():
                st.error("Please enter the subject name.")
            else:
                with st.spinner("📖 Reading PDF..."):
                    pdf_text, pdf_err = extract_pdf_text(uploaded)

                if pdf_err:
                    st.error(pdf_err)
                elif not pdf_text:
                    st.error("Could not extract text. Use a text-based PDF.")
                else:
                    with st.spinner("🧠 AI is summarizing your notes..."):
                        summary, err = summarize_pdf(pdf_text, s_subject)

                    if err:
                        st.error(f"Error: {err}")
                    else:
                        st.session_state.summary = summary
                        st.success("✅ Summary ready!")

    else:
        st.markdown("""
        <div class='upload-box'>
            <div style='font-size:2.5rem;margin-bottom:0.75rem'>📄</div>
            <div style='color:white;font-weight:600;margin-bottom:0.4rem'>
                Drop your PDF here
            </div>
            <div style='color:rgba(255,255,255,0.35);font-size:0.85rem'>
                Notes · Textbook chapters · Syllabus documents
            </div>
        </div>""", unsafe_allow_html=True)

    # Show summary
    if st.session_state.summary:
        st.markdown("<div class='section-title'>📋 AI Summary</div>",
                    unsafe_allow_html=True)
        st.markdown(st.session_state.summary)

        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="pdf_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MCQ QUIZ PAPER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>🧠 MCQ Question Paper Generator</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        💡 Upload your notes or textbook PDF — AI will generate a proper
        MCQ question paper with answer key and explanations.
    </div>""", unsafe_allow_html=True)

    # init quiz state
    if "questions"  not in st.session_state.quiz_state:
        st.session_state.quiz_state = {
            "questions": [], "answers": {},
            "submitted": False, "score": 0, "source": ""
        }

    qs = st.session_state.quiz_state

    q_subject = st.text_input("📚 Subject",
                               placeholder="e.g. Biology",
                               key="quiz_subj_inp")
    n_qs      = st.slider("Number of Questions", 3, 15, 8)
    mode      = st.radio("Mode",
                          ["Practice (see answers after submit)",
                           "Exam (no answers until end)"],
                          horizontal=True)

    quiz_pdf = st.file_uploader("📄 Upload PDF to generate questions from",
                                 type=["pdf"],
                                 key="quiz_pdf")

    if quiz_pdf:
        fsize = len(quiz_pdf.getvalue()) / 1024
        st.markdown(f"""
        <div class='tip'>
            📄 <strong>{quiz_pdf.name}</strong> · {fsize:.1f} KB
        </div>""", unsafe_allow_html=True)

        if st.button("🤖 Generate MCQ Paper from PDF",
                     use_container_width=True):
            if not q_subject.strip():
                st.error("Please enter the subject name.")
            else:
                with st.spinner("📖 Reading PDF..."):
                    pdf_text, pdf_err = extract_pdf_text(quiz_pdf)

                if pdf_err:
                    st.error(pdf_err)
                elif not pdf_text:
                    st.error("Empty PDF. Use a text-based PDF.")
                else:
                    with st.spinner(
                        f"🧠 Generating {n_qs} questions from your PDF..."
                    ):
                        questions, err = generate_mcq_paper(
                            pdf_text, q_subject, n_qs
                        )

                    if err:
                        st.error(f"Error: {err}")
                    elif questions:
                        st.session_state.quiz_state = {
                            "questions": questions,
                            "answers":   {},
                            "submitted": False,
                            "score":     0,
                            "source":    quiz_pdf.name,
                        }
                        st.success(
                            f"✅ {len(questions)} questions generated!"
                        )
                        st.rerun()

    else:
        st.markdown("""
        <div class='upload-box'>
            <div style='font-size:2.5rem;margin-bottom:0.75rem'>🧠</div>
            <div style='color:white;font-weight:600;margin-bottom:0.4rem'>
                Upload PDF to generate questions
            </div>
            <div style='color:rgba(255,255,255,0.35);font-size:0.85rem'>
                AI reads your material and creates exam-style MCQs
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Quiz UI ───────────────────────────────────────────────────────────────
    if qs.get("questions"):
        source = qs.get("source", "PDF")
        st.markdown(
            f"<div class='section-title'>"
            f"📝 Question Paper — {len(qs['questions'])} Questions"
            f"</div>",
            unsafe_allow_html=True)
        st.caption(f"Source: {source}")

        if not qs["submitted"]:
            with st.form("mcq_form"):
                for i, q in enumerate(qs["questions"]):
                    diff_color = {
                        "Easy":   "#6EE7B7",
                        "Medium": "#f59e0b",
                        "Hard":   "#f87171"
                    }.get(q.get("difficulty","Medium"), "#a78bfa")

                    st.markdown(f"""
                    <div class='mcq-card'>
                        <div style='display:flex;justify-content:space-between;
                                    align-items:flex-start;margin-bottom:8px'>
                            <div style='color:rgba(255,255,255,0.4);font-size:0.75rem'>
                                📌 {q.get('topic','')}
                            </div>
                            <span style='background:{diff_color}22;
                                  color:{diff_color};
                                  border:1px solid {diff_color}44;
                                  border-radius:99px;padding:2px 10px;
                                  font-size:0.72rem;font-weight:600'>
                                {q.get('difficulty','Medium')}
                            </span>
                        </div>
                        <div style='color:white;font-weight:600;font-size:1rem'>
                            Q{i+1}. {q['q']}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    qs["answers"][i] = st.radio(
                        f"Q{i+1}",
                        q["options"],
                        key=f"mcq_{i}",
                        label_visibility="collapsed"
                    )

                if st.form_submit_button("✅ Submit Paper",
                                          use_container_width=True):
                    score = sum(
                        1 for i, q in enumerate(qs["questions"])
                        if qs["answers"].get(i) == q["answer"]
                    )
                    st.session_state.quiz_state["score"]     = score
                    st.session_state.quiz_state["submitted"] = True
                    st.rerun()

        else:
            # ── Results ───────────────────────────────────────────────────────
            score = qs["score"]
            total = len(qs["questions"])
            pct   = score / total * 100
            color = ("#6EE7B7" if pct >= 75
                     else "#f59e0b" if pct >= 50
                     else "#f87171")
            emoji = "🎉" if pct >= 75 else "👍" if pct >= 50 else "📚"
            msg   = ("Excellent! Great preparation!" if pct >= 75
                     else "Good effort! Review wrong answers." if pct >= 50
                     else "Keep studying — re-read the PDF and try again!")

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,
                        rgba(79,110,245,0.12),rgba(110,231,183,0.08));
                 border:1px solid rgba(79,110,245,0.25);
                 border-radius:24px;padding:2.5rem;text-align:center;
                 margin:1rem 0'>
                <div style='font-size:3.5rem'>{emoji}</div>
                <div style='color:rgba(255,255,255,0.45);margin:0.5rem 0;
                            font-size:0.9rem'>Your Score</div>
                <div style='font-size:4rem;font-weight:800;color:{color};
                            line-height:1'>{score}/{total}</div>
                <div style='font-size:1.5rem;font-weight:700;color:{color};
                            margin-top:0.25rem'>{pct:.0f}%</div>
                <div style='color:rgba(255,255,255,0.5);margin-top:0.75rem'>
                    {msg}
                </div>
            </div>""", unsafe_allow_html=True)

            # Stats row
            easy_total  = sum(1 for q in qs["questions"]
                              if q.get("difficulty") == "Easy")
            med_total   = sum(1 for q in qs["questions"]
                              if q.get("difficulty") == "Medium")
            hard_total  = sum(1 for q in qs["questions"]
                              if q.get("difficulty") == "Hard")
            easy_score  = sum(
                1 for i, q in enumerate(qs["questions"])
                if q.get("difficulty") == "Easy"
                and qs["answers"].get(i) == q["answer"]
            )
            med_score   = sum(
                1 for i, q in enumerate(qs["questions"])
                if q.get("difficulty") == "Medium"
                and qs["answers"].get(i) == q["answer"]
            )
            hard_score  = sum(
                1 for i, q in enumerate(qs["questions"])
                if q.get("difficulty") == "Hard"
                and qs["answers"].get(i) == q["answer"]
            )

            sc1, sc2, sc3 = st.columns(3)
            for col, label, s, t, c in [
                (sc1, "🟢 Easy",   easy_score, easy_total, "#6EE7B7"),
                (sc2, "🟡 Medium", med_score,  med_total,  "#f59e0b"),
                (sc3, "🔴 Hard",   hard_score, hard_total, "#f87171"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='card' style='text-align:center'>
                        <div style='color:{c};font-weight:700;font-size:1.5rem'>
                            {s}/{t}
                        </div>
                        <div style='color:rgba(255,255,255,0.4);font-size:0.8rem'>
                            {label}
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Answer review
            st.markdown("<div class='section-title'>📋 Answer Review</div>",
                        unsafe_allow_html=True)

            for i, q in enumerate(qs["questions"]):
                user_ans    = qs["answers"].get(i, "")
                correct_ans = q["answer"]
                is_correct  = user_ans == correct_ans
                icon        = "✅" if is_correct else "❌"
                bc          = "#6EE7B7" if is_correct else "#f87171"
                diff_color  = {
                    "Easy":   "#6EE7B7",
                    "Medium": "#f59e0b",
                    "Hard":   "#f87171"
                }.get(q.get("difficulty","Medium"), "#a78bfa")

                wrong_html = "" if is_correct else f"""
                <div style='margin-top:6px;font-size:0.85rem'>
                    <span style='color:rgba(255,255,255,0.4)'>Correct: </span>
                    <span style='color:#6EE7B7;font-weight:600'>{correct_ans}</span>
                </div>"""

                explanation = q.get("explanation", "")
                exp_html = f"""
                <div style='margin-top:8px;background:rgba(79,110,245,0.08);
                     border-radius:8px;padding:0.6rem 0.8rem;
                     font-size:0.82rem;color:rgba(255,255,255,0.6)'>
                    💡 {explanation}
                </div>""" if explanation else ""

                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);
                     border:1px solid {bc}33;border-radius:14px;
                     padding:1.1rem 1.25rem;margin-bottom:0.75rem'>
                    <div style='display:flex;justify-content:space-between;
                                margin-bottom:6px'>
                        <span style='color:rgba(255,255,255,0.35);font-size:0.75rem'>
                            📌 {q.get('topic','')}
                        </span>
                        <span style='color:{diff_color};font-size:0.75rem;
                              font-weight:600'>
                            {q.get('difficulty','')}
                        </span>
                    </div>
                    <div style='color:white;font-weight:600;margin-bottom:8px'>
                        {icon} Q{i+1}. {q['q']}
                    </div>
                    <div style='font-size:0.85rem'>
                        <span style='color:rgba(255,255,255,0.4)'>Your answer: </span>
                        <span style='color:{bc};font-weight:500'>{user_ans}</span>
                    </div>
                    {wrong_html}
                    {exp_html}
                </div>""", unsafe_allow_html=True)

            # Action buttons
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state.quiz_state = {
                        "questions": qs["questions"],
                        "answers": {}, "submitted": False,
                        "score": 0, "source": qs["source"]
                    }
                    st.rerun()
            with b2:
                if st.button("📄 New PDF", use_container_width=True):
                    st.session_state.quiz_state = {
                        "questions": [], "answers": {},
                        "submitted": False, "score": 0, "source": ""
                    }
                    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;
     color:rgba(255,255,255,0.15);font-size:0.8rem'>
    StudyAI · Powered by Gemini AI · Built for students 🎓
</div>""", unsafe_allow_html=True)