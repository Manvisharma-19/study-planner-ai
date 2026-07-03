# 🎓 StudyAI — AI-Powered Exam Preparation Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_AI-000000?style=for-the-badge&logo=groq)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> An end-to-end AI-powered study planning platform that generates
> personalized study plans, summarizes PDFs, predicts performance
> and creates MCQ question papers — all powered by LLM + ML.

---

## 🌐 Live Demo

🔗 **[studyai.streamlit.app](https://study-planner-ai-manvisharma.streamlit.app/)**

---

## ✨ Features

### 📅 AI Study Plan Generator

- Enter your subject, exam date and syllabus
- AI generates a complete **day-by-day study plan**
- Includes revision strategy and exam day tips
- Download your plan as a text file

### 📄 PDF Summarizer

- Upload any PDF (notes, textbook chapter, syllabus)
- AI extracts and summarizes key concepts
- Outputs: key topics, important concepts, quick revision
  points, formulas and likely exam areas
- Download summary as text file

### 🧠 MCQ Question Paper Generator

- Upload your study material PDF
- AI generates exam-style MCQ questions
- Questions tagged with difficulty (Easy / Medium / Hard)
- Full answer review with explanations after submission
- Score breakdown by difficulty level

### 🤖 ML Performance Predictor

- Predicts expected exam score using Random Forest
- Classifies at-risk students
- Feature importance visualization
- Study hours impact simulator

### 📊 Analytics Dashboard

- Past vs predicted score comparison
- Topic completion tracking
- Priority score per subject
- Interactive charts

---

## 🧱 Project Structure

study-planner-ai/
│
├── app/
│ └── app.py # Main Streamlit application
│
├── src/
│ ├── data_pipeline.py # Data loading, cleaning, normalization
│ ├── feature_engineering.py # ML feature creation
│ ├── performance_model.py # ML model training (RF, LR)
│ └── planner.py # Smart scheduling algorithm
│
├── data/ # Data directory
├── models/ # Saved ML models
├── notebooks/ # Jupyter notebooks
├── .streamlit/
│ └── config.toml # Streamlit theme config
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- Groq API key (free at https://console.groq.com)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/study-planner-ai.git
cd study-planner-ai
```

**2. Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up API keys**

Create `.streamlit/secrets.toml`:

```toml
[groq]
api_key = "your_groq_api_key_here"
```

**5. Run the app**

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧰 Tech Stack

| Layer           | Technology                                      |
| --------------- | ----------------------------------------------- |
| Frontend        | Streamlit                                       |
| AI / LLM        | Groq (LLaMA 3.3 70B)                            |
| ML Models       | scikit-learn (Random Forest, Linear Regression) |
| Data Processing | Pandas, NumPy                                   |
| PDF Processing  | pdfplumber, PyPDF2                              |
| Visualization   | Matplotlib                                      |
| Deployment      | Streamlit Cloud                                 |

---

## 🧠 ML Architecture

### Data Pipeline

- Synthetic student data generation (200 students)
- Missing value handling, outlier clipping
- MinMax normalization, Label encoding

### Feature Engineering

- Urgency score = difficulty / (days_left + 1)
- Performance gap = target − past_score
- Study efficiency = score × consistency / (difficulty × hours + 1)
- Interaction features: productive hours, hard work remaining

### Models

| Model                    | Task                      | Metric       |
| ------------------------ | ------------------------- | ------------ |
| Random Forest Regressor  | Predict exam score        | MAE, R²      |
| Linear Regression        | Baseline score prediction | MAE, R²      |
| Random Forest Classifier | At-risk classification    | F1, Accuracy |
| Logistic Regression      | Baseline classification   | F1, Accuracy |

### Smart Scheduler

- Priority-based hour allocation
- Deadline boost (+20% if exam ≤ 5 days)
- Adaptive re-planning on progress update

---

## 🚀 Deployment

Deployed on **Streamlit Cloud** for free hosting.

Steps to deploy your own:

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file as `app/app.py`
5. Add Groq API key in Secrets settings
6. Click Deploy

Any push to GitHub **auto-deploys** to Streamlit Cloud.

---

## 🔐 Security Notes

- Never commit API keys to GitHub
- Always store keys in `.streamlit/secrets.toml`
- Add `secrets.toml` to `.gitignore`
- Use Streamlit Cloud Secrets for production keys

---

## 🎤 Interview Talking Points

> _"I built an end-to-end AI study platform that combines LLM-powered
> features (study plan generation, PDF summarization, MCQ creation)
> with traditional ML models (Random Forest for performance prediction)
> and a hybrid rule-based scheduling algorithm — deployed on Streamlit Cloud
> with CI/CD via GitHub."_

Key areas to explain:

- **LLM Integration** — Groq API with LLaMA 3.3 for text generation
- **ML Pipeline** — feature engineering, model training, inference
- **PDF Processing** — pdfplumber + PyPDF2 fallback
- **Deployment** — Streamlit Cloud with GitHub auto-deploy
- **Security** — secret management, .gitignore

---

## 📸 Screenshots

| Feature        | Description                   |
| -------------- | ----------------------------- |
| 📅 Study Plan  | Day-by-day AI generated plan  |
| 📄 PDF Summary | Smart notes summarization     |
| 🧠 MCQ Paper   | Exam-style questions from PDF |
| 📊 Analytics   | ML predictions and charts     |

---

## 🤝 Contributing

Pull requests are welcome!

1. Fork the repo
2. Create your branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License — free to use and modify.

---

## 👩‍💻 Author

**Manvi Sharma**

- GitHub: [@Manvisharma-19](https://github.com/Manvisharma-19)
- Email: manvisharma5189@gmail.com

---

<div align="center">
    Made with ❤️ for students · Powered by AI 🎓
</div>
