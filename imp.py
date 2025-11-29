# app.py
"""
Hypertension Detection Demo App (Creative Visual Edition)
- Twilio removed, Simple in-app notifications
- Last-login persisted to last_login.json
- Smooth navigation with safe rerun helper
- CREATIVE VISUAL: CSS, glassmorphism cards, animated metrics
- Demo-only: not medical advice
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import schedule
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Config ----------
MODEL_PATH = "hypertension_demo_model.pkl"
DB_PATH = "users_demo.db"
LAST_LOGIN_PATH = "last_login.json"
DEBUG_ST = False

# single page config call
st.set_page_config(page_title="🩺 Hypertension Guardian", layout="wide")

# ---------- Creative CSS (closed properly) ----------
CREATIVE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
        padding-top: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.25rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        margin: 0.75rem 0;
    }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        animation: pulse-glow 3s infinite;
    }
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 14px rgba(255,107,107,0.18); }
        50% { box-shadow: 0 0 22px rgba(78,205,196,0.24); }
        100% { box-shadow: 0 0 14px rgba(255,107,107,0.18); }
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        color: white;
        transition: all 0.18s ease;
    }
    .stButton > button:hover { transform: translateY(-3px); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        backdrop-filter: blur(8px);
    }
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: rgba(255,255,255,0.95);
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.06);
        padding: 0.6rem;
        font-family: 'Poppins', sans-serif;
    }
</style>
"""
st.markdown(CREATIVE_CSS, unsafe_allow_html=True)

# ---------- Small helper: safe_rerun (does not halt app) ----------
def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            # no immediate rerun; allow normal control flow to continue
            return
    except Exception:
        return

# ---------- Utilities / DB ----------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               email TEXT UNIQUE,
               phone TEXT,
               name TEXT,
               age INTEGER,
               gender TEXT,
               weight REAL,
               height REAL,
               bmi REAL,
               last_visit TEXT,
               data JSON
           )"""
    )
    conn.commit()
    return conn

DB_CONN = init_db()

def save_user_profile(profile: Dict):
    c = DB_CONN.cursor()
    now = datetime.utcnow().isoformat()
    data_json = json.dumps(profile.get("data", {}))
    try:
        c.execute(
            """
            INSERT INTO users (email, phone, name, age, gender, weight, height, bmi, last_visit, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile["email"],
                profile.get("phone"),
                profile["name"],
                profile["age"],
                profile["gender"],
                profile.get("weight"),
                profile.get("height"),
                profile.get("bmi"),
                now,
                data_json,
            ),
        )
    except sqlite3.IntegrityError:
        c.execute(
            """
            UPDATE users SET phone=?, name=?, age=?, gender=?, weight=?, height=?, bmi=?, last_visit=?, data=?
            WHERE email=?
            """,
            (
                profile.get("phone"),
                profile["name"],
                profile["age"],
                profile["gender"],
                profile.get("weight"),
                profile.get("height"),
                profile.get("bmi"),
                now,
                data_json,
                profile["email"],
            ),
        )
    DB_CONN.commit()

def load_user(email: str):
    if not email:
        return None
    c = DB_CONN.cursor()
    c.execute(
        "SELECT email, phone, name, age, gender, weight, height, bmi, last_visit, data FROM users WHERE email=?",
        (email,),
    )
    row = c.fetchone()
    if not row:
        return None
    email, phone, name, age, gender, weight, height, bmi, last_visit, data = row
    out = dict(
        email=email,
        phone=phone,
        name=name,
        age=age,
        gender=gender,
        weight=weight,
        height=height,
        bmi=bmi,
        last_visit=last_visit,
    )
    try:
        out["data"] = json.loads(data) if data else {}
    except Exception:
        out["data"] = {}
    return out

# ---------- Last-login persistence ----------
def save_last_login(email: str, name: str = "", phone: str = ""):
    try:
        with open(LAST_LOGIN_PATH, "w", encoding="utf-8") as f:
            json.dump({"email": email, "name": name, "phone": phone}, f)
    except Exception:
        pass

def load_last_login():
    if os.path.exists(LAST_LOGIN_PATH):
        try:
            with open(LAST_LOGIN_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# ---------- Notifications ----------
def init_notifications():
    if "notifications" not in st.session_state:
        st.session_state["notifications"] = []

def send_in_app_notification(title: str, message: str, icon: str = "🔔"):
    init_notifications()
    now = datetime.utcnow().isoformat()
    st.session_state["notifications"].append({"title": title, "message": message, "time": now, "icon": icon})
    st.success(f"{icon} {title}: {message}")

# ---------- BMI and demo ML ----------
def compute_bmi(weight_kg: float, height_cm: float):
    try:
        h_m = float(height_cm) / 100.0
        if h_m <= 0:
            return None
        bmi = float(weight_kg) / (h_m * h_m)
        return round(bmi, 2)
    except Exception:
        return None

def create_feature_vector(age, gender, bmi, qanswers):
    gender_num = 1 if str(gender).lower().startswith("m") else 0
    smoking = 1 if qanswers.get("smoking") == "yes" else 0
    alcohol = 1 if qanswers.get("alcohol") == "yes" else 0
    family_history = 1 if qanswers.get("family_history") == "yes" else 0
    sedentary = 1 if qanswers.get("sedentary") == "yes" else 0
    headache = 1 if qanswers.get("headache") == "yes" else 0
    stress = 1 if qanswers.get("stress") == "yes" else 0
    return [age, gender_num, bmi, smoking, alcohol, family_history, sedentary, headache, stress]

def rule_based_risk(age, bmi, qanswers):
    score = 0
    if age >= 60:
        score += 3
    elif age >= 45:
        score += 2
    elif age >= 30:
        score += 1
    if bmi is not None:
        if bmi >= 30:
            score += 3
        elif bmi >= 25:
            score += 2
        elif bmi >= 23:
            score += 1
    for k in ["smoking", "alcohol", "family_history", "sedentary", "stress"]:
        if qanswers.get(k) == "yes":
            score += 2
        elif qanswers.get(k) == "sometimes":
            score += 1
    if qanswers.get("headache") == "yes":
        score += 1
    max_score = 3 + 3 + (5 * 2) + 1
    prob = min(0.99, score / max_score)
    return round(prob, 2), score

def train_demo_model(path=MODEL_PATH):
    n = 1500
    rng = np.random.RandomState(42)
    ages = rng.randint(18, 85, size=n)
    genders = rng.choice([0, 1], size=n)
    bmis = rng.normal(loc=26, scale=5, size=n).clip(15, 45)
    smoking = rng.binomial(1, 0.15, size=n)
    alcohol = rng.binomial(1, 0.2, size=n)
    family = rng.binomial(1, 0.12, size=n)
    sedentary = rng.binomial(1, 0.4, size=n)
    headache = rng.binomial(1, 0.25, size=n)
    stress = rng.binomial(1, 0.35, size=n)

    logits = (
        (ages > 50).astype(int) * 1.2
        + (bmis > 30).astype(int) * 1.0
        + family * 1.2
        + smoking * 0.6
        + sedentary * 0.7
        + rng.normal(0, 0.8, size=n)
    )
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > 0.5).astype(int)

    X = np.vstack([ages, genders, bmis, smoking, alcohol, family, sedentary, headache, stress]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, preds)
    except Exception:
        auc = 0.5
    joblib.dump(pipe, path)
    return pipe, auc

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    else:
        model, auc = train_demo_model(path)
        return model

MODEL = load_model()

def model_predict_risk(age, gender, bmi, qanswers):
    if MODEL is None:
        return None
    fv = create_feature_vector(age, gender, bmi, qanswers)
    try:
        p = float(MODEL.predict_proba([fv])[0][1])
    except Exception:
        p = 0.0
    return round(p, 2)

# ---------- Questions ----------
QUESTIONS = [
    {"id": "smoking", "text": "Do you smoke?", "emoji": "🚬"},
    {"id": "alcohol", "text": "Do you consume alcohol regularly?", "emoji": "🍺"},
    {"id": "family_history", "text": "Any family history of hypertension?", "emoji": "👪"},
    {"id": "sedentary", "text": "Do you lead a mostly sedentary lifestyle?", "emoji": "🛋️"},
    {"id": "headache", "text": "Do you often have headaches or dizziness?", "emoji": "🤕"},
    {"id": "stress", "text": "Do you frequently feel stressed or anxious?", "emoji": "😰"},
    {"id": "salt", "text": "Do you eat a lot of salty foods?", "emoji": "🧂"},
    {"id": "sleep", "text": "Do you have trouble sleeping?", "emoji": "😴"},
]

# ---------- Pages ----------
def login_page():
    st.markdown('<div class="glass-card" style="max-width:900px; margin:0 auto;">', unsafe_allow_html=True)
    st.header("🚀 Welcome — Start Your Health Journey")
    last = load_last_login()
    email_prefill = st.session_state.get("email", last.get("email", ""))
    phone_prefill = st.session_state.get("phone", last.get("phone", ""))
    name_prefill = st.session_state.get("name", last.get("name", ""))

    col1, col2 = st.columns([1, 1])
    with col1:
        email = st.text_input("Email", value=email_prefill, key="login_email")
        name = st.text_input("Full name", value=name_prefill, key="login_name")
        age = st.number_input("Age", min_value=10, max_value=120, value=int(st.session_state.get("age", 25)), key="login_age")
    with col2:
        phone = st.text_input("Phone (optional)", value=phone_prefill, key="login_phone")
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0, key="login_gender")

    submitted = st.button("Continue to Health Check", key="login_submit")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        st.session_state["email"] = email.strip().lower()
        st.session_state["phone"] = phone.strip()
        st.session_state["name"] = name.strip()
        st.session_state["age"] = int(age)
        st.session_state["gender"] = gender

        save_last_login(st.session_state["email"], st.session_state["name"], st.session_state["phone"])

        user = load_user(st.session_state["email"])
        if user:
            for k, v in user.items():
                if k in ["weight", "height", "bmi", "age", "phone", "name", "gender"]:
                    st.session_state[k] = v

        profile = {
            "email": st.session_state["email"],
            "phone": st.session_state.get("phone"),
            "name": st.session_state.get("name"),
            "age": st.session_state.get("age"),
            "gender": st.session_state.get("gender"),
            "weight": st.session_state.get("weight"),
            "height": st.session_state.get("height"),
            "bmi": st.session_state.get("bmi"),
            "data": {},
        }
        save_user_profile(profile)

        st.session_state["page"] = "Intro"
        send_in_app_notification("Welcome", f"Hello {st.session_state['name']}!", "🎉")
        safe_rerun()

def intro_and_health_input():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Quick Health Check-in")
    name = st.session_state.get("name", "Friend")
    st.markdown(f"### Hey {name} — how are you today?")

    col1, col2 = st.columns(2)
    with col1:
        weight_default = float(st.session_state.get("weight") or 70.0)
        height_default = float(st.session_state.get("height") or 170.0)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=weight_default, format="%.1f", key="input_weight")
        height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=height_default, format="%.1f", key="input_height")
        bmi = compute_bmi(weight, height)
        if bmi:
            st.success(f"BMI: {bmi}")
    with col2:
        answers = {}
        for q in QUESTIONS:
            key = f"q_{q['id']}_radio"
            stored = st.session_state.get(f"q_{q['id']}", "no")
            answers[q["id"]] = st.radio(q["text"], options=["no", "sometimes", "yes"], index={"no":0,"sometimes":1,"yes":2}.get(stored,0), key=key, horizontal=True)

    submitted = st.button("Save & Analyze", key="analyze_submit")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        for k, v in answers.items():
            st.session_state["q_" + k] = v
        st.session_state["weight"] = float(weight)
        st.session_state["height"] = float(height)
        st.session_state["bmi"] = bmi
        profile = {
            "email": st.session_state.get("email"),
            "phone": st.session_state.get("phone"),
            "name": st.session_state.get("name"),
            "age": st.session_state.get("age"),
            "gender": st.session_state.get("gender"),
            "weight": st.session_state["weight"],
            "height": st.session_state["height"],
            "bmi": st.session_state["bmi"],
            "data": {"answers": answers},
        }
        save_user_profile(profile)
        st.session_state["page"] = "Risk"
        send_in_app_notification("Check-in", "Measurements saved. Redirecting to Risk page...", "📊")
        safe_rerun()

def risk_and_prescription():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Risk Assessment & Recommendations")
    age = st.session_state.get("age")
    gender = st.session_state.get("gender")
    bmi = st.session_state.get("bmi")
    if bmi is None:
        st.warning("Please complete measurements first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    qanswers = {q["id"]: st.session_state.get("q_" + q["id"], "no") for q in QUESTIONS}
    rule_prob, rule_score = rule_based_risk(age, bmi, qanswers)
    model_prob = model_predict_risk(age, gender, bmi, qanswers)
    combined_prob = model_prob if model_prob is not None else rule_prob
    if model_prob is not None:
        combined_prob = round((rule_prob + model_prob) / 2, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Age", age)
    col2.metric("Gender", gender)
    col3.metric("BMI", f"{bmi:.1f}" if bmi else "N/A")

    st.markdown(f"### Combined risk estimate: **{int(combined_prob*100)}%**")
    default_rx = ""
    if combined_prob >= 0.7:
        default_rx = "High risk — consult a physician. Lifestyle: low salt, weight loss, regular exercise, stress management."
        st.error("HIGH RISK — seek medical advice")
    elif combined_prob >= 0.4:
        default_rx = "Moderate risk — consider medical consultation and lifestyle changes."
        st.warning("Moderate risk")
    else:
        default_rx = "Low risk — continue healthy lifestyle."

    rx = st.text_area("Prescription / Advice", value=default_rx, height=150, key="prescription_text")
    if st.button("Save Prescription", key="save_prescription_btn"):
        user = load_user(st.session_state.get("email"))
        if user:
            data = user.get("data", {})
            data["last_prescription"] = {"text": rx, "timestamp": datetime.utcnow().isoformat()}
            save_user_profile({
                "email": user["email"], "phone": user["phone"], "name": user["name"],
                "age": user["age"], "gender": user["gender"], "weight": user.get("weight"),
                "height": user.get("height"), "bmi": user.get("bmi"), "data": data
            })
            st.success("Prescription saved to profile.")
            send_in_app_notification("Prescription saved", "Your prescription was saved to your profile.", "💾")
    st.markdown('</div>', unsafe_allow_html=True)

def user_profile_page():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Your Profile")
    user = load_user(st.session_state.get("email"))
    if not user:
        st.info("No profile found. Please login and complete the health check.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {user.get('name')}")
        st.write(f"**Email:** {user.get('email')}")
        st.write(f"**Age:** {user.get('age')}")
        st.write(f"**Phone:** {user.get('phone')}")
    with col2:
        bmi = user.get("bmi")
        try:
            st.metric("BMI", f"{bmi:.1f}" if bmi else "N/A")
        except Exception:
            st.metric("BMI", "N/A")
    if user.get("data", {}).get("answers"):
        st.markdown("### Recent Answers")
        for k, v in list(user["data"]["answers"].items())[:6]:
            st.write(f"- {k}: {v}")
    st.markdown('</div>', unsafe_allow_html=True)

def reminders_page():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Reminders")
    reminders = st.session_state.get("reminders", [
        {"time": "09:00", "message": "Take BP medication", "emoji": "💊"},
        {"time": "20:00", "message": "Evening walk", "emoji": "🚶"},
    ])
    for r in reminders:
        st.write(f"{r['emoji']} **{r['time']}** — {r['message']}")
    t = st.time_input("Reminder time", value=datetime.now().replace(hour=21, minute=0).time(), key="rem_time")
    msg = st.text_input("Message", value="Stay hydrated", key="rem_msg")
    if st.button("Add reminder", key="add_rem"):
        reminders.append({"time": t.strftime("%H:%M"), "message": msg, "emoji": "🔔"})
        st.session_state["reminders"] = reminders
        st.success("Reminder added")
        send_in_app_notification("New reminder", f"{t.strftime('%H:%M')} — {msg}", "⏰")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main ----------
def main():
    init_notifications()
    with st.sidebar:
        st.markdown('<div style="padding:0.5rem 0.75rem; border-radius:10px;">🔔 Notifications</div>', unsafe_allow_html=True)
        if st.session_state["notifications"]:
            for n in reversed(st.session_state["notifications"][-6:]):
                st.caption(f"{n['time'].split('T')[1][:5]} — {n['title']}")
        else:
            st.caption("No notifications yet")

        st.markdown("---")
        page = st.radio("Navigate", ["Home", "Intro", "Risk", "Profile", "Reminders"], index=0 if "page" not in st.session_state else ["Home","Intro","Risk","Profile","Reminders"].index(st.session_state.get("page","Home")))
        # sync page selection
        if page != st.session_state.get("page", "Home"):
            st.session_state["page"] = page
            safe_rerun()

    # ensure minimal session keys
    if "email" not in st.session_state:
        st.session_state["email"] = ""
    if "name" not in st.session_state:
        st.session_state["name"] = ""

    # route pages
    if st.session_state["page"] == "Home":
        login_page()
    elif st.session_state["page"] == "Intro":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            intro_and_health_input()
    elif st.session_state["page"] == "Risk":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            risk_and_prescription()
    elif st.session_state["page"] == "Profile":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            user_profile_page()
    elif st.session_state["page"] == "Reminders":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            reminders_page()

if __name__ == "__main__":
    main()
