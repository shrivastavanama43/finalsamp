# app.py
"""
Hypertension Detection Demo App (patched)
- Twilio removed
- Simple in-app notifications
- Last-login persisted to last_login.json (prefills login until file removed)
- Smooth navigation: safe rerun helper used instead of direct st.experimental_rerun()
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
DEBUG_ST = False  # toggle for debug info about streamlit object

st.set_page_config(page_title="Hypertension Detection Demo", layout="centered")


# -------------------------
# Safe rerun helper (works if experimental_rerun is missing)
# -------------------------
def safe_rerun():
    """
    Try immediate rerun if supported; otherwise stop the script.
    Using st.stop() as a fallback avoids AttributeError crashes while still
    allowing session_state changes to take effect on the next user interaction.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.stop()
    except Exception:
        st.stop()


# -------------------------
# Utilities / DB
# -------------------------
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
        out["data"] = json.loads(data)
    except Exception:
        out["data"] = {}
    return out


# -------------------------
# Last-login persistence (small JSON file)
# -------------------------
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


# -------------------------
# Simple in-app notifications
# (demo) stored in session_state and shown in sidebar
# -------------------------
def init_notifications():
    if "notifications" not in st.session_state:
        st.session_state["notifications"] = []


def send_in_app_notification(title: str, message: str):
    init_notifications()
    now = datetime.utcnow().isoformat()
    st.session_state["notifications"].append({"title": title, "message": message, "time": now})
    # immediate alert in main area for instant feedback
    st.info(f"{title}: {message}")


# -------------------------
# BMI and features
# -------------------------
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


# -------------------------
# Rule-based risk
# -------------------------
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


# -------------------------
# Demo ML model
# -------------------------
def train_demo_model(path=MODEL_PATH):
    n = 3000
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
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    joblib.dump(pipe, path)
    return pipe, auc


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
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
    p = float(MODEL.predict_proba([fv])[0][1])
    return round(p, 2)


# -------------------------
# Questions
# -------------------------
QUESTIONS = [
    {"id": "smoking", "text": "Do you smoke?"},
    {"id": "alcohol", "text": "Do you consume alcohol regularly?"},
    {"id": "family_history", "text": "Any family history of hypertension?"},
    {"id": "sedentary", "text": "Do you lead a mostly sedentary lifestyle?"},
    {"id": "headache", "text": "Do you often have headaches or dizziness?"},
    {"id": "stress", "text": "Do you frequently feel stressed or anxious?"},
    {"id": "salt", "text": "Do you eat a lot of salty foods? (processed / extra salt)"},
    {"id": "sleep", "text": "Do you have trouble sleeping? (insomnia, poor sleep)"},
]


# -------------------------
# Reminders (demo)
# -------------------------
REMINDERS_KEY = "reminders"


def schedule_in_app_reminders():
    def job():
        # demo-only: create an in-app notification
        send_in_app_notification("Reminder", "It's time to take your medication / do yoga.")

    schedule.every().day.at("09:00").do(job)
    schedule.every().day.at("21:00").do(job)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()


# -------------------------
# Pages & UI
# -------------------------
def login_page():
    st.header("Welcome — Login / Register")

    if DEBUG_ST:
        st.write("DEBUG: type(st) =", type(st))
        st.write("DEBUG: 'experimental_rerun' in st dir?", "experimental_rerun" in dir(st))
        st.write("streamlit version:", getattr(st, "_version_", "unknown"))

    # prefill from last-login file if available (only on first render)
    last = load_last_login()
    email_prefill = st.session_state.get("email", last.get("email", ""))
    phone_prefill = st.session_state.get("phone", last.get("phone", ""))
    name_prefill = st.session_state.get("name", last.get("name", ""))

    with st.form("login_form"):
        email = st.text_input("Email", value=email_prefill, key="login_email")
        phone = st.text_input("Phone (optional)", value=phone_prefill, key="login_phone")
        name = st.text_input("Full name", value=name_prefill, key="login_name")
        age = st.number_input("Age", min_value=10, max_value=120, value=int(st.session_state.get("age", 25)), key="login_age")
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0, key="login_gender")
        submitted = st.form_submit_button("Continue")

    if submitted:
        # Save into session_state
        st.session_state["email"] = email.strip().lower()
        st.session_state["phone"] = phone.strip()
        st.session_state["name"] = name.strip()
        st.session_state["age"] = int(age)
        st.session_state["gender"] = gender

        # persist last-login to disk so login stays prefilled across sessions until file removed
        save_last_login(st.session_state["email"], st.session_state["name"], st.session_state["phone"])

        st.success(f"Logged in as {st.session_state['name']} ({st.session_state['email']})")

        user = load_user(st.session_state["email"])
        if user:
            st.info("Loaded existing profile.")
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

        # Smooth navigation: set page then rerun via safe helper
        st.session_state["page"] = "Intro"
        send_in_app_notification("Login", "Redirecting to Intro page...")
        safe_rerun()


def intro_and_health_input():
    st.header("Intro & Quick Check-in")
    name = st.session_state.get("name", "Friend")
    st.subheader(f"Hey! {name} — how are you today?")
    st.write("This short check will ask a few questions about your health and lifestyle. Choose: No / Sometimes / Yes.")

    answers = {}

    # helper to map stored answer to radio index
    def answer_to_index(ans):
        if not ans:
            return 0
        ans = str(ans).lower()
        return {"no": 0, "sometimes": 1, "yes": 2}.get(ans, 0)

    with st.form("health_chat"):
        st.markdown("### Quick health questions")
        for q in QUESTIONS:
            key = f"q_{q['id']}_radio"
            stored = st.session_state.get(f"q_{q['id']}", "no")
            answers[q["id"]] = st.radio(
                q["text"],
                options=["no", "sometimes", "yes"],
                index=answer_to_index(stored),
                key=key,
                horizontal=True,
            )

        st.markdown("### Measurements")
        # safe defaults (avoid float(None))
        weight_default = float(st.session_state.get("weight") or 70.0)
        height_default = float(st.session_state.get("height") or 170.0)

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=weight_default,
            format="%.1f",
            key="input_weight",
        )
        height = st.number_input(
            "Height (cm)",
            min_value=80.0,
            max_value=250.0,
            value=height_default,
            format="%.1f",
            key="input_height",
        )

        submitted = st.form_submit_button("Save & Analyze")

    if submitted:
        # persist answers into session_state
        for k, v in answers.items():
            st.session_state["q_" + k] = v

        # store numeric values safely
        try:
            st.session_state["weight"] = float(weight)
        except Exception:
            st.session_state["weight"] = weight_default

        try:
            st.session_state["height"] = float(height)
        except Exception:
            st.session_state["height"] = height_default

        st.session_state["bmi"] = compute_bmi(st.session_state["weight"], st.session_state["height"])
        st.success("Saved measurements.")

        profile = {
            "email": st.session_state.get("email"),
            "phone": st.session_state.get("phone"),
            "name": st.session_state.get("name"),
            "age": st.session_state.get("age"),
            "gender": st.session_state.get("gender"),
            "weight": st.session_state.get("weight"),
            "height": st.session_state.get("height"),
            "bmi": st.session_state.get("bmi"),
            "data": {"answers": answers},
        }
        save_user_profile(profile)

        # navigate to Risk page (smooth)
        st.session_state["page"] = "Risk"
        send_in_app_notification("Check-in", "Measurements saved. Redirecting to Risk page...")
        safe_rerun()


def risk_and_prescription():
    st.header("Risk Assessment & Prescription")
    age = st.session_state.get("age")
    gender = st.session_state.get("gender")
    bmi = st.session_state.get("bmi")
    if bmi is None:
        st.info("Please complete measurements on the previous page first.")
        return

    qanswers = {}
    for q in QUESTIONS:
        qanswers[q["id"]] = st.session_state.get("q_" + q["id"], "no")

    st.subheader("Computed metrics")
    st.write(f"Age: {age}  •  Gender: {gender}  •  BMI: {bmi}")
    st.write("Key answers:")
    st.write(qanswers)

    rule_prob, rule_score = rule_based_risk(age, bmi, qanswers)
    st.metric("Rule-based Risk (probability)", f"{int(rule_prob * 100)}%")
    st.write(f"(Internal score: {rule_score})")

    model_prob = model_predict_risk(age, gender, bmi, qanswers)
    if model_prob is not None:
        st.metric("Model-based Risk (probability)", f"{int(model_prob * 100)}%")
    else:
        st.info("No ML model available; using rule-based scoring only.")

    combined_prob = model_prob if model_prob is not None else rule_prob
    if model_prob is not None:
        combined_prob = round((rule_prob + model_prob) / 2, 2)

    st.markdown(f"### Combined risk estimate: *{int(combined_prob * 100)}%*")

    st.markdown("## Prescription / Recommendations (editable)")
    default_rx = ""
    if combined_prob >= 0.7:
        default_rx = "High risk — consult a physician. Possible medications: ACE-inhibitor / ARB (as per doctor). Lifestyle: low salt, weight loss, regular exercise, stress management. Monitor BP twice daily."
    elif combined_prob >= 0.4:
        default_rx = "Moderate risk — discuss with doctor. Lifestyle: reduce salt, start daily brisk walking 30 mins, weight management, reduce alcohol and stop smoking."
    else:
        default_rx = "Low risk — continue healthy lifestyle: balanced low-salt diet, regular exercise, avoid smoking, maintain healthy weight."

    rx = st.text_area("Prescription / Advice", value=default_rx, height=180, key="prescription_text")
    if st.button("Save Prescription", key="save_prescription_btn"):
        user = load_user(st.session_state.get("email"))
        if user:
            data = user.get("data", {})
            data["last_prescription"] = {"text": rx, "timestamp": datetime.utcnow().isoformat()}
            user["data"] = data
            save_user_profile(
                {
                    "email": user["email"],
                    "phone": user["phone"],
                    "name": user["name"],
                    "age": user["age"],
                    "gender": user["gender"],
                    "weight": user.get("weight"),
                    "height": user.get("hei
