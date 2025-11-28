# app.py
"""
Hypertension Detection Demo App (corrected)
- Avoids accidental shadowing of `st`
- Uses safe experimental_rerun via fresh import
- Uses session_state flags instead of brittle reruns
- Demo-only: not medical advice
"""

import importlib
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

# Optional Twilio import (not required)
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# ---------- Config ----------
MODEL_PATH = "hypertension_demo_model.pkl"
DB_PATH = "users_demo.db"
DEBUG_ST = False  # Set True to show debug about streamlit object on the page

st.set_page_config(page_title="Hypertension Detection Demo", layout="centered")

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
# BMI and features
# -------------------------
def compute_bmi(weight_kg: float, height_cm: float):
    h_m = float(height_cm) / 100.0
    if h_m <= 0:
        return None
    bmi = float(weight_kg) / (h_m * h_m)
    return round(bmi, 2)


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
        print("[Reminder Job] It's time to take your medication / do yoga.")

    schedule.every().day.at("09:00").do(job)
    schedule.every().day.at("21:00").do(job)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()


# -------------------------
# Twilio helper (optional)
# -------------------------
def send_sms_via_twilio(to_phone, message, account_sid, auth_token, from_phone):
    if not TWILIO_AVAILABLE:
        raise RuntimeError("Twilio package not available. Install `twilio`.")
    client = TwilioClient(account_sid, auth_token)
    message = client.messages.create(body=message, from_=from_phone, to=to_phone)
    return message.sid


# -------------------------
# Pages (safe rerun & session flags)
# -------------------------
def safe_rerun_or_notice():
    """
    Attempt to call streamlit.experimental_rerun using a fresh import.
    If that fails, show a notice and rely on session_state flags.
    """
    try:
        _st = importlib.import_module("streamlit")
        if hasattr(_st, "experimental_rerun"):
            _st.experimental_rerun()
        else:
            st.info("Saved. Please refresh the page or navigate to continue.")
    except Exception:
        st.info("Saved. Please refresh the page or navigate to continue.")


def login_page():
    st.header("Welcome — Login / Register")

    if DEBUG_ST:
        st.write("DEBUG: type(st) =", type(st))
        st.write("DEBUG: 'experimental_rerun' in st dir?", "experimental_rerun" in dir(st))
        st.write("streamlit version:", getattr(st, "__version__", "unknown"))

    with st.form("login_form"):
        email = st.text_input("Email", value=st.session_state.get("email", ""))
        phone = st.text_input("Phone (optional)", value=st.session_state.get("phone", ""))
        name = st.text_input("Full name", value=st.session_state.get("name", ""))
        age = st.number_input("Age", min_value=10, max_value=120, value=int(st.session_state.get("age", 25)))
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
        submitted = st.form_submit_button("Continue")

    if submitted:
        st.session_state["email"] = email.strip().lower()
        st.session_state["phone"] = phone.strip()
        st.session_state["name"] = name.strip()
        st.session_state["age"] = int(age)
        st.session_state["gender"] = gender

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

        st.session_state["profile_saved"] = True
        safe_rerun_or_notice()


def intro_and_health_input():
    st.header("Intro & Quick Check-in")
    name = st.session_state.get("name", "Friend")
    st.subheader(f"Hey! {name} — how are you today?")
    st.write("This short check will ask a few questions about your health and lifestyle. Choose: Yes / No / Sometimes.")

    answers = {}
    with st.form("health_chat"):
        st.markdown("### Quick health questions")
        for q in QUESTIONS:
            key = f"q_{q['id']}"
            answers[q["id"]] = st.radio(q["text"], options=["no", "sometimes", "yes"], index=0, key=key, horizontal=True)
        st.markdown("### Measurements")
        weight = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=300.0, value=float(st.session_state.get("weight", 70.0))
        )
        height = st.number_input(
            "Height (cm)", min_value=80.0, max_value=250.0, value=float(st.session_state.get("height", 170.0))
        )
        submitted = st.form_submit_button("Save & Analyze")

    if submitted:
        for k, v in answers.items():
            st.session_state["q_" + k] = v
        st.session_state["weight"] = float(weight)
        st.session_state["height"] = float(height)
        bmi = compute_bmi(weight, height)
        st.session_state["bmi"] = bmi
        st.success("Saved measurements.")

        profile = {
            "email": st.session_state.get("email"),
            "phone": st.session_state.get("phone"),
            "name": st.session_state.get("name"),
            "age": st.session_state.get("age"),
            "gender": st.session_state.get("gender"),
            "weight": weight,
            "height": height,
            "bmi": bmi,
            "data": {"answers": answers},
        }
        save_user_profile(profile)

        st.session_state["profile_saved"] = True
        safe_rerun_or_notice()


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

    st.markdown(f"### Combined risk estimate: **{int(combined_prob * 100)}%**")

    st.markdown("## Prescription / Recommendations (editable)")
    default_rx = ""
    if combined_prob >= 0.7:
        default_rx = "High risk — consult a physician. Possible medications: ACE-inhibitor / ARB (as per doctor). Lifestyle: low salt, weight loss, regular exercise, stress management. Monitor BP twice daily."
    elif combined_prob >= 0.4:
        default_rx = "Moderate risk — discuss with doctor. Lifestyle: reduce salt, start daily brisk walking 30 mins, weight management, reduce alcohol and stop smoking."
    else:
        default_rx = "Low risk — continue healthy lifestyle: balanced low-salt diet, regular exercise, avoid smoking, maintain healthy weight."

    rx = st.text_area("Prescription / Advice", value=default_rx, height=180)
    if st.button("Save Prescription"):
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
                    "height": user.get("height"),
                    "bmi": user.get("bmi"),
                    "data": data,
                }
            )
            st.success("Prescription saved to profile.")


def generate_short_plan(user, answers):
    bmi = user.get("bmi", None)
    plan = []
    if bmi:
        if bmi >= 30:
            plan.append("Aim for weight loss: reduce 5-10% over 6 months.")
        elif bmi >= 25:
            plan.append("Light weight loss & activity: 150 min/week moderate exercise.")
        else:
            plan.append("Maintain weight with balanced diet.")
    plan.append("Reduce added salt; limit processed foods.")
    if answers.get("smoking") == "yes":
        plan.append("Stop smoking. Seek support.")
    if answers.get("alcohol") == "yes":
        plan.append("Limit or stop alcohol.")
    plan.append("Practice 20 min mindful breathing daily for stress.")
    plan.append("Monitor BP twice weekly and log readings.")
    return plan


def user_profile_page():
    st.header("Your Profile — Summary (short & precise)")
    user = load_user(st.session_state.get("email"))
    if not user:
        st.info("No profile found. Please login / register.")
        return
    st.subheader(f"{user.get('name')} — {user.get('email')}")
    st.write(f"Age: {user.get('age')}  •  Gender: {user.get('gender')}")
    st.write(f"Weight: {user.get('weight')} kg  •  Height: {user.get('height')} cm  •  BMI: {user.get('bmi')}")
    data = user.get("data", {})
    answers = data.get("answers", {})
    st.write("Recent answers (short):")
    for k, v in answers.items():
        st.write(f"- {k}: {v}")
    presc = data.get("last_prescription")
    if presc:
        st.write("Last prescription summary:")
        st.write(presc["text"][:500])
    else:
        st.write("No prescription saved yet.")

    st.markdown("### Diet & Lifestyle (short)")
    plan = generate_short_plan(user, answers)
    for item in plan:
        st.write(f"- {item}")


def reminders_page():
    st.header("Daily Reminders & Medication Schedule")
    st.write("This demo can show in-app reminders. For real SMS or call reminders, configure Twilio credentials below.")

    reminders = st.session_state.get(
        REMINDERS_KEY,
        [
            {"time": "09:00", "message": "Take BP tablet (if prescribed)"},
            {"time": "20:00", "message": "Evening: 20 min yoga (standing forward fold, cat-cow, child's pose)"},
        ],
    )
    for r in reminders:
        st.write(f"- {r['time']} — {r['message']}")

    with st.form("add_reminder"):
        t = st.time_input("Reminder time", value=datetime.now().replace(hour=9, minute=0).time())
        msg = st.text_input("Message", value="Take medication on time")
        added = st.form_submit_button("Add reminder")
    if added:
        reminders.append({"time": t.strftime("%H:%M"), "message": msg})
        st.session_state[REMINDERS_KEY] = reminders
        st.success("Reminder added (in-app)")

    st.markdown("### Twilio (optional) — send SMS / Call reminders")
    st.write("If you'd like the app to send SMS or make calls, provide Twilio credentials. This is optional and costs SMS/call credits.")
    use_twilio = st.checkbox("Enable Twilio reminders (requires credentials)", value=False)
    if use_twilio:
        sid = st.text_input("Twilio Account SID")
        token = st.text_input("Twilio Auth Token", type="password")
        from_phone = st.text_input("Twilio From Phone (E.164)")
        to_phone = st.text_input("Recipient Phone (E.164)", value=st.session_state.get("phone", ""))
        if st.button("Send test SMS now"):
            if not TWILIO_AVAILABLE:
                st.error("Twilio SDK not installed. Install `twilio` package.")
            else:
                try:
                    for r in reminders:
                        sid_resp = send_sms_via_twilio(to_phone, f"Reminder: {r['message']} at {r['time']}", sid, token, from_phone)
                    st.success("Test messages sent.")
                except Exception as e:
                    st.error(f"Failed to send SMS: {e}")


# -------------------------
# Main
# -------------------------
def main():
    st.title("Hypertension Detection — Demo App")

    if st.session_state.get("profile_saved"):
        st.success("Profile updated successfully.")
        st.session_state.pop("profile_saved", None)

    menu = ["Home / Login", "Intro & Health Input", "Risk & Prescription", "Profile", "Reminders"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if "email" not in st.session_state:
        st.session_state["email"] = ""
    if "name" not in st.session_state:
        st.session_state["name"] = ""

    if choice == "Home / Login":
        login_page()
    elif choice == "Intro & Health Input":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            intro_and_health_input()
    elif choice == "Risk & Prescription":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            risk_and_prescription()
    elif choice == "Profile":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            user_profile_page()
    elif choice == "Reminders":
        if not st.session_state.get("email"):
            st.info("Please login on the Home page first.")
        else:
            reminders_page()


if __name__ == "__main__":
    def main():
    st.title("Hypertension Detection — Demo App")

    # Initialise page if not set
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar navigation (manual override allowed)
    menu = ["Home", "Intro", "Risk", "Profile", "Reminders"]
    choice = st.sidebar.radio("Navigate", menu, index=menu.index(st.session_state.page))

    # Sync sidebar choice with session state
    if choice != st.session_state.page:
        st.session_state.page = choice

    # Render active page
    if st.session_state.page == "Home":
        login_page()
    elif st.session_state.page == "Intro":
        intro_and_health_input()
    elif st.session_state.page == "Risk":
        risk_and_prescription()
    elif st.session_state.page == "Profile":
        user_profile_page()
    elif st.session_state.page == "Reminders":
        reminders_page()

