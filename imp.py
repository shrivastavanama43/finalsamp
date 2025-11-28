# app.py
"""
Hypertension Detection Demo App — final UI + persistent login
- All original logic preserved (DB, model, scoring, reminders, prescriptions)
- No sidebar; navigation via Continue / Back buttons
- CSS animations + inline SVG visuals
- Persistent login: saves last login in last_login.json
- All forms include a submit button; nav buttons outside forms
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, Optional

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

# Optional Twilio import
try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# ---------- Config ----------
MODEL_PATH = "hypertension_demo_model.pkl"
DB_PATH = "users_demo.db"
LAST_LOGIN_PATH = "last_login.json"
DEBUG_ST = False

st.set_page_config(page_title="Hypertension Detection — Demo", layout="centered", page_icon="💓")

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


def load_user(email: str) -> Optional[Dict]:
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


def save_last_login(email: str):
    try:
        with open(LAST_LOGIN_PATH, "w") as f:
            json.dump({"email": email}, f)
    except Exception:
        pass


def load_last_login() -> Optional[str]:
    try:
        if os.path.exists(LAST_LOGIN_PATH):
            with open(LAST_LOGIN_PATH, "r") as f:
                d = json.load(f)
                return d.get("email")
    except Exception:
        pass
    return None


def clear_last_login():
    try:
        if os.path.exists(LAST_LOGIN_PATH):
            os.remove(LAST_LOGIN_PATH)
    except Exception:
        pass


# -------------------------
# BMI and model utilities
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
# Questions & reminders
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


def send_sms_via_twilio(to_phone, message, account_sid, auth_token, from_phone):
    if not TWILIO_AVAILABLE:
        raise RuntimeError("Twilio package not available. Install `twilio`.")
    client = TwilioClient(account_sid, auth_token)
    message = client.messages.create(body=message, from_=from_phone, to=to_phone)
    return message.sid


# -------------------------
# UI: CSS + visuals
# -------------------------
def inject_global_css():
    css = r"""
    <style>
    :root{
        --bg:#070707;
        --card:#0f1720;
        --muted:#9aa6b2;
        --accent1:#00d9ff;
        --accent2:#ff6ec7;
        --accent3:#7bff8c;
        --glass: rgba(255,255,255,0.04);
    }
    .stApp {
      background: radial-gradient(ellipse at top left, rgba(0,217,255,0.03), transparent 20%),
                  radial-gradient(ellipse at bottom right, rgba(255,110,199,0.03), transparent 20%),
                  var(--bg) !important;
      color: #e6eef6;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.03);
      transition: transform 300ms cubic-bezier(.2,.9,.25,1), box-shadow 300ms;
      margin-bottom: 18px;
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(0,0,0,0.7); }
    .input-anim { animation: floatIn 420ms ease forwards; opacity: 0; transform: translateY(8px) scale(0.995); }
    @keyframes floatIn { to { opacity: 1; transform: translateY(0) scale(1); } }
    .btn-primary { background: linear-gradient(90deg,var(--accent1), var(--accent2)); color: #04111a; font-weight:700; padding:10px 18px; border-radius:10px; border:none; box-shadow:0 6px 18px rgba(0,0,0,0.6); cursor:pointer; transition: transform 160ms, box-shadow 160ms; }
    .btn-primary:hover { transform: translateY(-3px); box-shadow:0 20px 36px rgba(0,0,0,0.7); }
    .btn-ghost { background: transparent; color: var(--accent1); border:1px solid rgba(255,255,255,0.06); padding:8px 14px; border-radius:8px; }
    .steps { display:flex; gap:14px; align-items:center; margin-bottom:12px; }
    .step { padding:8px 12px; border-radius:999px; font-weight:700; font-size:14px; color:#071322; background: linear-gradient(90deg,var(--accent2),var(--accent1)); box-shadow:0 6px 18px rgba(0,0,0,0.6); }
    .step-inactive { padding:8px 12px; border-radius:999px; color:var(--muted); background:transparent; border:1px solid rgba(255,255,255,0.03); }
    .visual { width:100%; display:flex; justify-content:center; margin-bottom: 8px; }
    .bubbles { position: absolute; left:0; right:0; top:0; bottom:0; pointer-events:none; opacity:0.08; }
    .bubble { width:100px; height:100px; border-radius:50%; background: radial-gradient(circle at 30% 30%, rgba(255,110,199,0.7), rgba(0,217,255,0.2)); filter: blur(18px); animation: floaty 8s linear infinite; }
    .bubble.b2 { left:70%; top:10%; animation-duration:12s; background: radial-gradient(circle at 30% 30%, rgba(123,255,140,0.6), rgba(0,217,255,0.12)); }
    @keyframes floaty { 0% { transform: translateY(0) } 50% { transform: translateY(-30px) scale(1.02) } 100% { transform: translateY(0) } }
    .draw path { stroke-dasharray: 400; stroke-dashoffset: 400; animation: dash 1.6s ease forwards; }
    @keyframes dash { to { stroke-dashoffset: 0; } }
    .rx-grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; align-items:start; }
    .rx-list { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:8px; }
    @media (max-width: 720px) { .rx-grid { grid-template-columns: 1fr; } }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_steps_header():
    page = st.session_state.get("page", "Home")
    def mk_step(name, active):
        cls = "step" if active else "step-inactive"
        return f'<div class="{cls}">{name}</div>'
    steps_html = f"""
    <div class="steps">
      {mk_step("1. Login", page=="Home")}
      {mk_step("2. Intro", page=="Intro")}
      {mk_step("3. Risk", page=="Risk")}
      {mk_step("4. Profile", page=="Profile")}
      {mk_step("5. Reminders", page=="Reminders")}
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)


def svg_bp_visual():
    svg = r'''
    <div class="visual card input-anim" style="padding:10px; margin-bottom:6px; display:flex; justify-content:center;">
      <svg width="220" height="120" viewBox="0 0 220 120" xmlns="http://www.w3.org/2000/svg" class="draw">
        <defs>
         <linearGradient id="g1" x1="0" x2="1">
           <stop offset="0" stop-color="#00d9ff"/>
           <stop offset="1" stop-color="#ff6ec7"/>
         </linearGradient>
        </defs>
        <g transform="translate(10,10)">
          <rect x="120" y="15" rx="8" ry="8" width="70" height="80" fill="#071722" stroke="#00d9ff" stroke-opacity="0.7" />
          <rect x="130" y="25" rx="6" ry="6" width="50" height="60" fill="#081820" stroke="#ff6ec7" stroke-opacity="0.6" />
          <path d="M110 60 C100 55, 85 55, 70 60" stroke="url(#g1)" stroke-width="4" fill="none" />
          <path d="M60 40
                   C60 25, 35 25, 35 40
                   C35 55, 60 70, 60 70
                   C60 70, 85 55, 85 40
                   C85 25, 60 25, 60 40 Z"
                fill="#ff6ec7" stroke="#fff5" stroke-width="0.6" />
        </g>
      </svg>
    </div>
    '''
    st.markdown(svg, unsafe_allow_html=True)


# -------------------------
# Pages (forms with submit; nav buttons outside forms)
# -------------------------
def login_page():
    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.subheader("Welcome — Login / Register")
    st.markdown('</div>', unsafe_allow_html=True)

    svg_bp_visual()

    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    with st.form("login_form_ui"):
        email = st.text_input("Email", value=st.session_state.get("email", ""), key="login_email_ui")
        phone = st.text_input("Phone (optional)", value=st.session_state.get("phone", ""), key="login_phone_ui")
        name = st.text_input("Full name", value=st.session_state.get("name", ""), key="login_name_ui")
        age = st.number_input("Age", min_value=10, max_value=120, value=int(st.session_state.get("age", 25)), key="login_age_ui")
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0, key="login_gender_ui")
        submitted = st.form_submit_button("Continue", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
        # persist last login
        save_last_login(st.session_state["email"])

        st.session_state["page"] = "Intro"
        st.stop()


def intro_and_health_input():
    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.subheader(f"Hey! {st.session_state.get('name','Friend')} — quick health check")
    st.write("Choose: No / Sometimes / Yes. Your answers are saved locally in this demo.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="bubbles"><div class="bubble"></div><div class="bubble b2"></div></div>', unsafe_allow_html=True)

    # FORM: inputs + form_submit_button only
    with st.form("health_form_ui"):
        st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
        st.markdown("### Quick health questions")
        answers = {}
        for q in QUESTIONS:
            key = f"q_{q['id']}_ui"
            stored = st.session_state.get(f"q_{q['id']}", "no")
            idx = {"no": 0, "sometimes": 1, "yes": 2}.get(str(stored).lower(), 0)
            answers[q["id"]] = st.radio(q["text"], options=["no", "sometimes", "yes"], index=idx, key=key, horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
        st.markdown("### Measurements")
        weight_default = float(st.session_state.get("weight") or 70.0)
        height_default = float(st.session_state.get("height") or 170.0)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=weight_default, format="%.1f", key="ui_weight")
        height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=height_default, format="%.1f", key="ui_height")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Save & Continue", use_container_width=True)

    # Back button outside the form (prevents missing submit button error)
    nav_cols = st.columns([1, 1, 1])
    with nav_cols[0]:
        if st.button("Back", key="intro_back_outside"):
            st.session_state["page"] = "Home"
            st.stop()

    if submitted:
        for k, v in answers.items():
            st.session_state["q_" + k] = v
        try:
            st.session_state["weight"] = float(weight)
        except Exception:
            st.session_state["weight"] = weight_default
        try:
            st.session_state["height"] = float(height)
        except Exception:
            st.session_state["height"] = height_default
        st.session_state["bmi"] = compute_bmi(st.session_state["weight"], st.session_state["height"])

        st.success("Measurements saved.")
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

        st.session_state["page"] = "Risk"
        st.stop()


def risk_and_prescription():
    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.subheader("Risk Assessment & Prescription")
    st.markdown('</div>', unsafe_allow_html=True)

    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown(
            """
            <div class="card input-anim" style="display:flex;justify-content:center;align-items:center;">
              <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="44" fill="none" stroke="rgba(0,217,255,0.06)" stroke-width="10"/>
                <defs><linearGradient id="g"><stop offset="0" stop-color="#00d9ff"/><stop offset="1" stop-color="#ff6ec7"/></linearGradient></defs>
                <g transform="translate(35,35)">
                  <path d="M20 18 C20 6, 2 6, 2 18 C2 30, 20 36, 20 48 C20 36, 38 30, 38 18 C38 6, 20 6, 20 18 Z" fill="#ff6ec7" transform-origin="19 28">
                    <animate attributeName="transform" dur="1.4s" values="scale(1); scale(1.06); scale(1)" repeatCount="indefinite" />
                  </path>
                </g>
              </svg>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        age = st.session_state.get("age")
        gender = st.session_state.get("gender")
        bmi = st.session_state.get("bmi")
        if bmi is None:
            st.info("Please complete measurements on the previous page first.")
            return
        qanswers = {q["id"]: st.session_state.get("q_" + q["id"], "no") for q in QUESTIONS}
        st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
        st.write(f"**Age:** {age}  •  **Gender:** {gender}  •  **BMI:** {bmi}")
        st.write("Key answers (short):")
        st.write(", ".join([f"{k}: {v}" for k, v in qanswers.items()]))
        rule_prob, rule_score = rule_based_risk(age, bmi, qanswers)
        model_prob = model_predict_risk(age, gender, bmi, qanswers)
        st.metric("Rule-based Risk", f"{int(rule_prob*100)}%")
        if model_prob is not None:
            st.metric("Model-based Risk", f"{int(model_prob*100)}%")
            combined_prob = round((rule_prob + model_prob) / 2, 2)
        else:
            combined_prob = rule_prob
        st.markdown(f"### Combined estimate: **{int(combined_prob*100)}%**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Prescription details
    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.markdown("## Prescription — Detailed")
    default_rx = ""
    if combined_prob >= 0.7:
        default_rx = "High risk — consult a physician promptly. Lifestyle: immediate salt reduction, supervised medication as per doctor, monitor BP twice daily, keep appointment within 1 week."
    elif combined_prob >= 0.4:
        default_rx = "Moderate risk — discuss with doctor. Lifestyle: reduce salt, 30 min moderate exercise daily, weight management, avoid smoking/alcohol."
    else:
        default_rx = "Low risk — maintain healthy lifestyle: balanced diet, regular activity, routine BP checks."

    med_name = st.text_input("Medication (if prescribed)", value="", key="med_name")
    med_dose = st.text_input("Dose & timing (e.g., 5 mg — morning)", value="", key="med_dose")
    rx = st.text_area("Prescription / Advice (editable)", value=default_rx, height=140, key="rx_text")
    followup = st.selectbox("Suggested follow-up", options=["2 weeks", "1 month", "3 months", "As needed"], index=1, key="followup_select")

    st.markdown("### Medication schedule")
    if "med_schedule" not in st.session_state:
        st.session_state["med_schedule"] = []
    ms_cols = st.columns([3, 2, 1])
    with ms_cols[0]:
        new_med = st.text_input("Medicine name", key="ms_name")
    with ms_cols[1]:
        new_time = st.time_input("Time", key="ms_time")
    with ms_cols[2]:
        if st.button("Add", key="ms_add"):
            if new_med:
                st.session_state["med_schedule"].append({"name": new_med, "time": new_time.strftime("%H:%M")})
                st.success("Added to schedule")

    if st.session_state.get("med_schedule"):
        st.table(pd.DataFrame(st.session_state["med_schedule"]))

    cols_btn = st.columns([1, 1, 1])
    with cols_btn[0]:
        if st.button("Back", key="risk_back"):
            st.session_state["page"] = "Intro"
            st.stop()
    with cols_btn[2]:
        if st.button("Save Prescription & Continue", key="risk_save_continue"):
            user = load_user(st.session_state.get("email"))
            if user:
                data = user.get("data", {})
                data["last_prescription"] = {
                    "text": rx,
                    "med_name": med_name,
                    "med_dose": med_dose,
                    "med_schedule": st.session_state.get("med_schedule", []),
                    "followup": followup,
                    "timestamp": datetime.utcnow().isoformat(),
                }
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
            st.session_state["page"] = "Profile"
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)


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
    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.subheader("Your Profile — Summary")
    st.markdown('</div>', unsafe_allow_html=True)

    user = load_user(st.session_state.get("email"))
    if not user:
        st.info("No profile found. Please login / register.")
        return

    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown(
            """
            <div class="card input-anim" style="display:flex;justify-content:center;align-items:center;">
              <svg width="120" height="100" viewBox="0 0 120 100">
                <rect x="10" y="10" width="100" height="60" rx="10" fill="rgba(0,217,255,0.04)"/>
                <circle cx="40" cy="40" r="6" fill="#ff6ec7"/><circle cx="60" cy="40" r="6" fill="#00d9ff"/>
                <path d="M20 70 L100 70" stroke="#7bff8c" stroke-width="3" stroke-linecap="round"/>
              </svg>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.write(f"**{user.get('name')}**  •  {user.get('email')}")
        st.write(f"Age: {user.get('age')}  •  Gender: {user.get('gender')}")
        st.write(f"Weight: {user.get('weight')} kg  •  Height: {user.get('height')} cm  •  BMI: {user.get('bmi')}")

    data = user.get("data", {})
    answers = data.get("answers", {})

    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.write("### Recent answers (short):")
    if answers:
        for k, v in answers.items():
            st.write(f"- **{k}**: {v}")
    else:
        st.write("No answers recorded yet.")

    presc = data.get("last_prescription")
    if presc:
        st.markdown("### Last prescription (summary)")
        with st.expander("View prescription details"):
            st.write("**Advice:**")
            st.write(presc.get("text", ""))
            st.write("**Medicine:**", presc.get("med_name", "—"))
            st.write("**Dose & timing:**", presc.get("med_dose", "—"))
            ms = presc.get("med_schedule", [])
            if ms:
                st.write("**Schedule:**")
                st.table(pd.DataFrame(ms))
            st.write("**Follow-up:**", presc.get("followup", "—"))
    else:
        st.write("No saved prescription yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.markdown("### Diet & Lifestyle (short & precise)")
    plan = generate_short_plan(user, answers)
    for item in plan:
        st.write(f"- {item}")
    st.markdown('</div>', unsafe_allow_html=True)

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Back to Risk", key="profile_back"):
            st.session_state["page"] = "Risk"
            st.stop()
    with cols[1]:
        if st.button("Go to Reminders", key="profile_next"):
            st.session_state["page"] = "Reminders"
            st.stop()


def reminders_page():
    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.subheader("Daily Reminders & Medication Schedule")
    st.write("In-app reminders are shown here. Configure Twilio to send SMS/call reminders (optional).")
    st.markdown('</div>', unsafe_allow_html=True)

    reminders = st.session_state.get(
        REMINDERS_KEY,
        [
            {"time": "09:00", "message": "Take BP tablet (if prescribed)"},
            {"time": "20:00", "message": "Evening: 20 min yoga (standing forward fold, cat-cow, child's pose)"},
        ],
    )

    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    for r in reminders:
        st.write(f"- **{r['time']}** — {r['message']}")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("add_reminder_ui"):
        t = st.time_input("Reminder time", value=datetime.now().replace(hour=9, minute=0).time(), key="rem_t")
        msg = st.text_input("Message", value="Take medication on time", key="rem_msg")
        added = st.form_submit_button("Add reminder")
    if added:
        reminders.append({"time": t.strftime("%H:%M"), "message": msg})
        st.session_state[REMINDERS_KEY] = reminders
        st.success("Reminder added (in-app)")

    st.markdown('<div class="card input-anim">', unsafe_allow_html=True)
    st.markdown("### Twilio (optional)")
    use_twilio = st.checkbox("Enable Twilio reminders (requires credentials)", value=False, key="use_tw_ui")
    if use_twilio:
        sid = st.text_input("Twilio Account SID", key="tw_sid_ui")
        token = st.text_input("Twilio Auth Token", type="password", key="tw_token_ui")
        from_phone = st.text_input("Twilio From Phone (E.164)", key="tw_from_ui")
        to_phone = st.text_input("Recipient Phone (E.164)", value=st.session_state.get("phone", ""), key="_"
