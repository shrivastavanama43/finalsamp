# app.py
"""
Hypertension Detection — guided linear flow edition
- Explicit Continue / Back navigation on each page
- Progress stepper at top
- Last-login persisted to last_login.json
- Defensive session_state initialization
- Demo-only: not medical advice
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict

import joblib
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Config ----------
MODEL_PATH = "hypertension_demo_model.pkl"
DB_PATH = "users_demo.db"
LAST_LOGIN_PATH = "last_login.json"

st.set_page_config(page_title="Hypertension - Flow", layout="wide")


# ---------- Helpers ----------
def safe_rerun_if_possible():
    """Call experimental rerun only if available to force immediate update."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass


def ensure_state_keys():
    """Create keys before any code tries to read them."""
    defaults = {
        "page": "Home",
        "email": "",
        "name": "",
        "phone": "",
        "age": 30,
        "gender": "Male",
        "weight": None,
        "height": None,
        "bmi": None,
        "notifications": [],
        "reminders": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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


def send_in_app_notification(title: str, message: str):
    if "notifications" not in st.session_state:
        st.session_state["notifications"] = []
    st.session_state["notifications"].append({"title": title, "message": message, "time": datetime.utcnow().isoformat()})
    st.info(f"{title}: {message}")


# ---------- Simple DB helpers ----------
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
            "INSERT INTO users (email, phone, name, age, gender, weight, height, bmi, last_visit, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile["email"], profile.get("phone"), profile["name"], profile["age"], profile["gender"], profile.get("weight"), profile.get("height"), profile.get("bmi"), now, data_json),
        )
    except sqlite3.IntegrityError:
        c.execute(
            "UPDATE users SET phone=?, name=?, age=?, gender=?, weight=?, height=?, bmi=?, last_visit=?, data=? WHERE email=?",
            (profile.get("phone"), profile["name"], profile["age"], profile["gender"], profile.get("weight"), profile.get("height"), profile.get("bmi"), now, data_json, profile["email"]),
        )
    DB_CONN.commit()


def load_user(email: str):
    if not email:
        return None
    c = DB_CONN.cursor()
    c.execute("SELECT email, phone, name, age, gender, weight, height, bmi, last_visit, data FROM users WHERE email=?", (email,))
    row = c.fetchone()
    if not row:
        return None
    email, phone, name, age, gender, weight, height, bmi, last_visit, data = row
    out = dict(email=email, phone=phone, name=name, age=age, gender=gender, weight=weight, height=height, bmi=bmi, last_visit=last_visit)
    try:
        out["data"] = json.loads(data) if data else {}
    except Exception:
        out["data"] = {}
    return out


# ---------- BMI & demo model (kept minimal) ----------
def compute_bmi(weight_kg: float, height_cm: float):
    try:
        h_m = float(height_cm) / 100.0
        if h_m <= 0:
            return None
        return round(float(weight_kg) / (h_m * h_m), 2)
    except Exception:
        return None


def train_demo_model(path=MODEL_PATH):
    # very small demo train so app is responsive
    n = 800
    rng = np.random.RandomState(42)
    ages = rng.randint(18, 80, size=n)
    genders = rng.choice([0, 1], size=n)
    bmis = rng.normal(26, 4, size=n).clip(15, 45)
    smoking = rng.binomial(1, 0.15, size=n)
    sedentary = rng.binomial(1, 0.4, size=n)
    logits = (ages > 50).astype(int) * 1.2 + (bmis > 30).astype(int) * 1.0 + smoking * 0.5 + sedentary * 0.7 + rng.normal(0, 0.8, size=n)
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > 0.5).astype(int)
    X = np.vstack([ages, genders, bmis, smoking, sedentary]).T
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=80, random_state=42))])
    pipe.fit(X, y)
    joblib.dump(pipe, path)
    return pipe


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return train_demo_model(path)
    else:
        return train_demo_model(path)


MODEL = load_model()


def model_predict_risk(age, gender, bmi, qanswers):
    try:
        fv = [age, 1 if str(gender).lower().startswith("m") else 0, bmi or 25, 1 if qanswers.get("smoking") == "yes" else 0, 1 if qanswers.get("sedentary") == "yes" else 0]
        p = MODEL.predict_proba([fv])[0][1]
        return round(float(p), 2)
    except Exception:
        return None


# ---------- Pages: linear flow controls ----------
PAGES = ["Home", "Intro", "Risk", "Profile", "Reminders"]


def top_flow_header():
    """Show a simple stepper and progress bar."""
    idx = PAGES.index(st.session_state.get("page", "Home"))
    cols = st.columns(len(PAGES))
    for i, name in enumerate(PAGES):
        with cols[i]:
            label = f"{i+1}. {name}" if i != idx else f"➡ {name}"
            st.caption(label)
    st.progress((idx + 1) / len(PAGES))


def nav_buttons(next_page=None, prev_page=None):
    """Render Back / Continue buttons consistently. Returns 'next'/'back' or None."""
    cols = st.columns([1, 1, 1])
    action = None
    with cols[0]:
        if prev_page and st.button("◀ Back"):
            st.session_state["page"] = prev_page
            action = "back"
    with cols[2]:
        if next_page and st.button("Continue ▶"):
            st.session_state["page"] = next_page
            action = "next"
    if action:
        # try to rerun so change is immediate in interactive environments
        safe_rerun_if_possible()
    return action


def page_home():
    st.header("Welcome — Login / Register (Home)")
    last = load_last_login()
    email_prefill = st.session_state.get("email") or last.get("email", "")
    name_prefill = st.session_state.get("name") or last.get("name", "")
    phone_prefill = st.session_state.get("phone") or last.get("phone", "")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", value=email_prefill, key="form_email")
        name = st.text_input("Full name", value=name_prefill, key="form_name")
        phone = st.text_input("Phone (optional)", value=phone_prefill, key="form_phone")
        age = st.number_input("Age", min_value=10, max_value=120, value=int(st.session_state.get("age", 30)), key="form_age")
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0, key="form_gender")
        submit = st.form_submit_button("Save & Continue")
    if submit:
        st.session_state["email"] = email.strip().lower()
        st.session_state["name"] = name.strip()
        st.session_state["phone"] = phone.strip()
        st.session_state["age"] = int(age)
        st.session_state["gender"] = gender
        save_last_login(st.session_state["email"], st.session_state["name"], st.session_state["phone"])
        # persist base profile to DB
        profile = {"email": st.session_state["email"], "phone": st.session_state.get("phone"), "name": st.session_state.get("name"), "age": st.session_state.get("age"), "gender": st.session_state.get("gender"), "weight": st.session_state.get("weight"), "height": st.session_state.get("height"), "bmi": st.session_state.get("bmi"), "data": {}}
        save_user_profile(profile)
        send_in_app_notification("Login", "Saved details — continue to Intro")
        st.session_state["page"] = "Intro"
        safe_rerun_if_possible()

    st.markdown("---")
    nav_buttons(next_page="Intro")


def page_intro():
    st.header("Intro & Quick Check-in")
    st.write("Answer a few lifestyle questions and add measurements.")

    # collect answers
    qkeys = ["smoking", "alcohol", "family_history", "sedentary", "headache", "stress", "salt", "sleep"]
    answers = {}
    with st.form("health_form"):
        for k in qkeys:
            default = st.session_state.get(f"q_{k}", "no")
            answers[k] = st.radio(k.replace("_", " ").title(), options=["no", "sometimes", "yes"], index={"no":0,"sometimes":1,"yes":2}.get(default,0), key=f"form_q_{k}", horizontal=True)
        # measurements
        w_default = float(st.session_state.get("weight") or 70.0)
        h_default = float(st.session_state.get("height") or 170.0)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=w_default, format="%.1f", key="form_weight")
        height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=h_default, format="%.1f", key="form_height")
        submit = st.form_submit_button("Save & Continue")

    if submit:
        for k, v in answers.items():
            st.session_state[f"q_{k}"] = v
        try:
            st.session_state["weight"] = float(weight)
            st.session_state["height"] = float(height)
        except Exception:
            st.session_state["weight"] = w_default
            st.session_state["height"] = h_default
        st.session_state["bmi"] = compute_bmi(st.session_state["weight"], st.session_state["height"])
        # update DB
        profile = {"email": st.session_state.get("email"), "phone": st.session_state.get("phone"), "name": st.session_state.get("name"), "age": st.session_state.get("age"), "gender": st.session_state.get("gender"), "weight": st.session_state.get("weight"), "height": st.session_state.get("height"), "bmi": st.session_state.get("bmi"), "data": {"answers": {k: st.session_state.get(f"q_{k}") for k in qkeys}}}
        save_user_profile(profile)
        send_in_app_notification("Check-in", "Saved answers & measurements")
        st.session_state["page"] = "Risk"
        safe_rerun_if_possible()

    st.markdown("---")
    nav_buttons(next_page="Risk", prev_page="Home")


def page_risk():
    st.header("Risk Assessment & Recommendations")
    # require login
    if not st.session_state.get("email"):
        st.warning("Please login on the Home page first")
        return

    age = st.session_state.get("age")
    gender = st.session_state.get("gender")
    bmi = st.session_state.get("bmi") or compute_bmi(st.session_state.get("weight") or 70, st.session_state.get("height") or 170)

    qanswers = {}
    for k in ["smoking", "alcohol", "family_history", "sedentary", "headache", "stress", "salt", "sleep"]:
        qanswers[k] = st.session_state.get(f"q_{k}", "no")

    st.subheader("Computed metrics")
    st.write(f"Age: {age}  •  Gender: {gender}  •  BMI: {bmi}")
    # rule-based quick scoring
    score = 0
    if age >= 60: score += 3
    elif age >= 45: score += 2
    elif age >= 30: score += 1
    if bmi >= 30: score += 3
    elif bmi >= 25: score += 2
    for k in ["smoking", "alcohol", "family_history", "sedentary", "stress"]:
        if qanswers.get(k) == "yes": score += 2
        elif qanswers.get(k) == "sometimes": score += 1
    max_score = 3 + 3 + (5 * 2)
    rule_prob = min(0.99, score / (max_score or 1))

    model_prob = model_predict_risk(age, gender, bmi, qanswers)
    combined = (model_prob + rule_prob) / 2 if (model_prob is not None) else rule_prob
    st.metric("Combined risk", f"{int(combined*100)}%")
    if combined >= 0.7:
        st.error("High risk — consult a doctor")
    elif combined >= 0.4:
        st.warning("Moderate risk — consider seeing a doctor")
    else:
        st.success("Low risk — maintain healthy lifestyle")

    rx_default = "Follow healthy diet, monitor BP, exercise regularly."
    rx = st.text_area("Recommendation (editable)", value=rx_default, height=150, key="rx_text")
    if st.button("Save Recommendation"):
        user = load_user(st.session_state.get("email"))
        if user:
            data = user.get("data", {})
            data["last_prescription"] = {"text": rx, "timestamp": datetime.utcnow().isoformat()}
            save_user_profile({**user, "data": data})
            send_in_app_notification("Recommendation saved", "Prescription saved to profile")

    st.markdown("---")
    nav_buttons(next_page="Profile", prev_page="Intro")


def page_profile():
    st.header("Profile Summary")
    if not st.session_state.get("email"):
        st.info("Login first on Home")
        return
    user = load_user(st.session_state.get("email"))
    if not user:
        st.info("No saved profile found yet")
        return
    st.write(f"**Name:** {user.get('name')}")
    st.write(f"**Email:** {user.get('email')}")
    st.write(f"**Phone:** {user.get('phone')}")
    st.write(f"**Age:** {user.get('age')}  •  Gender: {user.get('gender')}")
    st.write(f"**Weight:** {user.get('weight')} kg  •  Height: {user.get('height')} cm  •  BMI: {user.get('bmi')}")
    st.markdown("### Last prescription")
    last = user.get("data", {}).get("last_prescription")
    if last:
        st.write(last.get("text"))
        st.caption(last.get("timestamp"))
    else:
        st.write("No prescription saved yet.")
    st.markdown("---")
    nav_buttons(next_page="Reminders", prev_page="Risk")


def page_reminders():
    st.header("Reminders")
    reminders = st.session_state.get("reminders", [])
    for r in reminders:
        st.write(f"- {r['time']} — {r['message']}")
    with st.form("add_rem"):
        t = st.time_input("Time", value=datetime.now().replace(hour=21, minute=0).time(), key="rem_time")
        msg = st.text_input("Message", value="Take medication", key="rem_msg")
        add = st.form_submit_button("Add")
    if add:
        reminders.append({"time": t.strftime("%H:%M"), "message": msg})
        st.session_state["reminders"] = reminders
        send_in_app_notification("Reminder", f"Added {t.strftime('%H:%M')} — {msg}")
    st.markdown("---")
    nav_buttons(prev_page="Profile")


# ---------- Main ----------
def main():
    ensure_state_keys()
    top_flow_header()

    # sidebar with quick nav + notifications
    with st.sidebar:
        st.header("Quick navigation")
        page = st.radio("Jump to", PAGES, index=PAGES.index(st.session_state.get("page", "Home")))
        if page != st.session_state["page"]:
            st.session_state["page"] = page
            safe_rerun_if_possible()

        st.markdown("---")
        st.subheader("Notifications")
        for n in reversed(st.session_state.get("notifications", [])[-6:]):
            st.caption(f"{n['time'].split('T')[-1][:8]} — {n['title']}")
        st.markdown("---")
        if st.button("Reset flow (start over)"):
            st.session_state["page"] = "Home"
            # optional: clear profile/session keys (uncomment if desired)
            # for k in ["email","name","phone","age","gender","weight","height","bmi","reminders"]:
            #     st.session_state.pop(k, None)
            safe_rerun_if_possible()

    # route to current page
    page = st.session_state.get("page", "Home")
    if page == "Home":
        page_home()
    elif page == "Intro":
        page_intro()
    elif page == "Risk":
        page_risk()
    elif page == "Profile":
        page_profile()
    elif page == "Reminders":
        page_reminders()


if __name__ == "__main__":
    main()
