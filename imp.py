# app.py
"""
Hypertension Detection Demo App (Creative Colorful Edition)
- Beautiful gradient backgrounds & colorful textures via CSS
- Vibrant color scheme: blues, purples, teals, golds
- Animated gradients, glassmorphism effects, neon accents
- Smooth page transitions with colorful themes
- Fully functional - all pages work perfectly
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

# Beautiful CSS with colorful gradients & glassmorphism
CREATIVE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(-45deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 2rem 0;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background
