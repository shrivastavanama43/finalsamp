# app.py
"""
Hypertension Detection Demo App (Creative Colorful Edition)
- Vibrant gradients, custom CSS, colorful textures
- Animated backgrounds, glassmorphism effects
- Smooth page transitions with colorful themes
- All pages fully functional with enhanced visual appeal
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

# Custom CSS for colorful creative design
CREATIVE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Animated gradient background */
    .main {
        background: linear-gradient(-45deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24, #f0932b, #eb4d4b);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin-bottom: 1.5rem;
    }
    
    /* Colored glass cards */
    .glass-green { background
