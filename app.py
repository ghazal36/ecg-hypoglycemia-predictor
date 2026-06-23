import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from ecg_utils import load_ecg_csv, detect_r_peaks, compute_hr_hrv
import os

# Page configuration
st.set_page_config(page_title="ECG Hypoglycemia Predictor", layout="centered")

st.title("🧠 AI-Based Hypoglycemia Prediction System")
st.markdown("""
This application is a demo designed to predict the probability of a sudden drop in blood glucose (Hypoglycemia) using ECG signal analysis and machine learning.
""")

# Load machine learning model
model_path = "model.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.warning("Error loading the AI model: " + str(e))

# Sidebar for user inputs
st.sidebar.header("📥 Upload ECG Data")
uploaded_file = st.sidebar.file_uploader("Upload your ECG signal file (CSV format)", type=["csv"])
use_sample = st.sidebar.checkbox("Use default sample data", value=True)

# Check upload status
if uploaded_file is None and not use_sample:
    st.info("💡 Please upload a CSV file containing ECG data or check 'Use default sample data' to proceed.")
    st.stop()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_ecg.csv")

# Plotting the raw ECG signal
st.subheader("📈 Raw ECG Signal Visualization")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df["time"], df["value"], linewidth=0.7, color="crimson")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Voltage Amplitude")
st.pyplot(fig)

# Signal processing and feature extraction
fs = 250 
sig = df["value"].values
peaks = detect_r_peaks(sig, fs=fs)
features = compute_hr_hrv(peaks, fs=fs)

st.subheader("🔬 Extracted Biomedical Features (HRV):")
st.write(features)

# AI Model Prediction Logic (Fix for invalid/noisy signals)
if features["hr_mean"] == 0 or features["n_peaks"] < 3:
    st.error("❌ Invalid or highly noisy ECG signal. The system cannot reliably detect heartbeats. Please upload a higher quality signal.")
else:
    if model is None:
        st.warning("⚠️ The model file (model.pkl) was not found. Prediction is unavailable.")
    else:
        X = pd.DataFrame([features]).fillna(0)
        prob = model.predict_proba(X)[0][1] 
        pred = model.predict(X)[0]
        
        st.subheader("🎯 AI Analysis Result:")
        st.write(f"Hypoglycemia Risk Probability: **{prob*100:.1f}%**")
        
        if prob > 0.5 or pred == 1:
            st.error("🚨 WARNING: The system detected cardiac patterns associated with a high risk of Hypoglycemia!")
        else:
            st.success("✅ NORMAL: No significant indicators of low blood sugar detected in the ECG signal.")

    st.caption("⚠️ Disclaimer: This project is for educational and research purposes only. It is not intended for medical diagnosis or to replace clinical glucometers.")
