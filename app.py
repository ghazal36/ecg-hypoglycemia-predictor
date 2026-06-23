import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from ecg_utils import detect_r_peaks, compute_hr_hrv
import os

st.set_page_config(page_title="Oshahan - ECG Predictor", page_icon="🫀", layout="wide")
 
st.markdown("""
    <style>
    .main { background-color: #fafafa; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🫀 Oshahan ")
st.subheader("AI-Based Real-Time Hypoglycemia Prediction System via ECG Signals")
st.markdown("""
This application is an advanced academic demo designed to predict the probability of a sudden drop in blood glucose (Hypoglycemia) using non-invasive ECG signal analysis and machine learning.
""")
st.markdown("---")


model_path = "model.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.warning("Error loading the AI model: " + str(e))

st.sidebar.header("📥 Upload ECG Data")
uploaded_file = st.sidebar.file_uploader("Upload your ECG signal file (CSV format)", type=["csv"])
use_sample = st.sidebar.checkbox("Use Smart Demo Data", value=True)
 
if uploaded_file is None and not use_sample:
    st.info("💡 Please upload a CSV file containing ECG data or check 'Use Smart Demo Data' to proceed.")
    st.stop()
 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists("sample_ecg.csv"):
        df = pd.read_csv("sample_ecg.csv")
    else:
       
        t = np.arange(0, 60, 1/250)
        signal = 0.4 * np.sin(2 * np.pi * 1.1 * t)
        for p in range(1, 60):
            idx = int(p * 0.85 * 250)
            if idx < len(signal):
                signal[idx:idx+6] += 1.6    
        df = pd.DataFrame({"time": t, "value": signal})

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Live ECG Signal Visualization")
    fig, ax = plt.subplots(figsize=(10, 4.2))
   
    ax.plot(df["time"].iloc[:1000], df["value"].iloc[:1000], linewidth=1.2, color="crimson")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

fs = 250 
sig = df["value"].values
peaks = detect_r_peaks(sig, fs=fs)
features = compute_hr_hrv(peaks, fs=fs)

with col_right:
    st.subheader("🔬 Clinical Dashboard (HRV Biomarkers)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="❤️ Mean Heart Rate", value=f"{features['hr_mean']:.1f} BPM")
        st.metric(label="📊 SDNN (Variability)", value=f"{features['sdnn']:.4f} s")
    with c2:
        st.metric(label="🧬 RMSSD", value=f"{features['rmssd']:.4f} s")
        st.metric(label="📍 Detected Peaks", value=f"{features['n_peaks']} R-Peaks")

st.markdown("---")

if features["hr_mean"] == 0 or features["n_peaks"] < 3:
    st.error("❌ Invalid or highly noisy ECG signal. The system cannot reliably detect heartbeats. Please upload a higher quality signal.")
else:
    st.subheader("🎯 AI Analysis Result")
    if model is None:
        
        if features['hr_mean'] > 80:
            st.error("⚠️ **WARNING (Simulated):** High Risk of Hypoglycemia Detected! (Cardiac patterns indicate rapid heart rate and abnormal HRV stability)")
        else:
            st.success("✅ **NORMAL (Simulated):** No immediate risk of Hypoglycemia detected from current ECG biomarkers.")
    else:
        X = pd.DataFrame([features]).fillna(0)
        prob = model.predict_proba(X)[0][1] 
        pred = model.predict(X)[0]
        
        st.write(f"Hypoglycemia Risk Probability: **{prob*100:.1f}%**")
        
        if prob > 0.5 or pred == 1:
            st.error("🚨 **WARNING:** The AI system detected cardiac patterns associated with a high risk of Hypoglycemia!")
        else:
            st.success("✅ **NORMAL:** ECG biomarkers are stable. No significant indicators of low blood sugar detected.")

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🚀 Future Research & Academic Extensions (For Admission Reviewers)"):
    st.markdown("""
    - **Clinical Data Integration:** Transitioning from synthetic signals to real-world patient data via **PhysioNet (MIT-BIH & Clinical Databases)**.
    - **Advanced Noise Filtering:** Implementing digital Butterworth and Notch filters to clean real-world baseline wanders and powerline interference.
    - **Deep Learning Architectures:** Upgrading the core Random Forest classifier to Time-Series **LSTM (Long Short-Term Memory)** or **Transformer-based** neural networks for higher continuous prediction accuracy.
    """)

st.caption("⚠️ Disclaimer: This project is for educational and research purposes only. It is not intended for medical diagnosis or to replace clinical glucometers.")
