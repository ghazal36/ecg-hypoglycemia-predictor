import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from ecg_utils import load_ecg_csv, detect_r_peaks, compute_hr_hrv
import os

st.set_page_config(page_title="ECG Hypoglycemia Predictor (Demo)", layout="centered")

st.title("ECG-based Hypoglycemia Prediction — Demo")
st.markdown()

# Load model if exists
model_path = "model.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.warning("" + str(e))

st.sidebar.header("")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])
use_sample = st.sidebar.checkbox("", value=True)

if uploaded_file is None and not use_sample:
    st.info()
    st.stop()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_ecg.csv")

st.subheader("")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df["time"], df["value"], linewidth=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
st.pyplot(fig)
fs = 250 
sig = df["value"].values
peaks = detect_r_peaks(sig, fs=fs)
features = compute_hr_hrv(peaks, fs=fs)

st.subheader("")
st.write(features)

if model is None:
    st.warning("")
else:
    X = pd.DataFrame([features]).fillna(0)
    prob = model.predict_proba(X)[0][1] 
    pred = model.predict(X)[0]
    st.subheader("")
    st.write(f" **{prob:.2f}**")
    if prob > 0.5 or pred == 1:
        st.error("")
    else:
        st.success()

    st.caption("")
