# ecg_utils.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def load_ecg_csv(path):
    df = pd.read_csv(path)
    if "time" not in df.columns or "value" not in df.columns:
        raise ValueError("CSV must contain 'time' and 'value' columns.")
    return df["time"].values, df["value"].values

def detect_r_peaks(signal, fs=250, distance_sec=0.3):
    distance = int(distance_sec * fs)
    peaks, _ = find_peaks(signal, distance=distance, prominence=0.3)
    return peaks

def compute_hr_hrv(peaks_idx, fs=250):
    if len(peaks_idx) < 2:
        return {"hr_mean": 0, "sdnn": 0, "rmssd": 0, "nn50": 0, "pnn50": 0}
    times = np.array(peaks_idx) / fs
    rr = np.diff(times)
    hr_inst = 60.0 / rr
    hr_mean = np.mean(hr_inst)
    sdnn = np.std(rr)  # deviation of RR intervals
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    nn50 = np.sum(np.abs(np.diff(rr)) > 0.05)
    pnn50 = nn50 / max(1, len(rr)-1)
    features = {
        "hr_mean": float(hr_mean),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "nn50": int(nn50),
        "pnn50": float(pnn50),
        "n_peaks": int(len(peaks_idx))
    }
    return features

def extract_features_from_csv(path, fs=250):
    t, sig = load_ecg_csv(path)
    peaks = detect_r_peaks(sig, fs=fs)
    feats = compute_hr_hrv(peaks, fs=fs)
    return feats
