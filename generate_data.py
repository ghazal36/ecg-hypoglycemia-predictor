# generate_data.py
import numpy as np
import pandas as pd
from scipy.signal import sawtooth
import os

def generate_ecg(duration_sec=60, fs=250, hr=60, noise_level=0.02, hrv_variability=0.1)
    t = np.arange(0, duration_sec, 1/fs)
    bpm = hr + (np.random.randn(len(t)) * 0.5) 
    f = hr / 60.0
    signal = 0.6 * np.sin(2*np.pi*f*t) + 0.05 * np.sin(2*np.pi*2*f*t) 
    rr_mean = 60.0 / hr
    times = []
    cur = 0.0
    while cur < duration_sec:
        rr = rr_mean * (1 + hrv_variability * (np.random.randn()*0.5))
        times.append(cur)
        cur += max(0.3, rr)
    for tt in times:
        idx = int(tt * fs)
        if idx < len(signal):
            w = int(0.03 * fs)
            for k in range(max(0, idx-w), min(len(signal), idx+w)):
                signal[k] += 1.0 * np.exp(-((k-idx)/(w/2))**2) 
    signal += noise_level * np.random.randn(len(signal))
    return t, signal

def create_dataset(out_csv="sample_ecg.csv", n_samples=6):
    rows = []
    labels = []
    os.makedirs("data_samples", exist_ok=True)
    for i in range(n_samples):
        if i < n_samples // 2:
            hr = np.random.randint(55, 75)
            noise = 0.02
            hrv = 0.05
            label = 0
        else:
            hr = np.random.randint(80, 110)
            noise = 0.03
            hrv = 0.15
            label = 1
        t, sig = generate_ecg(duration_sec=60, fs=250, hr=hr, noise_level=noise, hrv_variability=hrv)
        df = pd.DataFrame({"time": t, "value": sig})
        filename = f"data_samples/sample_{i+1}_label_{label}.csv"
        df.to_csv(filename, index=False)
        labels.append({"file": filename, "label": label})
    sample_df = pd.read_csv(labels[0]["file"])
    sample_df.to_csv(out_csv, index=False)
    pd.DataFrame(labels).to_csv("data_samples/labels.csv", index=False)
    print(f"Generated {n_samples} samples in data_samples/. Wrote {out_csv} as sample.")

if __name__ == "__main__":
    create_dataset(out_csv="sample_ecg.csv", n_samples=8)
