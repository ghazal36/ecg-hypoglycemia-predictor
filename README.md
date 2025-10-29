# ECG-Based Hypoglycemia Prediction System

This project is an AI-based system designed to **predict hypoglycemia (low blood glucose levels)** using only **ECG (Electrocardiogram) signals**.  
The goal is to develop a **non-invasive, real-time early warning system** that can alert users before a dangerous drop in blood glucose occurs.

---

## 🧠 Project Overview

Traditional hypoglycemia detection requires invasive glucose sensors.  
However, studies have shown that certain ECG patterns — such as heart rate variability (HRV) and QRS complex changes — can indicate an upcoming hypoglycemic event.

This project uses **machine learning** to analyze ECG data and predict whether a person is at risk of hypoglycemia.

---

## ⚙️ How It Works

1. **Data Generation**  
   Synthetic ECG signals are generated to simulate normal and hypoglycemic conditions.

2. **Feature Extraction**  
   Key features such as heart rate, signal variability, and frequency-domain parameters are extracted.

3. **Model Training**  
   A machine learning model (Random Forest) is trained to classify ECG patterns into:
   - Normal
   - Hypoglycemia risk

4. **Real-Time App**  
   The trained model is integrated into a **Streamlit app**, allowing users to upload ECG data and receive predictions instantly.

---
**Note about `model.pkl`:**
- The repository does **not** include a pre-trained model file (`model.pkl`) due to file size and reproducibility reasons.
- To run the demo locally you should:
  1. Install Python and required packages: `pip install -r requirements.txt`
  2. Generate synthetic example data: `python generate_data.py`
  3. Train the model (this will create `model.pkl`): `python train_model.py`
  4. Run the Streamlit app: `streamlit run app.py`

## 🧩 File Structure

| File | Description |
|------|--------------|
| `generate_data.py` | Generates synthetic ECG signals |
| `ecg_utils.py` | Contains signal processing and feature extraction functions |
| `train_model.py` | Trains and saves the hypoglycemia prediction model |
| `app.py` | Streamlit web app for real-time prediction |
| `requirements.txt` | List of Python libraries needed to run the project |
| `README.md` | Project documentation |

---

## 🖥️ How to Run (Optional)

If you want to run this project locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python generate_data.py

# 3. Train the model
python train_model.py

# 4. Run the app
streamlit run app.py
> This project was built with help from OpenAI’s ChatGPT as part of my CS50 learning journey.
