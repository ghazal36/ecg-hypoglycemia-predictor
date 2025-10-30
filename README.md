# ECG-Based Hypoglycemia Prediction System

This project is an AI-based system designed to **predict hypoglycemia (low blood glucose levels)** using only **ECG (Electrocardiogram) signals**.  
The goal is to develop a **non-invasive, real-time early warning system** that can alert users before a dangerous drop in blood glucose occurs.

---

## 🧠 Project Overview

Traditional hypoglycemia detection requires invasive glucose sensors.  
However, studies have shown that certain ECG patterns — such as heart rate variability (HRV) and QRS complex changes — can indicate an upcoming hypoglycemic event.

This project uses **machine learning** to analyze ECG data and predict whether a person is at risk of hypoglycemia.

---

## Quick run for reviewers
1. git clone https://github.com/<you>/<repo>.git
2. cd ecg-hypoglycemia-predictor && pip install -r requirements.txt
3. streamlit run app.py
4. Open http://localhost:8501

---

 ## Colab Integration

 The project was trained and tested in Google Colab.
 The full workflow — from data generation to model training — was executed in Colab notebooks to ensure reproducibility and easy cloud-based experimentation.
 A trained model file (model.pkl) was generated and tested successfully.
 
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
- A pretrained model file (model.pkl) is included in the repository for reviewers who want to try the demo immediately. If you prefer to retrain from scratch, run python generate_data.py and python train_model.py to recreate the mode.
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

## 🧪 Run in Google Colab

If you prefer not to install Python locally, you can run this entire project in Google Colab:

1. Clone the repository inside Colab.
2. Run generate_data.py to create sample ECG data.
3. Run train_model.py to train and save the model (model.pkl).
4. (Optional) Launch the Streamlit app inside Colab using ngrok for a temporary public link.

This setup allows you to experiment with the AI model even without installing anything on your computer.

---

## 📁 Note for Reviewers

* The model file (model.pkl) included in this repository was trained in Google Colab using synthetic ECG samples.
* For reproducibility, anyone can retrain the model by running the provided notebooks or scripts.
* This project demonstrates how ECG signals can be used as non-invasive biomarkers for detecting hypoglycemia risk.

---

## 🎯 Goals and Limitations

### Goals

* Explore the use of ECG signals for early detection of hypoglycemia.
* Build a fully automated pipeline from data generation to real-time prediction.
* Learn and demonstrate practical machine learning and biomedical signal analysis skills.

### Limitations

* The ECG data used here is **synthetic**, not real patient data.
* The model is for **educational and research purposes only**, not medical use.
* Accuracy may vary due to simulated signal randomness

---

### Colab Demo Notebook
colab_demo.ipynb reproduces the full pipeline in Google Colab (data generation → training → prediction) and can be used to recreate the model without installing dependencies locally.

---

## 🧪 Colab Demo
You can explore and run the project interactively using Google Colab. The Colab notebook contains all the steps to:

1. Generate synthetic ECG data
2. Train the hypoglycemia prediction model
3. Test the model with sample ECG signals
4. Run predictions and visualize results

📓 Colab Notebook:
[colab_demo.ipynb](colab_demo.ipynb)

> Note: The Colab notebook provides a fully functional demo, including model training. Streamlit real-time links require ngrok setup; here we demonstrate everything within Colab itself.

---

## 📸 Example Outputs

| Model Training (Colab) | 
|------------------------|
| ![Training Output](assets/training_output.png.jpg) |

---

## 🙏 Acknowledgements

This project was created as part of my **CS50 learning journey** with guidance and assistance from **OpenAI’s ChatGPT**,
which helped in structuring code, debugging, and technical explanations

---

## 👩‍💻 Author
Developed by Ghazal Attari
Student participant in CS50

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
streamlit run app.py```


