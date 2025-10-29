import os
import pandas as pd
import joblib
from ecg_utils import extract_features_from_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def gather_features_labels(samples_folder="data_samples"):
    labels_df = pd.read_csv(os.path.join(samples_folder, "labels.csv"))
    X = []
    y = []
    for _, row in labels_df.iterrows():
        file = row["file"]
        label = int(row["label"])
        feats = extract_features_from_csv(file)
        X.append(feats)
        y.append(label)
    X_df = pd.DataFrame(X)
    y_sr = pd.Series(y)
    return X_df, y_sr

def train_and_save_model(out_path="model.pkl"):
    X, y = gather_features_labels()
    print("Features:\n", X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train.fillna(0), y_train)
    preds = clf.predict(X_test.fillna(0))
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    joblib.dump(clf, out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    train_and_save_model("model.pkl")
