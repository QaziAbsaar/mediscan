"""
MediScan AI - Training Pipeline
================================
Trains 4 classical ML models on the disease-symptom dataset,
evaluates them, and saves the best model to models/.

Usage:
    python train.py

Prerequisites:
    Place all 4 CSV files from the Kaggle dataset
    (itachi9604/disease-symptom-description-dataset) in the data/ folder.
"""

import sys
import os
import warnings
# Force UTF-8 output so Unicode characters don't crash on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -----------------------------------------------
# Paths
# -----------------------------------------------
DATA_PATH = os.path.join("data", "dataset.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
CONFUSION_MATRIX_PATH = "confusion_matrix.png"


# -----------------------------------------------
# STEP 1 - Load & Clean Data
# -----------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    print("\n[1/5] Loading and cleaning data...")
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from all string values
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Fill NaN with 0 (binary symptoms)
    df = df.fillna(0)

    print(f"    Dataset shape: {df.shape}")
    print(f"    Target classes: {df['prognosis'].nunique()}")
    print(f"    Sample diseases: {df['prognosis'].unique()[:5].tolist()}")
    return df


# -----------------------------------------------
# STEP 2 - Preprocessing
# -----------------------------------------------
def preprocess(df: pd.DataFrame):
    print("\n[2/5] Preprocessing data...")

    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save artifacts
    feature_columns = list(X.columns)
    joblib.dump(le, LABEL_ENCODER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    print(f"    Feature columns saved: {len(feature_columns)} symptoms")
    print(f"    Label encoder saved: {len(le.classes_)} classes")

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    print(f"    Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, le, feature_columns


# -----------------------------------------------
# STEP 3 - Train 4 Models
# -----------------------------------------------
def build_models():
    """Return a dict of {name: unfitted_model}."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42
        ),
    }


# -----------------------------------------------
# STEP 4 - Evaluate All Models
# -----------------------------------------------
def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test, le):
    print("\n[3/5] Training and evaluating models...")
    results = {}

    for name, model in models.items():
        print(f"\n  -- {name} --------------------------")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=le.classes_)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Classification Report:\n{report}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "y_pred": y_pred,
        }

    return results


def save_confusion_matrix(best_name: str, y_test, y_pred, le):
    """Save confusion matrix heatmap for the best model."""
    print(f"\n[4/5] Saving confusion matrix for {best_name}...")
    cm = confusion_matrix(y_test, y_pred)

    # For 41 classes, use a large figure
    n_classes = len(le.classes_)
    fig_size = max(20, n_classes // 2)

    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"Confusion Matrix — {best_name}", fontsize=14, pad=15)
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    plt.close()
    print(f"    Saved: {CONFUSION_MATRIX_PATH}")


def print_comparison_table(results: dict):
    print("\n[4/5] Model Comparison:")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1-Score (Weighted)':>20}")
    print(f"  {'='*20} {'='*10} {'='*20}")
    for name, r in results.items():
        print(f"  {name:<20} {r['accuracy']:>10.4f} {r['f1']:>20.4f}")


# -----------------------------------------------
# STEP 5 - Save Best Model
# -----------------------------------------------
def save_best_model(results: dict, y_test):
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_result = results[best_name]
    best_acc = best_result["accuracy"]
    best_model = best_result["model"]

    print(f"\n[5/5] Best model: {best_name}  |  Accuracy: {best_acc:.4f}")
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"    Saved: {BEST_MODEL_PATH}")
    return best_name, best_result["y_pred"]


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    print("=" * 60)
    print("  MediScan AI — Training Pipeline")
    print("=" * 60)

    df = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test, le, feature_columns = preprocess(df)

    models = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test, le)

    print_comparison_table(results)
    best_name, best_y_pred = save_best_model(results, y_test)
    save_confusion_matrix(best_name, y_test, best_y_pred, le)

    print("\n" + "=" * 60)
    print("  Training complete! Files saved:")
    print(f"    {BEST_MODEL_PATH}")
    print(f"    {LABEL_ENCODER_PATH}")
    print(f"    {FEATURE_COLUMNS_PATH}")
    print(f"    {CONFUSION_MATRIX_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
