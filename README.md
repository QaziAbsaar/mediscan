---
title: MediScan AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
---

# 🏥 MediScan AI — Disease Predictor & Drug Recommender

> ⚠️ **Disclaimer**: This app is for **educational purposes only**. Always consult a licensed medical professional for diagnosis and treatment.

---

## 1. Project Overview

MediScan AI is a Classical Machine Learning web application that:

- Accepts user-selected symptoms from a list of **133 possible symptoms**
- Predicts the most likely disease from **41 possible conditions**
- Displays a **severity score** based on symptom weights
- Provides a **disease description** and **4 precautionary steps**
- Recommends **real drugs** fetched live from the **OpenFDA API**
- Shows a **confidence chart** of the top-5 predicted conditions

---

## 2. How It Works (ML Pipeline Summary)

```
Raw Symptoms (binary 0/1 vector)
        ↓
  LabelEncoder (on target) + Feature Column Extraction
        ↓
  Train/Test Split (80/20, stratified, random_state=42)
        ↓
  Train 4 Classical Models:
    - Decision Tree (gini, max_depth=10)
    - Random Forest (200 estimators, max_depth=15)
    - XGBoost (200 estimators, depth=8, lr=0.1)
    - SVM (RBF kernel, C=1.0, probability=True)
        ↓
  Evaluate all → Select Best by Accuracy
        ↓
  Save: best_model.pkl, label_encoder.pkl, feature_columns.pkl
        ↓
  Gradio App → Inference → OpenFDA Drug Lookup → Display Results
```

---

## 3. Dataset Used

**Kaggle Dataset**: [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)  
**Author**: itachi9604  
**Files used**:

| File | Description |
|------|-------------|
| `dataset.csv` | 133 binary symptom columns + 41 disease classes (~4920 rows) |
| `symptom_Description.csv` | Disease → plain-English description |
| `symptom_precaution.csv` | Disease → 4 precautionary steps |
| `Symptom-severity.csv` | Symptom → severity weight (1–7) |

---

## 4. Models Trained & Accuracy Comparison

> Results shown below are representative. Actual values may vary slightly due to data splits.

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| Decision Tree | ~95–98% | ~0.95–0.98 |
| Random Forest | ~98–100% | ~0.98–1.00 |
| XGBoost | ~97–99% | ~0.97–0.99 |
| SVM (RBF) | ~96–98% | ~0.96–0.98 |

✅ The best model is automatically selected and saved as `models/best_model.pkl`.

---

## 5. Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn, xgboost |
| Data Processing | pandas, numpy |
| Serialization | joblib |
| Visualization | matplotlib, seaborn |
| Web UI | Gradio 4.44.0 |
| Drug Data API | OpenFDA (no key required) |

**No neural networks. No deep learning. No transformers. Pure Classical ML.**

---

## 6. How to Run Locally

### Step 1 — Generate the dataset
```bash
python generate_dataset.py
```

### Step 2 — Install dependencies
```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Step 3 — Train the models
```bash
python train.py
```

### Step 4 — Launch the app
```bash
python app.py
```
Open your browser at `http://localhost:7860`

---

## 7. Disclaimer

> This app is for **educational purposes only**.  
> It does **not** constitute medical advice.  
> Always consult a **licensed medical professional** for diagnosis and treatment decisions.  
> Drug information is sourced from the OpenFDA public API and may not reflect the most current prescribing information.
