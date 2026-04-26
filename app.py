"""
MediScan AI — Gradio Application
==================================
Disease Predictor & Drug Recommender powered by Classical ML.

Usage:
    python app.py

Prerequisites:
    1. Run train.py first to generate models/ artifacts
    2. Ensure all 4 CSV files are in data/
"""

import sys
import os
import warnings
# Force UTF-8 so Unicode chars don't crash on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import gradio as gr
from huggingface_hub import InferenceClient

DISEASE_DRUG_MAP = {
    "heart attack": ["Aspirin", "Clopidogrel", "Atorvastatin", "Metoprolol", "Nitroglycerin"],
    "diabetes": ["Metformin", "Insulin", "Glipizide", "Sitagliptin", "Empagliflozin"],
    "malaria": ["Chloroquine", "Artemether", "Quinine", "Doxycycline"],
    "typhoid": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone", "Ampicillin"],
    "pneumonia": ["Amoxicillin", "Azithromycin", "Levofloxacin", "Doxycycline"],
    "dengue": ["Paracetamol", "Acetaminophen", "ORS solution", "IV Fluids"],
    "fungal infection": ["Fluconazole", "Clotrimazole", "Terbinafine", "Ketoconazole"],
    "jaundice": ["Ursodeoxycholic acid", "Lactulose", "Vitamin K", "Cholestyramine"],
    "migraine": ["Sumatriptan", "Topiramate", "Propranolol", "Amitriptyline"],
    "hypertension": ["Amlodipine", "Lisinopril", "Losartan", "Metoprolol", "Hydrochlorothiazide"],
    "urinary tract infection": ["Nitrofurantoin", "Trimethoprim", "Ciprofloxacin", "Fosfomycin"],
    "allergy": ["Cetirizine", "Loratadine", "Diphenhydramine", "Fexofenadine"],
    "arthritis": ["Ibuprofen", "Naproxen", "Methotrexate", "Hydroxychloroquine"],
    "tuberculosis": ["Isoniazid", "Rifampicin", "Pyrazinamide", "Ethambutol"],
    "hepatitis a": ["Supportive care", "Rest", "IV Fluids", "Vitamin K"],
    "hepatitis b": ["Tenofovir", "Entecavir", "Lamivudine", "Adefovir"],
    "hepatitis c": ["Sofosbuvir", "Daclatasvir", "Ribavirin", "Ledipasvir"],
    "hepatitis d": ["Pegylated Interferon", "Tenofovir"],
    "hepatitis e": ["Supportive care", "Rest", "IV Fluids"],
    "alcoholic hepatitis": ["Prednisolone", "Pentoxifylline", "Thiamine", "Zinc"],
    "chronic cholestasis": ["Ursodeoxycholic acid", "Cholestyramine", "Rifampicin"],
    "drug reaction": ["Antihistamines", "Corticosteroids", "Epinephrine"],
    "peptic ulcer disease": ["Omeprazole", "Pantoprazole", "Amoxicillin", "Clarithromycin"],
    "aids": ["Tenofovir", "Emtricitabine", "Efavirenz", "Dolutegravir"],
    "gastroenteritis": ["ORS solution", "Metronidazole", "Loperamide", "Zinc supplements"],
    "bronchial asthma": ["Salbutamol", "Budesonide", "Montelukast", "Ipratropium"],
    "cervical spondylosis": ["Ibuprofen", "Diclofenac", "Pregabalin", "Cyclobenzaprine"],
    "paralysis (brain hemorrhage)": ["Aspirin", "Warfarin", "Alteplase", "Mannitol"],
    "chicken pox": ["Acyclovir", "Calamine lotion", "Antihistamines", "Paracetamol"],
    "common cold": ["Paracetamol", "Cetirizine", "Pseudoephedrine", "Vitamin C"],
    "dimorphic hemorrhoids (piles)": ["Hydrocortisone cream", "Docusate", "Fiber supplements"],
    "varicose veins": ["Compression stockings", "Diosmin", "Troxerutin"],
    "hypothyroidism": ["Levothyroxine", "Liothyronine"],
    "hyperthyroidism": ["Methimazole", "Propylthiouracil", "Propranolol", "Radioiodine"],
    "hypoglycemia": ["Dextrose", "Glucagon", "Glucose tablets", "Orange juice"],
    "osteoarthritis": ["Ibuprofen", "Naproxen", "Acetaminophen", "Duloxetine"],
    "vertigo": ["Meclizine", "Dimenhydrinate", "Betahistine", "Prochlorperazine"],
    "acne": ["Benzoyl Peroxide", "Clindamycin", "Tretinoin", "Doxycycline"],
    "psoriasis": ["Methotrexate", "Cyclosporine", "Betamethasone", "Calcipotriol"],
    "impetigo": ["Mupirocin", "Fusidic acid", "Amoxicillin-Clavulanate"],
    "diabetes mellitus": ["Metformin", "Insulin Glargine", "Sitagliptin", "Empagliflozin"],
}

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LLM Initialization
# ─────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    llm_client = InferenceClient("Qwen/Qwen2.5-72B-Instruct", token=HF_TOKEN)
else:
    llm_client = None

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
DESC_PATH = os.path.join(DATA_DIR, "symptom_Description.csv")
PRECAUTION_PATH = os.path.join(DATA_DIR, "symptom_precaution.csv")
SEVERITY_PATH = os.path.join(DATA_DIR, "Symptom-severity.csv")

# ─────────────────────────────────────────────
# STARTUP — Load all artifacts
# ─────────────────────────────────────────────
print("Loading MediScan AI models and data...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LE_PATH)
feature_columns = joblib.load(FEATURES_PATH)  # list of 133 symptom strings

# Build display names (underscores → spaces, title case) → map back to column names
display_to_col = {}
for col in feature_columns:
    display = col.replace("_", " ").strip().title()
    display_to_col[display] = col

symptom_display_names = sorted(display_to_col.keys())

# Load supporting CSVs
df_desc = pd.read_csv(DESC_PATH)
df_desc.columns = df_desc.columns.str.strip()
df_desc["Disease"] = df_desc["Disease"].str.strip()

df_prec = pd.read_csv(PRECAUTION_PATH)
df_prec.columns = df_prec.columns.str.strip()
df_prec["Disease"] = df_prec["Disease"].str.strip()

df_sev = pd.read_csv(SEVERITY_PATH)
df_sev.columns = df_sev.columns.str.strip()
df_sev["Symptom"] = df_sev["Symptom"].str.strip()
# Build severity weight lookup {symptom_column_name → weight}
severity_map = dict(zip(df_sev["Symptom"], df_sev["weight"]))

print("All artifacts loaded successfully.")


# ─────────────────────────────────────────────
# HELPER — Severity Badge
# ─────────────────────────────────────────────
def get_severity_badge(score: int) -> str:
    if score <= 10:
        return "🟢 Mild"
    elif score <= 20:
        return "🟡 Moderate"
    else:
        return "🔴 Severe — Consult a doctor immediately"


# ─────────────────────────────────────────────
# HELPER — Compute Severity Score
# ─────────────────────────────────────────────
def compute_severity(selected_display_names: list) -> tuple:
    score = 0
    for disp in selected_display_names:
        col = display_to_col.get(disp)
        if col:
            # Try exact match first, then case-insensitive
            weight = severity_map.get(col, severity_map.get(col.lower(), 0))
            # Also try replacing spaces with underscores either way
            if weight == 0:
                weight = severity_map.get(col.replace("_", " "), 0)
            score += weight
    badge = get_severity_badge(score)
    return score, badge


# ─────────────────────────────────────────────
# HELPER — Get Disease Info
# ─────────────────────────────────────────────
def get_disease_description(disease: str) -> str:
    row = df_desc[df_desc["Disease"].str.lower() == disease.lower()]
    if not row.empty:
        return row.iloc[0]["Description"]
    return "No description available for this condition."


def get_precautions(disease: str) -> list:
    if llm_client:
        prompt = f"List exactly 4 concise, actionable precautionary steps or home remedies for someone with {disease}. Output them as a simple numbered list without any intro or outro text."
        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            text = response.choices[0].message.content
            # Parse numbered list
            lines = [line.strip() for line in text.strip().split('\n') if line.strip() and line[0].isdigit()]
            precs = []
            for line in lines:
                parts = line.split('.', 1)
                if len(parts) == 2:
                    precs.append(parts[1].strip())
            if len(precs) >= 2:
                return precs[:4]
        except Exception:
            pass # Fall back to CSV if LLM fails

    row = df_prec[df_prec["Disease"].str.lower() == disease.lower()]
    if not row.empty:
        precs = []
        for i in range(1, 5):
            col = f"Precaution_{i}"
            if col in row.columns:
                val = row.iloc[0][col]
                if pd.notna(val) and str(val).strip():
                    precs.append(str(val).strip())
        return precs
    return []


# ─────────────────────────────────────────────
# HELPER — OpenFDA Drug Lookup
# ─────────────────────────────────────────────
def get_drug_recommendations(disease: str):
    disease_lower = disease.lower().strip()
    results = []

    # Use mapping first, fallback to disease name
    known_drugs = DISEASE_DRUG_MAP.get(disease_lower, [disease])[:3]

    for drug_name in known_drugs:
        try:
            url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1"
            response = requests.get(url, timeout=5)
            data = response.json()

            if "results" in data and data["results"]:
                item = data["results"][0]
                openfda = item.get("openfda", {})
                brand = openfda.get("brand_name", [drug_name])[0]
                generic = openfda.get("generic_name", [drug_name])[0]
                manufacturer = openfda.get("manufacturer_name", ["Consult a pharmacist"])[0]
                indication = item.get("indications_and_usage", ["Not available"])[0][:300]
            else:
                brand = drug_name
                generic = drug_name
                manufacturer = "Consult a pharmacist"
                indication = f"Commonly prescribed for {disease}. Please consult a licensed physician."

        except Exception:
            brand = drug_name
            generic = drug_name
            manufacturer = "Consult a pharmacist"
            indication = f"Commonly prescribed for {disease}. Please consult a licensed physician."

        results.append({
            "brand": brand,
            "generic": generic,
            "manufacturer": manufacturer,
            "indication": indication,
            "drug_name": drug_name
        })

    return results


# ─────────────────────────────────────────────
# HELPER — LLM Advice Generation
# ─────────────────────────────────────────────
def generate_llm_advice(disease: str, symptoms: list, drugs: list) -> str:
    if not llm_client:
        return "> ⚠️ **HF_TOKEN not found.** Add your Hugging Face token as a Secret in Space settings to enable the AI Doctor."
    
    drug_names = [d['brand'] for d in drugs] if drugs else ["None specific"]
    
    prompt = f"""You are an empathetic, professional AI medical assistant. 
A user has reported these symptoms: {', '.join(symptoms)}.
Our ML model predicts the condition: {disease}.
Suggested active ingredients/medications: {', '.join(drug_names)}.

Write a brief, friendly prescription and advice note (under 150 words).
Include:
1. A brief, simple explanation of the condition.
2. Recommended home care or lifestyle advice.
3. A strong disclaimer that they must consult a real doctor.

Output in Markdown format."""
    
    try:
        response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"> ⚠️ **AI Doctor Error:** `{str(e)}`"


# ─────────────────────────────────────────────
# HELPER — Build Top-5 Confidence Chart
# ─────────────────────────────────────────────
def build_confidence_chart(probas: np.ndarray) -> plt.Figure:
    """Build a horizontal bar chart of top-5 predicted conditions."""
    top5_idx = np.argsort(probas)[-5:][::-1]
    top5_diseases = label_encoder.inverse_transform(top5_idx)
    top5_probs = probas[top5_idx] * 100  # convert to %

    # Color gradient: best → worst
    colors = ["#1e88e5", "#42a5f5", "#64b5f6", "#90caf9", "#bbdefb"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bars = ax.barh(
        range(len(top5_diseases) - 1, -1, -1),
        top5_probs,
        color=colors[: len(top5_diseases)],
        edgecolor="white",
        linewidth=0.8,
        height=0.6,
    )

    # Labels on bars
    for i, (bar, prob) in enumerate(
        zip(bars, top5_probs[::-1])
    ):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    ax.set_yticks(range(len(top5_diseases)))
    ax.set_yticklabels(top5_diseases[::-1], fontsize=10)
    ax.set_xlabel("Confidence (%)", fontsize=10, color="#555555")
    ax.set_title(
        "Top 5 Predicted Conditions",
        fontsize=13,
        fontweight="bold",
        color="#1a1a2e",
        pad=12,
    )
    ax.set_xlim(0, min(110, max(top5_probs) + 15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", colors="#333333")
    ax.tick_params(axis="x", colors="#555555")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# CORE — Predict Function
# ─────────────────────────────────────────────
def predict_disease(selected_symptoms: list):
    """
    Main prediction function called by Gradio.
    Returns: (result_md, info_md, drug_md, chart_fig, severity_score_str, severity_badge, llm_md)
    """
    try:
        # ── Validation ──────────────────────────
        if not selected_symptoms or len(selected_symptoms) < 3:
            warn_md = "⚠️ **Please select at least 3 symptoms for accurate results.**"
            return warn_md, "", "", None, "0", "N/A", ""

        # ── Build feature vector ─────────────────
        input_vec = pd.DataFrame(
            np.zeros((1, len(feature_columns)), dtype=int),
            columns=feature_columns,
        )
        for disp in selected_symptoms:
            col = display_to_col.get(disp)
            if col and col in feature_columns:
                input_vec[col] = 1

        # ── Severity score ───────────────────────
        sev_score, sev_badge = compute_severity(selected_symptoms)

        # ── Model Prediction ─────────────────────
        probas = model.predict_proba(input_vec)[0]
        pred_idx = int(np.argmax(probas))
        disease = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probas[pred_idx] * 100

        # ── Output Card 1 — Prediction ───────────
        result_md = f"""
## 🔬 Prediction Results

| | |
|---|---|
| **Predicted Disease** | **{disease}** |
| **Confidence** | `{confidence:.1f}%` |
| **Severity Level** | {sev_badge} |
"""

        # ── Output Card 2 — Disease Info ─────────
        description = get_disease_description(disease)
        precautions = get_precautions(disease)
        prec_list = "\n".join(
            [f"{i+1}. {p}" for i, p in enumerate(precautions)]
        ) if precautions else "_No precautions data available._"

        info_md = f"""
## 📋 Disease Information

### Description
{description}

### Precautionary Steps
{prec_list}
"""

        # ── Output Card 3 — Drug Recommendation ──
        drugs = get_drug_recommendations(disease)
        google_url = (
            f"https://www.google.com/search?q=buy+{disease.replace(' ', '+')}+medicine"
        )

        if drugs:
            drug_sections = []
            for i, d in enumerate(drugs, 1):
                brand = d.get('brand', 'Unknown')
                generic = d.get('generic', 'Unknown')
                fallback = d.get('drug_name', 'Unknown')
                
                # 1. Use openfda.brand_name if available
                if brand and brand != "Unknown":
                    display_name = brand
                # 2. Else use openfda.generic_name if available
                elif generic and generic != "Unknown":
                    display_name = generic
                # 3. Else use drug_name from DISEASE_DRUG_MAP
                else:
                    display_name = fallback
                
                # Ensure we NEVER display "Unknown" for brand/generic/manufacturer
                display_brand = display_name if brand == "Unknown" else brand
                display_generic = display_name if generic == "Unknown" else generic
                display_mfg = "Consult a pharmacist" if d['manufacturer'] == "Unknown" else d['manufacturer']

                google_drug_url = (
                    f"https://www.google.com/search?q=buy+"
                    f"{display_name.replace(' ', '+')}+medicine"
                )
                drugs_com_url = (
                    f"https://www.drugs.com/search.php?searchterm="
                    f"{display_name.replace(' ', '+')}"
                )
                drug_sections.append(
                    f"""
#### 💊 Drug {i}: {display_name}
| | |
|---|---|
| **Brand Name** | {display_brand} |
| **Generic Name** | {display_generic} |
| **Manufacturer** | {display_mfg} |

**Purpose / Indication:**
> {d['indication']}

🛒 [Search on Google]({google_drug_url}) &nbsp;&nbsp; 📖 [More Info on Drugs.com]({drugs_com_url})
"""
                )
            drug_md = (
                "## 💊 Drug Recommendations\n\n"
                + "\n---\n".join(drug_sections)
            )
        else:
            drug_md = f"""
## 💊 Drug Recommendations

> ℹ️ **No drug data found via API.** Please consult a licensed physician.

🛒 [Search for {disease} on Google]({google_url})
"""

        # ── Output Card 4 — Confidence Chart ─────
        chart_fig = build_confidence_chart(probas)
        
        # ── Output Card 5 — AI Doctor's Advice ───
        llm_md = "## 🤖 AI Doctor's Advice\n\n" + generate_llm_advice(disease, selected_symptoms, drugs)

        sev_score_str = str(sev_score)
        return result_md, info_md, drug_md, chart_fig, sev_score_str, sev_badge, llm_md

    except Exception as e:
        err_md = f"❌ **Something went wrong.** Please try again.\n\n`{str(e)}`"
        return err_md, "", "", None, "0", "N/A", ""


# ─────────────────────────────────────────────
# GRADIO — Severity live update helper
# ─────────────────────────────────────────────
def update_severity(selected_symptoms: list):
    """Live severity calculation as user selects symptoms."""
    if not selected_symptoms:
        return "0", "Select symptoms above"
    score, badge = compute_severity(selected_symptoms)
    return str(score), badge


# ─────────────────────────────────────────────
# GRADIO — UI Layout
# ─────────────────────────────────────────────
CSS = """
/* ── Global ── */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ── Header ── */
#header-row {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 24px;
    margin-bottom: 8px;
    text-align: center;
}
#app-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
#app-subtitle {
    color: #90caf9;
    font-size: 1.05rem;
    margin-top: 8px;
}

/* ── Section headings ── */
.section-label {
    font-size: 1rem;
    font-weight: 600;
    color: #1e88e5;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

/* ── Severity badge display ── */
#severity-badge textarea {
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    border-radius: 8px;
}

/* ── Output cards ── */
.output-card {
    background: var(--background-fill-primary);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Primary button ── */
#submit-btn {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #1e88e5, #1565c0) !important;
    box-shadow: 0 4px 15px rgba(30,136,229,0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30,136,229,0.45) !important;
}

/* ── Info note ── */
.info-note {
    color: #546e7a;
    font-size: 0.88rem;
    margin-top: 4px;
    font-style: italic;
}

/* ── AI Doctor Advice Card ── */
#llm-card {
    border-top: 6px solid #1e88e5;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    padding: 30px;
}
#llm-card h2 {
    color: #1e88e5;
    font-weight: 800;
    border-bottom: 2px solid var(--border-color-primary);
    padding-bottom: 10px;
    margin-bottom: 20px;
}
#llm-card h3 {
    color: #1565c0;
    font-weight: 700;
    margin-top: 25px;
}
#llm-card p {
    font-size: 1.05rem;
    line-height: 1.7;
}
#llm-card ul {
    list-style: none;
    padding-left: 0;
    margin-top: 15px;
}
#llm-card ul li {
    padding-left: 30px;
    position: relative;
    margin-bottom: 12px;
    font-size: 1.05rem;
    line-height: 1.5;
}
#llm-card ul li::before {
    content: "✦";
    color: #1e88e5;
    position: absolute;
    left: 4px;
    font-size: 1.2rem;
}
#llm-card strong {
    color: #0d47a1;
    background: rgba(30, 136, 229, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
}
.dark #llm-card strong {
    color: #90caf9;
    background: rgba(144, 202, 249, 0.15);
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="MediScan AI — Disease Predictor & Drug Recommender",
) as demo:

    # ── Header ──────────────────────────────────
    with gr.Row(elem_id="header-row"):
        gr.HTML(
            """
            <div>
                <div id="app-title">🏥 MediScan AI</div>
                <div id="app-subtitle">
                    Select your symptoms to get a disease prediction and drug recommendation
                </div>
            </div>
            """
        )

    gr.HTML("<br>")

    # ── Input Section ────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML('<div class="section-label">🩺 Symptom Selection</div>')
            symptom_checkbox = gr.CheckboxGroup(
                choices=symptom_display_names,
                label="Select Your Symptoms",
                elem_id="symptom-selector",
            )
            gr.HTML(
                '<div class="info-note">'
                "ℹ️ Select at least 3 symptoms for better accuracy."
                "</div>"
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">📊 Severity Assessment</div>')
            severity_score_box = gr.Textbox(
                label="Total Symptom Severity Score",
                value="0",
                interactive=False,
                elem_id="severity-score",
            )
            severity_badge_box = gr.Textbox(
                label="Severity Level",
                value="Select symptoms above",
                interactive=False,
                elem_id="severity-badge",
            )
            gr.HTML(
                '<div class="info-note">'
                "Score range: 0–7 per symptom | 0–10 🟢 Mild | 11–20 🟡 Moderate | 21+ 🔴 Severe"
                "</div>"
            )

    # Live severity update as user selects/deselects symptoms
    symptom_checkbox.change(
        fn=update_severity,
        inputs=[symptom_checkbox],
        outputs=[severity_score_box, severity_badge_box],
    )

    gr.HTML("<br>")

    with gr.Row():
        submit_btn = gr.Button(
            "🔍 Analyze Symptoms",
            variant="primary",
            elem_id="submit-btn",
        )

    gr.HTML("<br>")

    # ── Output Section ───────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">🔬 Prediction Results</div>')
            result_output = gr.Markdown(
                value="*Submit your symptoms to see results.*",
                elem_id="result-card",
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">📈 Confidence Chart</div>')
            chart_output = gr.Plot(
                label="Top 5 Predicted Conditions",
                elem_id="chart-card",
            )

    gr.HTML("<br>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">📋 Disease Information</div>')
            info_output = gr.Markdown(
                value="",
                elem_id="info-card",
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">💊 Drug Recommendations</div>')
            drug_output = gr.Markdown(
                value="",
                elem_id="drug-card",
            )

    gr.HTML("<br>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">🤖 AI Doctor\'s Advice</div>')
            llm_output = gr.Markdown(
                value="",
                elem_id="llm-card",
                elem_classes=["output-card"]
            )

    # ── Submit Button Action ─────────────────────
    submit_btn.click(
        fn=predict_disease,
        inputs=[symptom_checkbox],
        outputs=[
            result_output,
            info_output,
            drug_output,
            chart_output,
            severity_score_box,
            severity_badge_box,
            llm_output,
        ],
    )

    # ── Footer ───────────────────────────────────
    gr.HTML(
        """
        <br>
        <div style="text-align:center; color:#90a4ae; font-size:0.82rem; padding:16px 0;">
            ⚠️ <strong>Disclaimer:</strong> This app is for educational purposes only.
            Always consult a licensed medical professional for diagnosis and treatment.<br>
            🔬 Powered by Classical ML (Decision Tree · Random Forest · XGBoost · SVM)
            &nbsp;|&nbsp; Drug data sourced from OpenFDA Public API
        </div>
        """
    )


# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # On HF Spaces the env var SPACE_ID is set automatically
    is_hf_space = os.environ.get("SPACE_ID") is not None
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=not is_hf_space,  # open browser locally only
    )
