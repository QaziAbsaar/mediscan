"""
Generates data/dataset.csv matching the itachi9604 Kaggle schema:
- 132 binary symptom columns
- 'prognosis' column with 41 disease classes
"""
import os, random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)
os.makedirs("data", exist_ok=True)

SYMPTOMS = [
    "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering",
    "chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting",
    "vomiting","burning_micturition","spotting_urination","fatigue","weight_gain",
    "anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness",
    "lethargy","patches_in_throat","irregular_sugar_level","cough","high_fever",
    "sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache",
    "yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes",
    "back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
    "yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach",
    "swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm",
    "throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion",
    "chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements",
    "pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness",
    "cramps","bruising","obesity","swollen_legs","swollen_blood_vessels",
    "puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties",
    "excessive_hunger","extra_marital_contacts","drying_and_tingling_lips",
    "slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck",
    "swelling_joints","movement_stiffness","spinning_movements","loss_of_balance",
    "unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort",
    "foul_smell_of_urine","continuous_feel_of_urine","passage_of_gases",
    "internal_itching","toxic_look_(typhos)","depression","irritability","muscle_pain",
    "altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation",
    "dischromic_patches","watering_from_eyes","increased_appetite","polyuria",
    "family_history","mucoid_sputum","rusty_sputum","lack_of_concentration",
    "visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections",
    "coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption",
    "fluid_overload.1","blood_in_sputum","prominent_veins_on_calf","palpitations",
    "painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling",
    "silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister",
    "red_sore_around_nose","yellow_crust_ooze",
]

DISEASE_SYMPTOMS = {
    "Fungal infection":["itching","skin_rash","nodal_skin_eruptions","dischromic_patches"],
    "Allergy":["continuous_sneezing","shivering","chills","watering_from_eyes"],
    "GERD":["stomach_pain","acidity","ulcers_on_tongue","vomiting","cough","chest_pain"],
    "Chronic cholestasis":["itching","vomiting","yellowish_skin","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes"],
    "Drug Reaction":["itching","skin_rash","stomach_pain","burning_micturition","spotting_urination"],
    "Peptic ulcer disease":["vomiting","indigestion","loss_of_appetite","abdominal_pain","passage_of_gases","internal_itching"],
    "AIDS":["muscle_wasting","patches_in_throat","high_fever","extra_marital_contacts"],
    "Diabetes":["fatigue","weight_loss","restlessness","lethargy","irregular_sugar_level","polyuria","increased_appetite","blurred_and_distorted_vision"],
    "Gastroenteritis":["vomiting","sunken_eyes","dehydration","diarrhoea"],
    "Bronchial Asthma":["fatigue","cough","high_fever","breathlessness","family_history","mucoid_sputum"],
    "Hypertension":["headache","dizziness","loss_of_balance","lack_of_concentration"],
    "Migraine":["acidity","indigestion","headache","blurred_and_distorted_vision","excessive_hunger","stiff_neck","depression","irritability"],
    "Cervical spondylosis":["back_pain","weakness_in_limbs","neck_pain","dizziness","loss_of_balance"],
    "Paralysis (brain hemorrhage)":["vomiting","headache","weakness_of_one_body_side","altered_sensorium"],
    "Jaundice":["itching","vomiting","fatigue","weight_loss","high_fever","yellowish_skin","dark_urine","abdominal_pain"],
    "Malaria":["chills","vomiting","high_fever","sweating","headache","nausea","diarrhoea","muscle_pain"],
    "Chicken pox":["itching","skin_rash","fatigue","lethargy","high_fever","headache","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise","red_spots_over_body","phlegm"],
    "Dengue":["skin_rash","chills","joint_pain","vomiting","fatigue","high_fever","headache","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","mild_fever","red_spots_over_body","muscle_pain"],
    "Typhoid":["chills","vomiting","fatigue","high_fever","headache","nausea","constipation","abdominal_pain","diarrhoea","toxic_look_(typhos)","belly_pain"],
    "Hepatitis A":["joint_pain","vomiting","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","diarrhoea","mild_fever","yellowing_of_eyes","muscle_pain"],
    "Hepatitis B":["itching","fatigue","lethargy","yellowish_skin","dark_urine","loss_of_appetite","abdominal_pain","yellow_urine","yellowing_of_eyes","malaise","receiving_blood_transfusion","receiving_unsterile_injections"],
    "Hepatitis C":["fatigue","yellowish_skin","nausea","loss_of_appetite","family_history"],
    "Hepatitis D":["joint_pain","vomiting","fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes"],
    "Hepatitis E":["joint_pain","vomiting","fatigue","high_fever","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes","acute_liver_failure","coma","stomach_bleeding"],
    "Alcoholic hepatitis":["vomiting","yellowish_skin","abdominal_pain","swelling_of_stomach","history_of_alcohol_consumption","fluid_overload.1","pain_during_bowel_movements","ascites"],
    "Tuberculosis":["chills","vomiting","fatigue","weight_loss","cough","high_fever","breathlessness","sweating","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise","phlegm","blood_in_sputum","rusty_sputum"],
    "Common Cold":["continuous_sneezing","chills","fatigue","cough","high_fever","headache","swelled_lymph_nodes","malaise","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","loss_of_smell","muscle_pain"],
    "Pneumonia":["chills","fatigue","cough","high_fever","breathlessness","sweating","malaise","phlegm","chest_pain","fast_heart_rate","rusty_sputum"],
    "Dimorphic hemmorhoids(piles)":["constipation","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus"],
    "Heart attack":["vomiting","breathlessness","sweating","chest_pain"],
    "Varicose veins":["fatigue","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","prominent_veins_on_calf"],
    "Hypothyroidism":["fatigue","weight_gain","cold_hands_and_feets","mood_swings","lethargy","dizziness","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","depression","irritability","abnormal_menstruation"],
    "Hyperthyroidism":["fatigue","mood_swings","weight_loss","restlessness","sweating","diarrhoea","fast_heart_rate","excessive_hunger","muscle_weakness","irritability","abnormal_menstruation"],
    "Hypoglycemia":["vomiting","fatigue","anxiety","sweating","headache","nausea","blurred_and_distorted_vision","slurred_speech","irritability","palpitations","muscle_weakness","excessive_hunger"],
    "Osteoarthritis":["joint_pain","neck_pain","knee_pain","hip_joint_pain","swelling_joints","painful_walking"],
    "Arthritis":["muscle_weakness","stiff_neck","swelling_joints","movement_stiffness","loss_of_balance"],
    "(Vertigo) Paroxysmal Positional Vertigo":["vomiting","headache","nausea","spinning_movements","loss_of_balance","unsteadiness"],
    "Acne":["skin_rash","pus_filled_pimples","blackheads","scurring"],
    "Urinary tract infection":["burning_micturition","bladder_discomfort","foul_smell_of_urine","continuous_feel_of_urine"],
    "Psoriasis":["skin_rash","joint_pain","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails"],
    "Impetigo":["skin_rash","high_fever","blister","red_sore_around_nose","yellow_crust_ooze"],
}

# Remove any symptom not in our SYMPTOMS list
all_sym_set = set(SYMPTOMS)
DISEASE_SYMPTOMS_CLEAN = {}
for disease, syms in DISEASE_SYMPTOMS.items():
    DISEASE_SYMPTOMS_CLEAN[disease] = [s for s in syms if s in all_sym_set]

rows = []
for disease, core_syms in DISEASE_SYMPTOMS_CLEAN.items():
    for _ in range(30):  # 30 samples per disease
        row = {s: 0 for s in SYMPTOMS}
        # Always include core symptoms
        for s in core_syms:
            row[s] = 1
        # Add 0-2 random noise symptoms
        noise_pool = [s for s in SYMPTOMS if s not in core_syms]
        for s in random.sample(noise_pool, k=random.randint(0, 2)):
            row[s] = 1
        row["prognosis"] = disease
        rows.append(row)

df = pd.DataFrame(rows)
cols = SYMPTOMS + ["prognosis"]
df = df[cols]
df.to_csv("data/dataset.csv", index=False)
print(f"[OK] data/dataset.csv created: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"[OK] Diseases: {df['prognosis'].nunique()}")

# Also generate symptom_Description.csv
desc_rows = [{"Disease": d, "Description": f"A medical condition known as {d}."} for d in DISEASE_SYMPTOMS_CLEAN]
pd.DataFrame(desc_rows).to_csv("data/symptom_Description.csv", index=False)
print("[OK] data/symptom_Description.csv created")

# symptom_precaution.csv
prec_rows = [{"Disease": d, "Precaution_1": "Consult a doctor", "Precaution_2": "Rest well",
              "Precaution_3": "Stay hydrated", "Precaution_4": "Take prescribed medication"}
             for d in DISEASE_SYMPTOMS_CLEAN]
pd.DataFrame(prec_rows).to_csv("data/symptom_precaution.csv", index=False)
print("[OK] data/symptom_precaution.csv created")

# Symptom-severity.csv
sev_rows = [{"Symptom": s, "weight": random.randint(1, 7)} for s in SYMPTOMS]
pd.DataFrame(sev_rows).to_csv("data/Symptom-severity.csv", index=False)
print("[OK] data/Symptom-severity.csv created")

print("\nAll done! Run: python train.py")
