from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# تحميل الموديلات
diabetes_model = joblib.load("diabetes_final.pkl")
heart_model = joblib.load("heart__final.pkl")
kidney_model = joblib.load("kindey__final.pkl")

# =====================================================
#                   DIABETES
# =====================================================

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):

    NewBMI_Underweight = 0
    NewBMI_Overweight = 0
    NewBMI_Obesity_1 = 0
    NewBMI_Obesity_2 = 0
    NewBMI_Obesity_3 = 0
    NewInsulinScore_Normal = 0
    NewGlucose_Low = 0
    NewGlucose_Normal = 0
    NewGlucose_Overweight = 0
    NewGlucose_Secret = 0

    if data.BMI <= 18.5:
        NewBMI_Underweight = 1
    elif 24.9 < data.BMI <= 29.9:
        NewBMI_Overweight = 1
    elif 29.9 < data.BMI <= 34.9:
        NewBMI_Obesity_1 = 1
    elif 34.9 < data.BMI <= 39.9:
        NewBMI_Obesity_2 = 1
    elif data.BMI > 39.9:
        NewBMI_Obesity_3 = 1

    if 16 <= data.Insulin <= 166:
        NewInsulinScore_Normal = 1

    if data.Glucose <= 70:
        NewGlucose_Low = 1
    elif 70 < data.Glucose <= 99:
        NewGlucose_Normal = 1
    elif 99 < data.Glucose <= 126:
        NewGlucose_Overweight = 1
    elif data.Glucose > 126:
        NewGlucose_Secret = 1

    user_input = [
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age,
        NewBMI_Underweight,
        NewBMI_Overweight,
        NewBMI_Obesity_1,
        NewBMI_Obesity_2,
        NewBMI_Obesity_3,
        NewInsulinScore_Normal,
        NewGlucose_Low,
        NewGlucose_Normal,
        NewGlucose_Overweight,
        NewGlucose_Secret,
    ]

    prediction = diabetes_model.predict([user_input])
    return {"prediction": int(prediction[0])}


# =====================================================
#                   HEART
# =====================================================

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.post("/predict/heart")
def predict_heart(data: HeartInput):

    user_input = [
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal,
    ]

    prediction = heart_model.predict([user_input])
    return {"prediction": int(prediction[0])}


# =====================================================
#                   KIDNEY
# =====================================================

class KidneyInput(BaseModel):
    age: float
    blood_pressure: float
    specific_gravity: float
    albumin: float
    sugar: float
    red_blood_cells: float
    pus_cell: float
    pus_cell_clumps: float
    bacteria: float
    blood_glucose_random: float
    blood_urea: float
    serum_creatinine: float
    sodium: float
    potassium: float
    haemoglobin: float
    packed_cell_volume: float
    white_blood_cell_count: float
    red_blood_cell_count: float
    hypertension: float
    diabetes_mellitus: float
    coronary_artery_disease: float
    appetite: float
    peda_edema: float
    aanemia: float


@app.post("/predict/kidney")
def predict_kidney(data: KidneyInput):

    user_input = [
        data.age,
        data.blood_pressure,
        data.specific_gravity,
        data.albumin,
        data.sugar,
        data.red_blood_cells,
        data.pus_cell,
        data.pus_cell_clumps,
        data.bacteria,
        data.blood_glucose_random,
        data.blood_urea,
        data.serum_creatinine,
        data.sodium,
        data.potassium,
        data.haemoglobin,
        data.packed_cell_volume,
        data.white_blood_cell_count,
        data.red_blood_cell_count,
        data.hypertension,
        data.diabetes_mellitus,
        data.coronary_artery_disease,
        data.appetite,
        data.peda_edema,
        data.aanemia,
    ]

    prediction = kidney_model.predict([user_input])
    return {"prediction": int(prediction[0])}