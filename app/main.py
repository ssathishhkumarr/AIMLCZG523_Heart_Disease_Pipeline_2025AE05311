from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI(title="Heart Disease Prediction API", description="API for predicting heart disease using a Random Forest model.")
Instrumentator().instrument(app).expose(app)

# Load the trained ML pipeline
model = joblib.load("models/model.pkl")

# Define the expected input data schema using Pydantic
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is live!"}

@app.post("/predict")
def predict_heart_disease(data: PatientData):
    # Convert the JSON payload to a pandas DataFrame (which our pipeline expects)
    input_data = pd.DataFrame([data.model_dump()])
    
    # Generate prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "diagnosis": "Presence of Heart Disease" if prediction == 1 else "Absence of Heart Disease"
    }