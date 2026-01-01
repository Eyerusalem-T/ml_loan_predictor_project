from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Enable CORS
origins = ["*"]  # allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
log_model = joblib.load("../models/logistic_model.joblib")
dt_model = joblib.load("../models/decision_tree_model.joblib")
scaler = joblib.load("../models/scaler.joblib")

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict_loan(data: dict, model_type: str = "logistic"):
    features = np.array([[
        data["Gender"],
        data["Married"],
        data["Dependents"],
        data["Education"],
        data["Self_Employed"],
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Loan_Amount_Term"],
        data["Credit_History"],
        data["Property_Area"]
    ]])

    features_scaled = scaler.transform(features)

    if model_type == "decision_tree":
        prediction = dt_model.predict(features)
    else:
        prediction = log_model.predict(features_scaled)

    return {"loan_approved": "Yes" if prediction[0] == 1 else "No"}
