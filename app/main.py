from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from app.schemas import ChurnRequest, PredictionResponse


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "ann_model.keras"
PROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
# ==================================================
# CREATE FASTAPI APP
# ==================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ANN based Customer Churn Prediction API",
    version="1.0.0"
)


# ==================================================
# LOAD PREPROCESSOR
# ==================================================

processor = joblib.load(PROCESSOR_PATH)


# ==================================================
# LOAD ANN MODEL
# ==================================================

model = load_model(MODEL_PATH)


# ==================================================
# HOME ROUTE
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


# ==================================================
# PREDICTION ROUTE
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(data: ChurnRequest):

    # ----------------------------------------------
    # 1. Convert Pydantic data into dictionary
    # ----------------------------------------------

    customer_data = data.model_dump()


    # ----------------------------------------------
    # 2. Convert dictionary into DataFrame
    # ----------------------------------------------

    df = pd.DataFrame([customer_data])


    # ----------------------------------------------
    # 3. Apply the SAME processor used during
    #    model training
    #
    #    This handles:
    #
    #    - SimpleImputer
    #    - OneHotEncoder
    #    - StandardScaler
    #
    # ----------------------------------------------

    processed_data = processor.transform(df)


    # ----------------------------------------------
    # 4. Convert processed data into NumPy array
    # ----------------------------------------------

    processed_data = np.asarray(
        processed_data,
        dtype=np.float32
    )


    # ----------------------------------------------
    # 5. ANN prediction
    # ----------------------------------------------

    prediction_probability = model.predict(
        processed_data,
        verbose=0
    )


    # ----------------------------------------------
    # 6. Get probability
    # ----------------------------------------------

    probability = float(
        prediction_probability[0][0]
    )


    # ----------------------------------------------
    # 7. Convert probability into class
    #
    # 0.0 - 0.49 → Stay
    # 0.50 - 1.0 → Churn
    # ----------------------------------------------

    prediction = 1 if probability >= 0.5 else 0


    # ----------------------------------------------
    # 8. Human-readable result
    # ----------------------------------------------

    if prediction == 1:

        result = "Customer will churn"

    else:

        result = "Customer will stay"


    # ----------------------------------------------
    # 9. Return response
    # ----------------------------------------------

    return {
        "churn_probability": probability,
        "prediction": prediction,
        "result": result
    }