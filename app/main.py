from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

app = FastAPI(title="ANN Churn Prediction API")

# 1. Load the Scaler and the ANN Model when the app starts
try:
    print("Loading scaler...")
    scaler = joblib.load("models/churn_scaler.pkl")
    
    print("Loading ANN model...")
    ann_model = load_model("models/churn_ann_model.keras")
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# 2. Define the exact features your model expects (10 features total)
class ChurnRequest(BaseModel):
    CreditScore: float = Field(default=619.0, description="Customer credit score")
    Age: float = Field(default=42.0, description="Customer age")
    Gender: float = Field(default=0.0, description="Gender (0 for Female, 1 for Male)")
    Tenure: float = Field(default=2.0, description="Tenure with the bank")
    Balance: float = Field(default=0.0, description="Account balance")
    NumOfProducts: float = Field(default=1.0, description="Number of bank products used")
    IsActiveMember: float = Field(default=1.0, description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(default=101348.88, description="Estimated salary")
    Geography_Germany: float = Field(default=0.0, description="One-hot encoded column for Germany")
    Geography_Spain: float = Field(default=0.0, description="One-hot encoded column for Spain")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 619.0,
                "Age": 42.0,
                "Gender": 0.0,
                "Tenure": 2.0,
                "Balance": 0.0,
                "NumOfProducts": 1.0,
                "IsActiveMember": 1.0,
                "EstimatedSalary": 101348.88,
                "Geography_Germany": 0.0,
                "Geography_Spain": 0.0,
            }
        }
    )

@app.post("/predict")
def predict_churn(request: ChurnRequest):
    try:
        # 1. Convert the incoming JSON request into a Pandas DataFrame
        input_dict = request.model_dump()
        input_df = pd.DataFrame([input_dict])

        # 2. Separate numerical features (the 6 columns the scaler expects) 
        # from binary/one-hot features (the 4 columns that stay unscaled)
        numerical_cols = [
            'CreditScore', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'EstimatedSalary'
        ]
        binary_cols = [
            'Gender', 'IsActiveMember', 
            'Geography_Germany', 'Geography_Spain'
        ]

        # 3. Scale ONLY the 6 numerical features
        scaled_numerical = scaler.transform(input_df[numerical_cols].to_numpy())

        # 4. Extract the binary features as a numpy array
        binary_features = input_df[binary_cols].to_numpy()

        # 5. Combine them back together into the full 10-feature array for the neural network
        final_features = np.hstack((scaled_numerical, binary_features))

        # 6. Pass the combined features to the Neural Network
        prediction_prob = churn_ann_model.predict(final_features)[0][0]

        # 7. Convert probability to a final Churn decision (Threshold = 0.5)
        is_churn = bool(prediction_prob > 0.5)

        return {
            "churn_probability": float(prediction_prob),
            "prediction": "Churn" if is_churn else "No Churn"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")