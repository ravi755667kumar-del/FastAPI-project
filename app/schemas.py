from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class ChurnRequest(BaseModel):
    CreditScore: float = Field(..., description="Customer credit score", examples=[650.0])
    Gender: Literal["Male", "Female"] = Field(..., description="Gender( Male or Female )")
    Age: int = Field(..., description="Customer age")
    Tenure: int = Field(..., description="Tenure with the bank")
    Balance: float = Field(..., description="Account balance")
    NumOfProducts: int = Field(..., description="Number of bank products used")
    IsActiveMember: int = Field(..., description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., description="Estimated salary")
    Geography: Literal["France", "Germany", "Spain"] = Field(..., description="Choose from France, Germany, or Spain")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 619.0,
                "Gender": "Male",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
                "Geography": "France",
            }
        }
    )

class PredictionResponse(BaseModel):

    churn_probability: float = Field(
        ...,
        description="Probability that the customer will churn"
    )

    prediction: int = Field(
        ...,
        description="1 = Customer will churn, 0 = Customer will stay"
    )

    result: str = Field(
        ...,
        description="Final churn prediction"
    )