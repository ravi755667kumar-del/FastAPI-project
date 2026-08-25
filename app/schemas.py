from pydantic import BaseModel, ConfigDict, Field

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