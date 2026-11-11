
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make the scripts folder accessible to Python
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from User_Response import (
    log_reg_respond,
    linear_svm_respond,
    llm_respond
)

# Create the API
app = FastAPI(title="Literary Time Classifier API")

# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the expected input
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


# Prediction endpoint
@app.post("/predict")
def predict(request: TextRequest):

    user_text = request.text.strip()

    if not user_text:
        raise HTTPException(
            status_code=400,
            detail="Please provide a text excerpt."
        )

    try:
        # Logistic Regression
        log_pred, log_conf = log_reg_respond(user_text)

        # Linear SVM
        svm_pred = linear_svm_respond(user_text)

        # Qwen LLM
        llm_pred, llm_conf, llm_explanation = llm_respond(user_text)

        return {
            "logistic_regression": {
                "prediction": str(log_pred),
                "confidence": float(log_conf)
            },
            "linear_svm": {
                "prediction": str(svm_pred)
            },
            "qwen": {
                "prediction": llm_pred,
                "confidence": llm_conf,
                "explanation": llm_explanation
            }
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trained model files could not be found."
        )
