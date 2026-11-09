import json
from ollama import chat
from pydantic import BaseModel, Field
from typing import Literal
from load_dataset import get_train_and_test_data
from score import print_final_results

MODEL = "qwen2.5:3b"

# Structured output schema
class Prediction(BaseModel):
    answer: Literal[
        "1800-1832",
        "1833-1866",
        "1867-1899"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

# Predict individual samples
results = []

X_train, X_test, y_train, y_test = get_train_and_test_data()

for i, sample in enumerate(X_test):

    prompt = f"""
    Determine which period this historical book excerpt most likely
    originates from, based on its language, spelling, vocabulary,
    grammar, writing style and any other possible factors.

    Possible periods:
    - 1800-1832 (inclusive)
    - 1833-1866 (inclusive)
    - 1867-1899 (inclusive)

    Book excerpt:
    {sample}

    Provide your best prediction and explain your reasoning
    for your prediction very concisely. Record a confidence score, 
    of how confident you are in your answer, as well. The
    confidence score is a decimal from 0.0 to 1.0 inclusive.
    """

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format=Prediction.model_json_schema()
    )

    prediction = Prediction.model_validate_json(
        response.message.content
    )

    results.append({
        "prediction": prediction.model_dump()
    })

    print(f"Sample {i+1}/{len(X_test)}")
    print(prediction.model_dump())
    print("Expected: ", y_test[i])
    print("-" * 50)


y_pred = [pred["prediction"]["answer"] for pred in results]

print("Final Results")
print_final_results(y_test, y_pred)