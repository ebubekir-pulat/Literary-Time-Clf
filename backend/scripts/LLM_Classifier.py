import json
from ollama import chat, ChatResponse
from pydantic import BaseModel, Field
from typing import Literal

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

for i, sample in enumerate(samples):

    prompt = f"""
    Determine which period this historical book excerpt most likely
    originates from, based on its language, spelling, vocabulary,
    grammar, and writing style.

    Possible periods:
    - 1800-1832
    - 1833-1866
    - 1867-1899

    Book excerpt:
    {sample['text']}

    Classify the excerpt based on its linguistic characteristics,
    NOT its provided publication year.

    Provide your best prediction and explain your reasoning.
    """

    response = ollama.chat(
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
        "book_id": sample["book_id"],
        "actual_year": sample["year"],
        "prediction": prediction.model_dump()
    })

    print(f"Sample {i+1}/{len(samples)}")
    print(prediction.model_dump())
    print("-" * 50)


# Save predictions
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Predictions saved to predictions.json")