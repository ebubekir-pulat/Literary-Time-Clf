import json
from ollama import chat
from pydantic import BaseModel, Field
from typing import Literal

from ML_Classifier import get_trained_tfidf_logreg, get_trained_tfidf_linearsvm
from LLM_Classifier import Prediction

def log_reg_respond(user_text):
    model = get_trained_tfidf_logreg()
    pred = model.predict([user_text])[0]

    pos_of_pred = -1
    for i in range(len(model.classes_)):
        if model.classes_[i] == pred:
            pos_of_pred = i
            break

    confidence = model.predict_proba([user_text])[0][pos_of_pred]

    return pred, confidence


def linear_svm_respond(user_text):
    model = get_trained_tfidf_linearsvm()
    pred = model.predict([user_text])[0]

    return pred


def llm_respond(user_text):
    MODEL = "qwen2.5:3b"

    prompt = f"""
    Determine which period this historical book excerpt most likely
    originates from, based on its language, spelling, vocabulary,
    grammar, writing style and any other possible factors.

    Possible periods:
    - 1800-1832 (inclusive)
    - 1833-1866 (inclusive)
    - 1867-1899 (inclusive)

    Book excerpt:
    {user_text}

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

    prediction = prediction.model_dump()

    pred = prediction["answer"]
    confidence = prediction["confidence"]
    explanation = prediction["explanation"]

    return pred, confidence, explanation


if __name__ == "__main__":
    pred, confidence = log_reg_respond("Sample text.")
    print("Logistic Regression Pred: ", pred)
    print("Logistic Regression Confidence: ", confidence)

    pred = linear_svm_respond("Sample text.")
    print("\nLinear SVM Pred: ", pred)

    pred, confidence, explanation = llm_respond("Sample text.")
    print("\nLLM Pred: ", pred)
    print("LLM Confidence: ", confidence)
    print("LLM Explanation: ", explanation)