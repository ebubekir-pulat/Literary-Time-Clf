import json
from ollama import chat
from pydantic import BaseModel, Field
from typing import Literal

from ML_Classifier import get_trained_tfidf_logreg, get_trained_tfidf_linearsvm

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



if __name__ == "__main__":
    pred, confidence = log_reg_respond("Sample text.")
    print("Logistic Regression Pred: ", pred)
    print("Logistic Regression Confidence: ", confidence)

    pred = linear_svm_respond("Sample text.")
    print("\nLinear SVM Pred: ", pred)

