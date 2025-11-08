"""Predict spam/ham for a given input message using trained model.
Usage (PowerShell):
  python predict.py "Your message here"
If no argument is provided, enters interactive mode.
"""
from __future__ import annotations
import os
import sys
import joblib

MODEL_PATH = os.path.join("models", "spam_model.joblib")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Please run train.py first.")
    return joblib.load(path)


def predict(text: str):
    model = load_model(MODEL_PATH)
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    # Find probability for predicted class
    label_index = list(model.classes_).index(pred)
    confidence = proba[label_index]
    return pred, confidence


def interactive_loop():
    print("Enter message text (type /quit to exit):")
    while True:
        line = input("> ").strip()
        if line.lower() in {"/quit", "quit", "exit"}:
            break
        if not line:
            continue
        label, conf = predict(line)
        print(f"=> {label.upper()} (confidence={conf:.4f})")


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, conf = predict(text)
        print(f"Prediction: {label} (confidence={conf:.4f})")
    else:
        interactive_loop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
