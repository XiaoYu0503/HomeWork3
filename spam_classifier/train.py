"""Train a simple SMS spam classifier using TF-IDF + Logistic Regression.
Dataset: sms_spam_no_header.csv (no header). Assumes first column is label (spam/ham) and second column is message text.
Outputs: models/spam_model.joblib and prints evaluation metrics.
"""
from __future__ import annotations
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_FILE = "sms_spam_no_header.csv"
MODEL_PATH = os.path.join("models", "spam_model.joblib")
TEST_SPLIT = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    # 檔案已含引號且無表頭：直接給 names，讓 pandas 自動處理引號與內部逗號
    df = pd.read_csv(path, encoding="utf-8", names=["label", "text"])
    # 清理：移除缺失，確保文字型態
    df = df.dropna(subset=["label", "text"])  # Drop rows with missing values
    df["text"] = df["text"].astype(str)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2,
        )),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])


def main():
    print("[INFO] Loading data...")
    df = load_data(DATA_FILE)
    X = df["text"].values
    y = df["label"].values

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )

    print("[INFO] Building pipeline...")
    pipe = build_pipeline()

    print("[INFO] Training model...")
    pipe.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    # 使用壓縮以縮小檔案體積，利於雲端部署
    joblib.dump(pipe, MODEL_PATH, compress=3)
    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
