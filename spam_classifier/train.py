"""Train a simple SMS spam classifier using TF-IDF + Logistic Regression.
Dataset: sms_spam_no_header.csv (no header). Assumes first column is label (spam/ham) and second column is message text.
Outputs: models/spam_model.joblib and prints evaluation metrics.
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATA_FILE = "sms_spam_no_header.csv"
MODEL_PATH = os.path.join("models", "spam_model.joblib")
TEST_METRICS_PATH = os.path.join("models", "test_metrics.csv")
TEST_SPLIT_PATH = os.path.join("models", "test_split.csv")
TEST_PRED_PATH = os.path.join("models", "test_predictions.csv")
TEST_CM_PATH = os.path.join("models", "test_confusion_matrix.json")
MODEL_META_PATH = os.path.join("models", "model_meta.json")
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
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Confusion matrix (test set)
    classes = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    # 使用壓縮以縮小檔案體積，利於雲端部署
    joblib.dump(pipe, MODEL_PATH, compress=3)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    # Save test metrics for Streamlit "Model Performance (Test)" tab
    try:
        df_metrics = pd.DataFrame(report).transpose()
        df_metrics.insert(0, "label", df_metrics.index)
        df_metrics.to_csv(TEST_METRICS_PATH, index=False)
        print(f"[INFO] Test metrics saved to {TEST_METRICS_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save test metrics: {e}")

    # Save test split (may allow re-analysis)
    try:
        pd.DataFrame({"label": y_test, "text": X_test}).to_csv(TEST_SPLIT_PATH, index=False)
        print(f"[INFO] Test split saved to {TEST_SPLIT_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save test split: {e}")

    # Save test predictions with probabilities
    try:
        if hasattr(pipe, "predict_proba"):
            proba_test = pipe.predict_proba(X_test)
            pred_test = y_pred
            class_list = list(pipe.classes_)
            rows = []
            for i, (label_true, text_val) in enumerate(zip(y_test, X_test)):
                row = {"label": label_true, "text": text_val, "pred": pred_test[i]}
                for ci, cname in enumerate(class_list):
                    row[f"prob_{cname}"] = proba_test[i][ci]
                rows.append(row)
            pd.DataFrame(rows).to_csv(TEST_PRED_PATH, index=False)
            print(f"[INFO] Test predictions saved to {TEST_PRED_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save test predictions: {e}")

    # Save confusion matrix JSON
    try:
        cm_obj = {
            "classes": classes,
            "matrix": cm.tolist()
        }
        with open(TEST_CM_PATH, "w", encoding="utf-8") as f:
            json.dump(cm_obj, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Test confusion matrix saved to {TEST_CM_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save test confusion matrix: {e}")

    # Save model metadata (general info)
    try:
        vect = pipe.named_steps.get("tfidf")
        clf = pipe.named_steps.get("clf")
        meta = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model_path": MODEL_PATH,
            "algorithm": clf.__class__.__name__ if clf else None,
            "vectorizer": {
                "type": vect.__class__.__name__ if vect else None,
                "ngram_range": getattr(vect, "ngram_range", None),
                "max_df": getattr(vect, "max_df", None),
                "min_df": getattr(vect, "min_df", None),
                "stop_words": getattr(vect, "stop_words", None),
                "feature_count": len(vect.get_feature_names_out()) if vect else None
            },
            "test_accuracy": acc,
            "classes": list(pipe.classes_) if hasattr(pipe, "classes_") else None,
            "test_size": TEST_SPLIT,
            "random_state": RANDOM_STATE
        }
        with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Model metadata saved to {MODEL_META_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save model metadata: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
