"""Streamlit Demo for SMS Spam Classifier.
Run:
  streamlit run spam_classifier/streamlit_app.py
"""
from __future__ import annotations
import os
from typing import Optional, List
from collections import Counter

import pandas as pd
import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support

# 與訓練腳本輸出一致：模型位於專案根目錄下 models/spam_model.joblib
MODEL_PATH = os.path.join("models", "spam_model.joblib")
DATA_FILE = "sms_spam_no_header.csv"

@st.cache_resource(show_spinner=False)
def load_model() -> Optional[object]:
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    # 資料為無表頭且含引號，指定 header=None 與欄位名稱
    df = pd.read_csv(path, encoding="utf-8", header=None, names=["label", "text"])
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    return df

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="wide")
st.title("📨 SMS Spam Classifier Demo")

model = load_model()
if model is None:
    st.error("Model not found. Please run: python .\\spam_classifier\\train.py in project root.")
    st.stop()
dataset: Optional[pd.DataFrame] = None
default_dataset = load_dataset(DATA_FILE)
if default_dataset is None:
    st.warning("Default dataset sms_spam_no_header.csv not found. You may upload one manually for visualization.")

with st.sidebar:
    st.header("Settings")
    show_prob = st.checkbox("Show class probabilities", True)
    batch_limit = st.number_input("Batch prediction display limit", min_value=5, max_value=200, value=50, step=5)
    st.markdown("---")
    st.subheader("Token Patterns")
    token_scope = st.selectbox("Text scope", ["All", "ham", "spam"], index=0)
    token_ngram = st.slider("n-gram length", min_value=1, max_value=2, value=1, step=1)
    token_topk = st.slider("Top N frequent tokens", min_value=10, max_value=100, value=30, step=10)
    top_n_terms = st.slider("Top feature weight count", min_value=5, max_value=50, value=20, step=5)
    show_wordcloud = st.checkbox("Show WordCloud", True)
    st.markdown("---")
    st.subheader("Model Performance")
    spam_threshold = st.slider("Spam probability threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    st.markdown("---")
    st.markdown("**Model path**: ``{}``".format(MODEL_PATH))
    st.markdown("---")
    st.subheader("Dataset Source")
    dataset_source = st.radio("Select dataset", ["Default (sms_spam_no_header.csv)", "Upload CSV"], index=0)
    uploaded_dataset_file = None
    if dataset_source.startswith("Upload"):
        uploaded_dataset_file = st.file_uploader("Upload dataset CSV", type=["csv"], key="dataset_uploader")
    # Decide dataset
    if dataset_source.startswith("Default"):
        dataset = default_dataset
    else:
        if uploaded_dataset_file is not None:
            try:
                # Try read with header; if doesn't contain required columns, fallback
                uploaded_dataset_file.seek(0)
                df_tmp = pd.read_csv(uploaded_dataset_file)
                if set(df_tmp.columns) >= {"label", "text"}:
                    dataset = df_tmp
                elif df_tmp.shape[1] >= 2:
                    uploaded_dataset_file.seek(0)
                    dataset = pd.read_csv(uploaded_dataset_file, header=None, names=["label", "text"])
                else:
                    st.error("Uploaded CSV must have at least 2 columns (label,text).")
                    dataset = None
                if dataset is not None:
                    dataset = dataset.dropna(subset=["text"]).copy()
                    dataset["text"] = dataset["text"].astype(str)
                    st.success(f"Loaded uploaded dataset: {len(dataset):,} rows.")
            except Exception as e:
                st.error(f"Failed to read uploaded dataset: {e}")
                dataset = None
        else:
            dataset = None
    if dataset is None and dataset_source.startswith("Upload"):
        st.info("No uploaded dataset loaded yet.")
    elif dataset is None:
        st.info("Dataset unavailable; only prediction features active.")

# 單筆輸入
st.subheader("Single Message Prediction")
text = st.text_area("Enter message text:", height=120, placeholder="e.g. Free entry in a weekly cash prize draw")
col_predict, col_clear = st.columns([1,1])
if col_predict.button("🔮 Predict"):
    if not text.strip():
        st.warning("Please enter message text.")
    else:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        prob_map = dict(zip(classes, proba))
        is_spam = pred.lower() == "spam"
        color = "#d9534f" if is_spam else "#5cb85c"
        st.markdown(f"<div style='padding:12px;border-radius:6px;background:{color};color:#fff;font-weight:bold;'>Prediction: {pred.upper()}</div>", unsafe_allow_html=True)
        if show_prob:
            df_prob = pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False)
            st.table(df_prob)
if col_clear.button("🧹 Clear"):
    st.experimental_set_query_params()  # 简單刷新

st.markdown("---")

# 批次上傳
st.subheader("Batch Prediction (CSV Upload)")
st.caption("Format: no header -> first column label (optional), second column text. If header exists the app will detect.")
uploaded = st.file_uploader("Choose CSV file", type=["csv"]) 
if uploaded is not None:
    try:
        # 嘗試讀取：先嘗試含表頭，不行則指定欄位
        try:
            df_up = pd.read_csv(uploaded)
            if set(df_up.columns) >= {"label", "text"}:
                pass
            elif df_up.shape[1] >= 2:
                df_up = pd.read_csv(uploaded, header=None, names=["label", "text"])
            else:
                st.error("CSV requires at least 2 columns (label,text).")
                df_up = None
        except Exception:
            uploaded.seek(0)
            df_up = pd.read_csv(uploaded, header=None, names=["label", "text"])
        if df_up is not None:
            df_up = df_up.dropna(subset=["text"]).head(batch_limit)
            preds = model.predict(df_up["text"].astype(str))
            probas = model.predict_proba(df_up["text"].astype(str))
            classes = list(model.classes_)
            df_result = df_up.copy()
            df_result["pred"] = preds
            # 取 spam 機率以便排序（假設存在 spam 類別）
            if "spam" in classes:
                spam_index = classes.index("spam")
                df_result["spam_prob"] = [p[spam_index] for p in probas]
            if show_prob:
                # 展開各類別機率
                for ci, cname in enumerate(classes):
                    df_result[f"prob_{cname}"] = [p[ci] for p in probas]
            st.write(df_result)
            st.success(f"Completed {len(df_result)} predictions.")
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")

st.markdown("---")

# 資料集視覺化
st.subheader("Data Exploration / Visualization")
if dataset is not None:
    with st.expander("First 10 rows"):
        st.dataframe(dataset.head(10))

    st.markdown("---")

    # 儀表板分頁
    tabs = st.tabs(["Data Overview", "Top Tokens by Class", "Token Patterns", "Model Performance (Full)", "Model Performance (Test)"])

    # Data Overview
    with tabs[0]:
        st.subheader("Data Overview")
        if dataset is None:
            st.info("No dataset loaded.")
        else:
            col_stats, col_len = st.columns([1,2])
            with col_stats:
                st.markdown("**Basic Stats**")
                st.metric("Rows", f"{len(dataset):,}")
                lbl_counts = dataset["label"].value_counts(dropna=False)
                st.write("Label counts:")
                st.table(lbl_counts.to_frame("count"))
                # Length metrics
                lengths = dataset["text"].str.len()
                st.metric("Avg Length", f"{lengths.mean():.1f}")
                st.metric("Median Length", f"{lengths.median():.1f}")
                st.metric("Max Length", f"{lengths.max():,}")
                st.metric("Min Length", f"{lengths.min():,}")
                missing_label = dataset["label"].isna().sum()
                missing_text = dataset["text"].isna().sum()
                st.caption(f"Missing label rows: {missing_label}; Missing text rows: {missing_text}")
                with st.expander("Preview (head 10)"):
                    st.dataframe(dataset.head(10))

            # Length distribution
            dataset["__length__"] = dataset["text"].str.len()
            fig_len, ax_len = plt.subplots(figsize=(7,3))
            ax_len.hist(dataset["__length__"], bins=40, color="#4e79a7", alpha=0.7, label="All")
            try:
                ax_len.hist(dataset.loc[dataset.label.str.lower()=="ham","__length__"], bins=40, alpha=0.5, label="ham")
                ax_len.hist(dataset.loc[dataset.label.str.lower()=="spam","__length__"], bins=40, alpha=0.5, label="spam")
                ax_len.legend()
            except Exception:
                pass
            ax_len.set_title("Message Length Histogram")
            ax_len.set_xlabel("Characters")
            ax_len.set_ylabel("Frequency")
            col_len.pyplot(fig_len, clear_figure=True)

    # Top Tokens by Class
    with tabs[1]:
        st.subheader("Top Tokens by Class")
        if dataset is None:
            st.info("No dataset loaded.")
        else:
            if hasattr(model, "named_steps") and "tfidf" in model.named_steps:
                vect = model.named_steps["tfidf"]
                # logistic regression coefficients
                if "clf" in model.named_steps and hasattr(model.named_steps["clf"], "coef_"):
                    clf = model.named_steps["clf"]
                    feature_names = np.array(vect.get_feature_names_out())
                    coefs = clf.coef_[0]
                    # Spam positive class assumed -> high positive => spam, negative => ham
                    spam_top_idx = np.argsort(coefs)[-top_n_terms:][::-1]
                    ham_top_idx = np.argsort(coefs)[:top_n_terms]
                    spam_df = pd.DataFrame({"token": feature_names[spam_top_idx], "weight": coefs[spam_top_idx]})
                    ham_df = pd.DataFrame({"token": feature_names[ham_top_idx], "weight": coefs[ham_top_idx]})
                    col_spam, col_ham = st.columns(2)
                    with col_spam:
                        st.markdown(f"**Top Spam Tokens (N={top_n_terms})**")
                        st.table(spam_df)
                    with col_ham:
                        st.markdown(f"**Top Ham Tokens (N={top_n_terms})**")
                        st.table(ham_df)
                    # Combined difference view
                    diff_df = pd.concat([
                        spam_df.assign(class_label="spam"),
                        ham_df.assign(class_label="ham")
                    ])
                    with st.expander("Download token weights"):
                        csv_bytes = diff_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button("Download token weights CSV", data=csv_bytes, file_name="token_weights.csv", mime="text/csv")
                    if show_wordcloud:
                        try:
                            st.markdown("**WordClouds**")
                            spam_tokens = {t: w for t, w in zip(spam_df["token"], spam_df["weight"])}
                            ham_tokens = {t: abs(w) for t, w in zip(ham_df["token"], ham_df["weight"])}
                            fig_wc, (ax_spam, ax_ham) = plt.subplots(1,2, figsize=(10,4))
                            WordCloud(width=600, height=400, background_color="white").generate_from_frequencies(spam_tokens)
                            ax_spam.imshow(WordCloud(width=600, height=400, background_color="white").generate_from_frequencies(spam_tokens))
                            ax_spam.axis("off"); ax_spam.set_title("Spam")
                            ax_ham.imshow(WordCloud(width=600, height=400, background_color="white").generate_from_frequencies(ham_tokens))
                            ax_ham.axis("off"); ax_ham.set_title("Ham")
                            st.pyplot(fig_wc, clear_figure=True)
                        except Exception as e:
                            st.info(f"WordCloud failed: {e}")
                else:
                    st.info("Classifier coefficients unavailable.")
            else:
                st.info("Vectorizer not found in pipeline.")

    # 令牌模式
    with tabs[2]:
        st.subheader("Token Patterns (Vectorizer based)")
        try:
            if dataset is not None and hasattr(model, "named_steps") and "tfidf" in model.named_steps:
                vect = model.named_steps["tfidf"]
                analyzer = vect.build_analyzer()
                # 篩選資料範圍
                if token_scope == "ham":
                    texts = dataset.loc[dataset.label.str.lower()=="ham","text"].astype(str).tolist()
                elif token_scope == "spam":
                    texts = dataset.loc[dataset.label.str.lower()=="spam","text"].astype(str).tolist()
                else:
                    texts = dataset["text"].astype(str).tolist()
                counter = Counter()
                for t in texts:
                    toks = analyzer(t)
                    # 篩選 n-gram 長度
                    for tok in toks:
                        if (tok.count(" ")+1) == token_ngram:
                            counter[tok] += 1
                common = counter.most_common(token_topk)
                df_tok = pd.DataFrame(common, columns=["token", "count"])
                st.caption(f"Top {token_topk} tokens (n={token_ngram}, scope={token_scope})")
                st.table(df_tok)
                # 可選：點選一個 token 顯示範例句
                if len(df_tok):
                    picked = st.selectbox("Show sample sentences containing token:", ["(None)"] + df_tok["token"].head(20).tolist())
                    if picked and picked != "(None)":
                        examples = [s for s in texts if picked in s][:5]
                        for ex in examples:
                            st.write("• ", ex)
            else:
                st.info("Missing dataset or vectorizer; cannot show token patterns.")
        except Exception as e:
            st.info(f"Token pattern computation failed: {e}")

    # 模型效能 (Full Dataset)
    with tabs[3]:
        st.subheader("Model Performance (Full Dataset Inference)")
        try:
            if dataset is not None:
                y_true = dataset["label"].astype(str)
                # 使用機率 + 閾值產生預測
                if hasattr(model, "predict_proba"):
                    proba_full = model.predict_proba(dataset["text"].astype(str))
                    classes = list(model.classes_)
                    if "spam" in classes:
                        spam_index = classes.index("spam")
                        spam_scores = proba_full[:, spam_index]
                        y_pred_thr = np.where(spam_scores >= spam_threshold, "spam", "ham")
                    else:
                        # 後備：直接使用 predict
                        y_pred_thr = model.predict(dataset["text"].astype(str))
                        spam_scores = None
                else:
                    y_pred_thr = model.predict(dataset["text"].astype(str))
                    spam_scores = None

                # 指標
                acc = accuracy_score(y_true, y_pred_thr)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_thr, labels=["ham","spam"], average=None)
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Accuracy", f"{acc:.4f}")
                # 顯示 spam 這一類的 P/R/F1
                try:
                    spam_idx = ["ham","spam"].index("spam")
                    col_m2.metric("Precision (spam)", f"{prec[spam_idx]:.4f}")
                    col_m3.metric("Recall (spam)", f"{rec[spam_idx]:.4f}")
                    st.caption(f"F1 (spam) = {f1[spam_idx]:.4f}")
                except Exception:
                    pass

                # 混淆矩陣
                cm = confusion_matrix(y_true, y_pred_thr, labels=["ham","spam"])
                fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                im = ax_cm.imshow(cm, cmap="Blues")
                ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(["ham","spam"])
                ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(["ham","spam"])
                ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
                for (i,j), v in np.ndenumerate(cm):
                    ax_cm.text(j, i, str(v), ha="center", va="center", color="black")
                ax_cm.set_title(f"Confusion Matrix (threshold={spam_threshold:.2f})")
                st.pyplot(fig_cm, clear_figure=True)

                # ROC（與閾值無關）
                if spam_scores is not None:
                    y_bin = (y_true.str.lower()=="spam").astype(int)
                    fpr, tpr, _ = roc_curve(y_bin, spam_scores)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots(figsize=(4,3))
                    ax_roc.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
                    ax_roc.plot([0,1],[0,1], linestyle="--", color="gray")
                    ax_roc.set_xlabel("FPR")
                    ax_roc.set_ylabel("TPR")
                    ax_roc.set_title("ROC Curve (spam positive class)")
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc, clear_figure=True)

                # 匯出
                st.markdown("### Export Predictions")
                full_df = dataset.copy()
                full_df["pred"] = y_pred_thr
                if spam_scores is not None:
                    full_df["spam_prob"] = spam_scores
                csv_bytes = full_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Download full predictions CSV", data=csv_bytes, file_name="spam_predictions.csv", mime="text/csv")
            else:
                st.info("Dataset missing; performance metrics not available.")
        except Exception as e:
            st.info(f"Performance calculation failed: {e}")

    # Model Performance (Test)
    with tabs[4]:
        st.subheader("Model Performance (Test)")
        metrics_path = os.path.join("models", "test_metrics.csv")
        meta_path = os.path.join("models", "model_meta.json")
        cm_path = os.path.join("models", "test_confusion_matrix.json")
        if os.path.exists(metrics_path):
            try:
                df_test_metrics = pd.read_csv(metrics_path)
                st.caption("Loaded test set metrics produced during training.")
                st.dataframe(df_test_metrics)
                # Optional: summary cards
                try:
                    acc_row = df_test_metrics[df_test_metrics["label"]=="accuracy"].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{acc_row['precision']:.4f}")
                    if os.path.exists(meta_path):
                        import json as _json
                        meta = _json.load(open(meta_path, "r", encoding="utf-8"))
                        col2.metric("Features", str(meta.get("vectorizer",{}).get("feature_count")))
                        col3.metric("Algorithm", meta.get("algorithm","-"))
                except Exception:
                    pass
                # Confusion matrix (test)
                if os.path.exists(cm_path):
                    try:
                        import json as _json
                        cm_obj = _json.load(open(cm_path, "r", encoding="utf-8"))
                        classes = cm_obj.get("classes", [])
                        cm = np.array(cm_obj.get("matrix", []))
                        if cm.size:
                            fig_cm_t, ax_cm_t = plt.subplots(figsize=(4,3))
                            im = ax_cm_t.imshow(cm, cmap="Purples")
                            ax_cm_t.set_xticks(range(len(classes))); ax_cm_t.set_xticklabels(classes)
                            ax_cm_t.set_yticks(range(len(classes))); ax_cm_t.set_yticklabels(classes)
                            ax_cm_t.set_xlabel("Predicted"); ax_cm_t.set_ylabel("Actual")
                            for (i,j), v in np.ndenumerate(cm):
                                ax_cm_t.text(j, i, str(v), ha="center", va="center", color="black")
                            ax_cm_t.set_title("Test Confusion Matrix")
                            st.pyplot(fig_cm_t, clear_figure=True)
                    except Exception as e:
                        st.info(f"Unable to show test confusion matrix: {e}")
            except Exception as e:
                st.error(f"Failed to load test metrics: {e}")
        else:
            st.info("Test metrics file not found. Re-run training to generate.")

# 說明區塊
st.markdown("---")
with st.expander("Help / Guide"):
    st.markdown(
        """
        **Usage**
        - Enter a single message and press 'Predict'.
        - Upload a CSV for batch prediction (columns: label,text). Label can be blank.
        - If model is missing run `python .\\spam_classifier\\train.py`.

        **Dashboard Tabs**
        - Data Distribution: counts and length histogram.
        - Token Patterns: frequent tokens by scope and n-gram, plus feature weight ranking & word clouds.
        - Model Performance: metrics, confusion matrix, ROC and export.

        **Possible Improvements**
        - Add text cleaning (URLs, emojis, numbers normalization).
        - Try alternative models (SVC, Naive Bayes, deep learning).
        - Add training versioning & incremental retraining.
        """
    )
