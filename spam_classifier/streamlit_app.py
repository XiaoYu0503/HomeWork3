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
st.title("📨 SMS 垃圾簡訊分類器 Demo")

model = load_model()
if model is None:
    st.error("模型尚未建立，請先在根目錄執行: python .\\spam_classifier\\train.py")
    st.stop()

dataset = load_dataset(DATA_FILE)
if dataset is None:
    st.warning("找不到資料檔 sms_spam_no_header.csv，資料視覺化功能將停用。")

with st.sidebar:
    st.header("設定")
    show_prob = st.checkbox("顯示所有類別機率", True)
    batch_limit = st.number_input("批次預測顯示筆數上限", min_value=5, max_value=200, value=50, step=5)
    st.markdown("---")
    st.subheader("令牌模式")
    token_scope = st.selectbox("資料範圍", ["全部", "ham", "spam"], index=0)
    token_ngram = st.slider("n-gram 長度", min_value=1, max_value=2, value=1, step=1)
    token_topk = st.slider("顯示前 N 個常見令牌", min_value=10, max_value=100, value=30, step=10)
    top_n_terms = st.slider("Top 權重詞顯示數量", min_value=5, max_value=50, value=20, step=5)
    show_wordcloud = st.checkbox("顯示詞雲 (WordCloud)", True)
    st.markdown("---")
    st.subheader("模型效能")
    spam_threshold = st.slider("Spam 判定閾值", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    st.markdown("---")
    st.markdown("**模型路徑**: ``{}``".format(MODEL_PATH))

# 單筆輸入
st.subheader("單筆訊息預測")
text = st.text_area("輸入簡訊內容：", height=120, placeholder="例如：Free entry in a weekly cash prize draw")
col_predict, col_clear = st.columns([1,1])
if col_predict.button("🔮 預測"):
    if not text.strip():
        st.warning("請輸入訊息內容。")
    else:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        prob_map = dict(zip(classes, proba))
        is_spam = pred.lower() == "spam"
        color = "#d9534f" if is_spam else "#5cb85c"
        st.markdown(f"<div style='padding:12px;border-radius:6px;background:{color};color:#fff;font-weight:bold;'>分類結果： {pred.upper()}</div>", unsafe_allow_html=True)
        if show_prob:
            df_prob = pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False)
            st.table(df_prob)
if col_clear.button("🧹 清除"):
    st.experimental_set_query_params()  # 简單刷新

st.markdown("---")

# 批次上傳
st.subheader("批次預測 (CSV 上傳)")
st.caption("格式：無表頭，第一欄 label(可留空)、第二欄 text。若已有表頭亦可上傳，程式會嘗試辨識。")
uploaded = st.file_uploader("選擇 CSV 檔", type=["csv"]) 
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
                st.error("CSV 欄位不足，需至少 2 欄。")
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
            st.success(f"完成 {len(df_result)} 筆預測。")
    except Exception as e:
        st.error(f"讀取或預測時發生錯誤: {e}")

st.markdown("---")

# 資料集視覺化
st.subheader("資料探索 / 視覺化")
if dataset is not None:
    with st.expander("原始資料前 10 筆"):
        st.dataframe(dataset.head(10))

    st.markdown("---")

    # 儀表板分頁
    tabs = st.tabs(["資料分佈", "令牌模式", "模型效能"])

    # 資料分佈
    with tabs[0]:
        st.subheader("資料分佈")
        if dataset is not None:
            with st.expander("原始資料前 10 筆"):
                st.dataframe(dataset.head(10))

            col_a, col_b = st.columns([1,2])
            label_counts = dataset["label"].value_counts()
            col_a.metric("總筆數", f"{len(dataset):,}")
            col_a.write(label_counts)

            dataset["length"] = dataset["text"].str.len()
            fig_len, ax_len = plt.subplots(figsize=(6,3))
            ax_len.hist(dataset["length"], bins=40, color="#4e79a7", alpha=0.7, label="All")
            # 類別對比直方圖
            try:
                ax_len.hist(dataset.loc[dataset.label.str.lower()=="ham","length"], bins=40, alpha=0.5, label="ham")
                ax_len.hist(dataset.loc[dataset.label.str.lower()=="spam","length"], bins=40, alpha=0.5, label="spam")
                ax_len.legend()
            except Exception:
                pass
            ax_len.set_title("訊息長度直方圖")
            ax_len.set_xlabel("字元數")
            ax_len.set_ylabel("頻率")
            col_b.pyplot(fig_len, clear_figure=True)
        else:
            st.info("資料檔缺失，僅能使用預測功能。")

    # 令牌模式
    with tabs[1]:
        st.subheader("令牌模式（依資料與向量器）")
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
                st.caption(f"Top {token_topk} 令牌（n={token_ngram}, 範圍={token_scope}）")
                st.table(df_tok)
                # 可選：點選一個 token 顯示範例句
                if len(df_tok):
                    picked = st.selectbox("查看包含此令牌的範例句：", ["(不選)"] + df_tok["token"].head(20).tolist())
                    if picked and picked != "(不選)":
                        examples = [s for s in texts if picked in s][:5]
                        for ex in examples:
                            st.write("• ", ex)
            else:
                st.info("缺少資料或向量器，無法顯示令牌模式。")
        except Exception as e:
            st.info(f"令牌模式計算失敗：{e}")

    # 模型效能
    with tabs[2]:
        st.subheader("模型效能（整份資料集重跑推論）")
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
                ax_cm.set_title(f"Confusion Matrix (thresh={spam_threshold:.2f})")
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
                    ax_roc.set_title("ROC Curve (spam as positive)")
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc, clear_figure=True)

                # 匯出
                st.markdown("### 匯出預測結果")
                full_df = dataset.copy()
                full_df["pred"] = y_pred_thr
                if spam_scores is not None:
                    full_df["spam_prob"] = spam_scores
                csv_bytes = full_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下載完整預測結果 CSV", data=csv_bytes, file_name="spam_predictions.csv", mime="text/csv")
            else:
                st.info("資料檔缺失，無法計算效能。")
        except Exception as e:
            st.info(f"效能計算失敗：{e}")

# 說明區塊
st.markdown("---")
with st.expander("說明 / Help"):
    st.markdown(
        """
        **使用說明**
        - 單筆輸入區輸入訊息後按下『預測』。
        - 批次上傳支援 CSV，前兩欄視為 label 與 text；label 可為空用於推論。
        - 若尚未訓練模型，請先在專案根目錄執行：`python .\\spam_classifier\\train.py`。

        **改進建議**
        - 可增加資料清理（URL、表情符號正規化）。
        - 可替換模型為 SVC、Naive Bayes 或深度學習。
        - 可加入混淆矩陣與 ROC 曲線視覺化。
        """
    )
