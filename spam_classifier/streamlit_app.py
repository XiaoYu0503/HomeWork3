"""Streamlit Demo for SMS Spam Classifier.
Run:
  streamlit run spam_classifier/streamlit_app.py
"""
from __future__ import annotations
import os
from typing import Optional, List

import pandas as pd
import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
    top_n_terms = st.slider("Top 權重詞顯示數量", min_value=5, max_value=50, value=20, step=5)
    show_wordcloud = st.checkbox("顯示詞雲 (WordCloud)", True)
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

    col_a, col_b, col_c = st.columns(3)
    # 標籤分佈
    label_counts = dataset["label"].value_counts()
    col_a.metric("總筆數", f"{len(dataset):,}")
    col_a.write(label_counts)
    # 長度直方圖
    fig_len, ax_len = plt.subplots(figsize=(4,3))
    ax_len.hist(dataset["length"], bins=40, color="#4e79a7", alpha=0.85)
    ax_len.set_title("訊息長度直方圖")
    ax_len.set_xlabel("字元數")
    ax_len.set_ylabel("頻率")
    col_b.pyplot(fig_len, clear_figure=True)

    # 訊息長度分佈
    dataset["length"] = dataset["text"].str.len()
    col_b.caption("訊息長度分佈 (部分統計)")
    col_b.write(dataset["length"].describe())

    # Top TF-IDF 詞彙（簡易：擷取模型向量器特徵 + spam 類別對應的 LogisticRegression 權重）
    try:
        if hasattr(model, "named_steps") and "tfidf" in model.named_steps and "clf" in model.named_steps:
            vect = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            feature_names: List[str] = list(vect.get_feature_names_out())
            if len(clf.classes_) == 2:
                spam_index = list(clf.classes_).index("spam")
                # Binary logistic regression coef_ shape could be (1, n_features)
                if clf.coef_.shape[0] == 1:
                    weights_spam = clf.coef_[0]
                    weights_ham = -clf.coef_[0]  # approximate opposite
                else:
                    weights_spam = clf.coef_[spam_index]
                    ham_index = 1 - spam_index
                    weights_ham = clf.coef_[ham_index]
                # Top spam
                spam_top_idx = np.argsort(weights_spam)[::-1][:top_n_terms]
                ham_top_idx = np.argsort(weights_ham)[::-1][:top_n_terms]
                df_spam_top = pd.DataFrame([(feature_names[i], float(weights_spam[i])) for i in spam_top_idx], columns=["term","weight"]) 
                df_ham_top = pd.DataFrame([(feature_names[i], float(weights_ham[i])) for i in ham_top_idx], columns=["term","weight"]) 
                st.markdown("### 類別關鍵詞 Top 排行")
                col_spam, col_ham = st.columns(2)
                col_spam.caption("Spam Top 詞彙")
                col_spam.table(df_spam_top)
                col_ham.caption("Ham Top 詞彙")
                col_ham.table(df_ham_top)
                # 詞雲
                if show_wordcloud:
                    st.markdown("### 詞雲視覺化")
                    spam_text = " ".join(dataset[dataset.label.str.lower()=="spam"]["text"].tolist())
                    ham_text = " ".join(dataset[dataset.label.str.lower()=="ham"]["text"].tolist())
                    wc_spam = WordCloud(width=600, height=400, background_color="white").generate(spam_text)
                    wc_ham = WordCloud(width=600, height=400, background_color="white").generate(ham_text)
                    col_w1, col_w2 = st.columns(2)
                    col_w1.image(wc_spam.to_array(), caption="Spam 詞雲", use_column_width=True)
                    col_w2.image(wc_ham.to_array(), caption="Ham 詞雲", use_column_width=True)
    except Exception as e:
        st.info(f"無法計算詞彙排行榜/詞雲: {e}")

    # 混淆矩陣 & ROC
    st.markdown("### 評估 (整份資料集重跑推論)")
    try:
        y_true = dataset["label"].astype(str)
        y_pred_full = model.predict(dataset["text"].astype(str))
        cm = confusion_matrix(y_true, y_pred_full, labels=["ham","spam"])
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(["ham","spam"])
        ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(["ham","spam"])
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        for (i,j), v in np.ndenumerate(cm):
            ax_cm.text(j, i, str(v), ha="center", va="center", color="black")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm, clear_figure=True)

        # ROC (spam as positive)
        if hasattr(model, "predict_proba"):
            proba_full = model.predict_proba(dataset["text"].astype(str))
            classes = list(model.classes_)
            if "spam" in classes:
                spam_index = classes.index("spam")
                spam_scores = proba_full[:, spam_index]
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
    except Exception as e:
        st.info(f"評估計算失敗: {e}")

    # 下載整體預測結果
    try:
        st.markdown("### 匯出預測結果")
        full_df = dataset.copy()
        full_df["pred"] = y_pred_full
        if "spam_scores" in locals():
            full_df["spam_prob"] = spam_scores
        csv_bytes = full_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載完整預測結果 CSV", data=csv_bytes, file_name="spam_predictions.csv", mime="text/csv")
    except Exception as e:
        st.info(f"無法產生下載：{e}")
else:
    st.info("資料檔缺失，僅能使用預測功能。")

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
