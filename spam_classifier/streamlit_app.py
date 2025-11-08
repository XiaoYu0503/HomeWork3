"""Streamlit Demo for SMS Spam Classifier.
Run:
  streamlit run spam_classifier/streamlit_app.py
"""
from __future__ import annotations
import os
import pandas as pd
import joblib
import streamlit as st
from typing import Optional

# 與訓練腳本輸出一致：模型位於專案根目錄下 models/spam_model.joblib
MODEL_PATH = os.path.join("models", "spam_model.joblib")
DATA_FILE = "sms_spam_no_header.csv"

@st.cache_resource(show_spinner=False)
def load_model() -> Optional[object]:
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="wide")
st.title("📨 SMS 垃圾簡訊分類器 Demo")

model = load_model()
if model is None:
    st.error("模型尚未建立，請先在根目錄執行: python .\\spam_classifier\\train.py")
    st.stop()

with st.sidebar:
    st.header("設定")
    show_prob = st.checkbox("顯示所有類別機率", True)
    batch_limit = st.number_input("批次預測顯示筆數上限", min_value=5, max_value=200, value=50, step=5)
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
