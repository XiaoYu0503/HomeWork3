# HomeWork3

## SMS Spam Classifier + Streamlit Demo

本專案包含：
- `sms_spam_no_header.csv`：簡訊垃圾資料集（無表頭）。
- `spam_classifier/train.py`：使用 TF-IDF + LogisticRegression 訓練模型並輸出 `models/spam_model.joblib`。
- `spam_classifier/predict.py`：命令列預測工具（單筆或互動模式）。
- `spam_classifier/streamlit_app.py`：Streamlit 網頁 Demo，可單筆與批次預測。
- `requirements.txt`：部署與重現所需 Python 套件。

### 快速開始
```powershell
python .\spam_classifier\train.py
streamlit run .\spam_classifier\streamlit_app.py
```

### 部署到 Streamlit Cloud
1. 推送此專案（包含 `models/spam_model.joblib`）到 GitHub。
2. 在 Streamlit Cloud 建立 App，路徑：`spam_classifier/streamlit_app.py`。
3. 自動安裝 `requirements.txt` 後即可使用。

### 授權 / 用途
作業示範用途；可擴充模型、加入評估視覺化或資料清理程序。
