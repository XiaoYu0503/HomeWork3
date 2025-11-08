# SMS 垃圾簡訊分類器

使用 TF-IDF + Logistic Regression 建立簡單的 SMS Spam/Ham 分類模型。

## 環境需求
- Python 3.11
- 套件：pandas、scikit-learn、numpy、joblib

## 資料集
工作區根目錄的 `sms_spam_no_header.csv` 無表頭，假設：
```
label,text
spam,Win a free iPhone today...
ham,Hello how are you
```
實際檔案無表頭，因此程式內部以 `names=["label", "text"]` 讀取。

## 安裝套件
```powershell
pip install pandas scikit-learn numpy joblib
```
(已由系統安裝可略過)

## 訓練模型
於專案根目錄執行：
```powershell
python .\spam_classifier\train.py
```
輸出：
- 訓練/測試指標
- 模型儲存於 `spam_classifier/models/spam_model.joblib`

## 進行單筆預測
```powershell
python .\spam_classifier\predict.py "Free entry in a weekly cash prize draw"
```
或啟動互動模式：
```powershell
python .\spam_classifier\predict.py
```
輸入訊息後按 Enter，輸入 `/quit` 離開。

## 啟動 Streamlit 示範網頁
```powershell
streamlit run .\spam_classifier\streamlit_app.py
```

### 頁面功能（已內建）
- 單筆與批次預測（CSV 上傳）
- 標籤分佈與訊息長度統計＋直方圖
- 依模型權重的關鍵詞排行（Spam/Ham 各自 Top N，可於側邊調整數量）
- 詞雲（WordCloud）：Spam 與 Ham 各一張
- 評估圖表：混淆矩陣、ROC Curve（spam 為正類）
- 下載整體資料集的預測結果 CSV

## 部署到 Streamlit Cloud（streamlit.app）
1. 將此專案推到 GitHub（請包含 `models/spam_model.joblib` 與 `requirements.txt`）。
2. 登入 https://streamlit.io/ → Deploy an app → 連結你的 GitHub 儲存庫。
3. 選擇 Branch：`main`，App file path：`spam_classifier/streamlit_app.py`。
4. 其他設定維持預設即可，建立後系統會依 `requirements.txt` 自動安裝套件並啟動。
5. 若顯示找不到模型，請確認 `models/spam_model.joblib` 有一併推送至 GitHub。

小提醒：若要重新訓練並更新雲端模型，請在本機執行訓練、確認 `models/spam_model.joblib` 更新後推送至 GitHub，Cloud 會自動重新部署。

## 可能改進方向
- 使用中文或多語資料前需加斷詞與停用詞。
- 模型可改為 LinearSVC、Naive Bayes 或深度學習模型。
- 加入資料清理（URL、數字、Emoji 正規化）。
- 保存訓練/測試拆分及版本紀錄。

## 授權
僅供課堂作業練習使用。
