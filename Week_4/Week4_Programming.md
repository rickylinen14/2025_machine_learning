## 0. 前言

本次作業需先將原始資料轉換為兩個可供監督式學習使用的資料集。為此，我撰寫了 `Week4_Programming1.py`，透過程式將原始資料處理並輸出為 **`regression_dataset.csv`** 與 **`classification_dataset.csv`**。

接著，於 `Week4_Programming2.py` 中，分別載入上述兩個資料集，進行分類與回歸模型的訓練與評估。

另外，為了更直觀地理解原始 **.xml** 格式資料在二維空間中的結構，我額外撰寫了 `Dataform.py`，將資料轉換並輸出為 **`output_data.xlsx`**，以利檢視與分析。

## 1. 研究目標與資料

本次實驗的目標為利用格點資料建立兩種機器學習模型：

1. **分類模型 (Classification)**：  
    以經度 (lon)、緯度 (lat) 預測該格點是否為有效值 (label = 0 或 1)。
    
2. **回歸模型 (Regression)**：  
    以經度 (lon)、緯度 (lat) 預測對應的溫度觀測值 (value)。
    
原始資料分別存放於兩個 CSV 檔案，格式如下：

- `classification_dataset.csv` → (lon, lat, label)
- `regression_dataset.csv` → (lon, lat, value)

---
## 2. 方法與模型

### 2.1 資料處理

- 使用 **Pandas** 讀取 CSV 檔案，將欄位重新命名為 `lon, lat, label` 與 `lon, lat, value`。
- 使用 `train_test_split` 劃分訓練集與測試集 (test_size=0.2)。

### 2.2 初始模型設計

在第一次嘗試時，為了保持模型簡單，分別選擇了：

- **分類任務**：邏輯迴歸 (Logistic Regression)
- **回歸任務**：線性迴歸 (Linear Regression)

---
## 3. 實驗過程與調整

### 3.1 初始結果

使用邏輯迴歸與線性迴歸後，得到以下結果：

- **分類模型 (Logistic Regression)**
    - 準確率 (Accuracy)：≈ 57%
    - 混淆矩陣顯示模型幾乎無法正確辨識「有效值=1」的樣本。
- **回歸模型 (Linear Regression)**
    - 均方誤差 (MSE)：≈ 32
    - 決定係數 (R²)：≈ 0.05

👉 結果顯示：線性模型難以捕捉座標 (lon, lat) 與標籤或數值之間的非線性關係，模型表現不理想。

---
### 3.2 調整方向

- 改用 RandomForest 作為核心模型。
- 保持資料分割方式 (train/test split) 不變，以確保結果具可比性。
- 引入模型可解釋的視覺化：
    - **混淆矩陣 (Confusion Matrix)**，檢視分類誤差。
    - **真實值 vs. 預測值散佈圖**，觀察回歸模型的擬合程度。

---
### 3.3 **最終模型訓練**

- RandomForestClassifier 與 RandomForestRegressor 皆成功訓練並得到穩定表現。
---
## 4. 分析與討論

![[Pasted image 20250929220333.png]]
### (a) 分類模型結果

- **準確率 (Accuracy)**: **98.4%**
- **分類報告**: precision、recall、f1-score 幾乎均在 **0.98–0.99**。
- **混淆矩陣觀察**: 僅有少數誤判 (8 個 0 → 1，17 個 1 → 0)，顯示模型幾乎能正確分類所有樣本。
### (b) 回歸模型結果

- **均方誤差 (MSE)**: **4.82**
- **R² 分數**: **0.86**
- **散佈圖觀察**: 散點緊貼紅色理想線，顯示模型對於真實值的擬合能力強。
### (c) 比較

- 初始模型：表現不足，幾乎無法有效預測。
- RandomForest：顯著提升效能，說明其在處理非線性與高維度特徵的優勢。
---
## 5. 結論

1. 初始嘗試的基礎模型表現不佳，分類準確率僅約 57%，回歸的解釋力幾乎為零。
2. 引入 **RandomForest** 後，模型效能大幅提升：
    - 分類準確率提升至 **98%+**，幾乎能完美預測。
    - 回歸 R² 提升至 **0.86**，能有效解釋資料變異。
3. 本次實驗完整呈現了「嘗試 → 發現不足 → 模型改進 → 成效提升」的過程，體現了機器學習模型開發中的調整思路。
---
## 6. 相關圖表 (附錄)

- **分類模型混淆矩陣**  
    ![[classification_confusion_matrix.png]]
- **回歸模型真實值 vs. 預測值![[regression_true_vs_pred.png]]**