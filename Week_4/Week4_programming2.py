import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# ========== 1. 載入資料 ==========
path_class = r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\classification_dataset.csv"
path_reg   = r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\regression_dataset.csv"

df_class = pd.read_csv(path_class, header=None, skiprows=1)
df_reg   = pd.read_csv(path_reg, header=None, skiprows=1)

df_class.columns = ["lon", "lat", "label"]
df_reg.columns   = ["lon", "lat", "value"]

# ========== 2. 分類模型 ==========
X_class = df_class[["lon", "lat"]]
y_class = df_class["label"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(Xc_train, yc_train)

yc_pred = clf.predict(Xc_test)

print("\n=== 分類模型 (RandomForest) ===")
print("✅ 準確率:", accuracy_score(yc_test, yc_pred))
print("混淆矩陣:\n", confusion_matrix(yc_test, yc_pred))
print("分類報告:\n", classification_report(yc_test, yc_pred))

# --- 圖 1: 混淆矩陣熱力圖 ---
cm = confusion_matrix(yc_test, yc_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Classification Confusion Matrix")
plt.savefig(r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\classification_confusion_matrix.png", dpi=300)
plt.close()

# ========== 3. 回歸模型 ==========
X_reg = df_reg[["lon", "lat"]]
y_reg = df_reg["value"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = RandomForestRegressor(random_state=42)
reg.fit(Xr_train, yr_train)

yr_pred = reg.predict(Xr_test)

print("\n=== 回歸模型 (RandomForest) ===")
print("✅ 均方誤差 (MSE):", mean_squared_error(yr_test, yr_pred))
print("✅ R² 分數:", r2_score(yr_test, yr_pred))

# --- 圖 2: 真實值 vs. 預測值 ---
plt.figure(figsize=(6,6))
plt.scatter(yr_test, yr_pred, alpha=0.5)
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Regression: True vs. Predicted")
plt.savefig(r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\regression_true_vs_pred.png", dpi=300)
plt.close()
