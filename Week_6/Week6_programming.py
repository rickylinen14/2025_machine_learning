# ============================================================
# Week6: Gaussian Discriminant Analysis & Piecewise Regression
# Author: Ricky Lin (Final Refined Version)
# Date  : 2025-10
# Note  : Custom implementation of QDA and Polynomial Regression.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma # <--- 新增：為了遮罩 -999
from sklearn.model_selection import train_test_split

# ======================================================
# 1. 讀取與準備資料
# ======================================================
path_class = r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\classification_dataset.csv"
path_reg   = r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\regression_dataset.csv"

df_class = pd.read_csv(path_class)
df_reg   = pd.read_csv(path_reg)

# 分類任務資料
X_class = df_class[["longitude", "latitude"]].values
y_class = df_class["label"].values
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# 迴歸任務資料
X_reg = df_reg[["longitude", "latitude"]].values
y_reg = df_reg["value"].values

print(f"分類訓練集大小: {Xc_train.shape}, 測試集大小: {Xc_test.shape}")
print(f"迴歸資料集大小: {X_reg.shape}")

# ======================================================
# 2. QDA 分類模型
# ======================================================
class MyQDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {}
        self.covs_ = {}
        self.priors_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = np.mean(X_c, axis=0)
            self.covs_[c] = np.cov(X_c, rowvar=False)
            self.priors_[c] = len(X_c) / len(X)
        return self # 讓 fit 可以鏈式操作

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes_:
                mean, cov, prior = self.means_[c], self.covs_[c], self.priors_[c]
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                diff = x - mean
                log_prob = -0.5 * np.log(det_cov) \
                           -0.5 * diff.T @ inv_cov @ diff \
                           + np.log(prior)
                posteriors.append(log_prob)
            preds.append(self.classes_[np.argmax(posteriors)])
        return np.array(preds)

# --- 計算 NLL 的獨立函式 ---
def calculate_nll(X, y, model):
    """計算給定資料的負對數概似 (NLL)"""
    total_log_prob = 0
    num_features = X.shape[1]
    for i in range(len(X)):
        x_i, y_i = X[i], y[i]
        mean = model.means_[y_i]
        cov = model.covs_[y_i]
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        diff = x_i - mean
        
        # 多變量高斯分佈的完整對數機率密度
        log_pdf = -0.5 * np.log(det_cov) \
                  -0.5 * diff.T @ inv_cov @ diff \
                  - (num_features / 2) * np.log(2 * np.pi)
        
        total_log_prob += log_pdf
        
    return -total_log_prob / len(X)

# --- 訓練與評估 QDA ---
qda = MyQDA()
qda.fit(Xc_train, yc_train)

# --- 計算並印出 Loss ---
train_loss = calculate_nll(Xc_train, yc_train, qda)
test_loss = calculate_nll(Xc_test, yc_test, qda)

y_pred = qda.predict(Xc_test)
accuracy = np.mean(y_pred == yc_test)

print(f"\nTrain Loss (NLL): {train_loss:.4f}")
print(f"Test Loss (NLL): {test_loss:.4f}")
print(f"QDA Test Accuracy: {accuracy*100:.2f}%")

# ======================================================
# 3. 二次多項式迴歸模型
# ======================================================
class MyPolynomialRegressor:
    def _create_features(self, X):
        lon, lat = X[:, 0], X[:, 1]
        return np.c_[np.ones(len(lon)), lon, lat, lon*lat, lon**2, lat**2]

    def fit(self, X, y):
        X_poly = self._create_features(X)
        self.theta_ = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
        return self

    def predict(self, X):
        X_poly = self._create_features(X)
        return X_poly @ self.theta_

# --- 訓練迴歸模型 ---
regressor = MyPolynomialRegressor()
regressor.fit(X_reg, y_reg)

# ======================================================
# 4. 組合分段模型 h(x)
# ======================================================
def combined_model(X, classifier, regressor):
    class_pred = classifier.predict(X)
    reg_pred = np.zeros_like(class_pred, dtype=float)
    mask = (class_pred == 1)
    if np.any(mask):
        reg_pred[mask] = regressor.predict(X[mask])
    return np.where(mask, reg_pred, -999.0)

# ======================================================
# 5. 可視化
# ======================================================
# --- 準備繪圖網格 ---
lon_min, lon_max = X_class[:, 0].min() - 0.1, X_class[:, 0].max() + 0.1
lat_min, lat_max = X_class[:, 1].min() - 0.1, X_class[:, 1].max() + 0.1
lon_grid, lat_grid = np.meshgrid(np.linspace(lon_min, lon_max, 200),
                                 np.linspace(lat_min, lat_max, 200))
X_grid = np.c_[lon_grid.ravel(), lat_grid.ravel()]

# --- 圖一：QDA 決策邊界 ---
Z_qda = qda.predict(X_grid).reshape(lon_grid.shape)
plt.figure(figsize=(8, 6))
plt.contourf(lon_grid, lat_grid, Z_qda, cmap='coolwarm', alpha=0.7)
# --- 繪製訓練集資料點，以顯示台灣輪廓 ---
plt.scatter(Xc_train[:, 0], Xc_train[:, 1], c=yc_train, s=5, cmap='coolwarm', alpha=0.5)
plt.title("QDA Decision Boundary (on Training Data)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# --- 圖二：組合模型 h(x) 輸出 ---
Z_h = combined_model(X_grid, qda, regressor).reshape(lon_grid.shape)
# --- 使用遮罩 (masking) 來處理 -999 ---
Z_h_masked = ma.masked_where(Z_h == -999, Z_h)

plt.figure(figsize=(8, 6))
# --- 繪製遮罩後的資料，並更換 cmap ---
contour = plt.contourf(lon_grid, lat_grid, Z_h_masked, levels=100, cmap='jet')
plt.colorbar(contour, label="Predicted Value")
plt.title("Piecewise Regression Surface h(x)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()