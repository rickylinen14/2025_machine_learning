import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re

# ===== 1. 讀取 XML =====
xml_file = r"C:/Users/tom85/OneDrive/桌面/研究所/機器學習/2025_machine_learning/Week_4/O-A0038-003.xml"  # 改成你的 XML 路徑
tree = ET.parse(xml_file)
root = tree.getroot()

# Namespace
ns = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}

# ===== 2. 找到 <Content> 數據 =====
content = root.find(".//cwa:Content", ns).text.strip()
if content.startswith('"') and content.endswith('"'):
    content = content[1:-1]

# 切分數據（處理逗號、空白、換行）
values = re.split(r"[\s,]+", content.strip())
values = [float(v) for v in values if v != ""]

# ===== 3. 基本參數 =====
nx, ny = 67, 120  # 經向、緯向格點數
lon0, lat0 = 120.00, 21.88  # 左下角座標
res = 0.03  # 經緯度解析度

# 轉成矩陣 (ny, nx)
grid = np.array(values).reshape(ny, nx)

# ===== 4. 建立經緯度座標網格 =====
lons = [lon0 + i * res for i in range(nx)]
lats = [lat0 + j * res for j in range(ny)]
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ===== 5. 建立 Classification 資料集 =====
# 格式：(經度, 緯度, label)
labels = np.where(grid == -999.0, 0, 1)
df_class = pd.DataFrame({
    "longitude": lon_grid.ravel(),
    "latitude": lat_grid.ravel(),
    "label": labels.ravel()
})

# ===== 6. 建立 Regression 資料集 =====
# 格式：(經度, 緯度, Value) 只保留有效值
mask = grid != -999.0
df_reg = pd.DataFrame({
    "longitude": lon_grid[mask],
    "latitude": lat_grid[mask],
    "value": grid[mask]
})

# ===== 7. 輸出成檔案 =====
df_class.to_csv(r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\classification_dataset.csv", index=False)
df_reg.to_csv(r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\regression_dataset.csv", index=False)

print("✅ 已建立兩個資料集：classification_dataset.csv, regression_dataset.csv")
