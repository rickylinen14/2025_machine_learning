import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ===== 1. 載入 XML 檔案 =====
xml_file = r"C:/Users/tom85/OneDrive/桌面/研究所/機器學習/2025_machine_learning/Week_4/O-A0038-003.xml"  # 改成你的檔案路徑
tree = ET.parse(xml_file)
root = tree.getroot()

# ===== 2. 處理 namespace =====
ns = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}

# ===== 3. 找到數據 (在 <Content>) =====
content = root.find(".//cwa:Content", ns).text.strip()

# 去掉頭尾引號（有些資料會包在 "" 裡）
if content.startswith('"') and content.endswith('"'):
    content = content[1:-1]

# ===== 4. 拆分數據並轉成 float =====
# 先把換行符號替換掉
clean_content = content.replace("\n", ",").replace("\r", ",")
# 再依逗號切分
values = [v for v in clean_content.split(",") if v.strip() != ""]
values = list(map(float, values))

# ===== 5. 處理缺值 -999.0E+00 → None
values = [None if abs(v + 999) < 1e-6 else v for v in values] 

# ===== 6. 轉成 120 × 67 矩陣 =====
grid = np.array(values).reshape(120, 67)

# ===== 7. 存成 Excel =====
df = pd.DataFrame(grid)
output_file = r"C:\Users\tom85\OneDrive\桌面\研究所\機器學習\2025_machine_learning\Week_4\output_data.xlsx"  # 改成你想存的路徑 
df.to_excel(output_file, index=False)

print(f"✅ 已輸出 Excel 檔案: {output_file}")
