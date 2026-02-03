import pandas as pd
import numpy as np

# 行索引：城市、年份
index = pd.MultiIndex.from_product(
    [["Beijing", "Shanghai"], [2022, 2023]], names=["City", "Year"]
)

# 列索引：产品、指标
columns = pd.MultiIndex.from_product(
    [["Phone", "PC"], ["Price", "Sales"]], names=["Product", "Metric"]
)

# 生成随机数据
data = np.random.randint(100, 1000, size=(4, 4))

df_complex = pd.DataFrame(data, index=index, columns=columns)

print(df_complex)
