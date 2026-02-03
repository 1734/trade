import pandas as pd
import numpy as np

# 造一个小表
df = pd.DataFrame(
    data=[[100, 0.5], [102, 0.6]],  # 3. 内容 (Content)
    index=["2023-01-01", "2023-01-02"],  # 1. 行索引 (Index)
    columns=["close", "factor"],  # 2. 列索引 (Columns)
)

print(f"1. 行索引 (Index):   {df.index}")
print(f"2. 列索引 (Columns): {df.columns}")
print(f"3. 内容 (Values):    \n{df.values}")
