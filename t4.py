#!/usr/bin/env python3
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib
import os

# 设置绘图后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. 工具函数：因子处理的核心逻辑
# ==========================================


def winsorize_mad(x, n=3.0):
    """
    MAD 去极值法
    x: Series, 某一天的所有股票因子值
    n: 几倍的 MAD，通常用 3 或 5
    """
    # 1. 找中位数
    median = x.median()
    # 2. 算每个数离中位数的距离 (绝对偏差)
    mad = (x - median).abs().median()

    # 3. 确定上下界
    # 如果 mad 为 0 (比如大量停牌或大量相同值)，直接返回原值，避免除 0 错误
    if mad == 0:
        return x

    # 通常用 1.4826 这个系数来近似正态分布的标准差
    upper_limit = median + n * mad * 1.4826
    lower_limit = median - n * mad * 1.4826

    # 4. 盖帽 (Clip): 超过上界的变成上界，低于下界的变成下界
    return x.clip(lower=lower_limit, upper=upper_limit)


def standardize_zscore(x):
    """
    Z-Score 标准化
    公式: (x - mean) / std
    """
    return (x - x.mean()) / x.std()


# ==========================================
# 2. 数据获取 (复用之前的缓存逻辑)
# ==========================================
CACHE_FILE = "stock_data_cache.pkl"


def get_data():
    if os.path.exists(CACHE_FILE):
        print(f"⚡ 读取缓存: {CACHE_FILE}")
        return pd.read_pickle(CACHE_FILE)
    else:
        # 这里为了演示方便，如果没缓存就报错，请先运行 t3.py 生成缓存
        # 或者你可以把 t3.py 的下载逻辑贴过来
        raise FileNotFoundError("请先运行 t3.py 生成数据缓存！")


df = get_data()

# ==========================================
# 3. 构造原始因子
# ==========================================
print("正在计算原始因子...")
# 还是用 20日动量
df["factor_raw"] = df.groupby(level="ticker")["close"].pct_change(20)

# 预处理：算收益、去空值
df["next_ret"] = df.groupby(level="ticker")["close"].pct_change().shift(-1)
df = df.dropna()

print(f"原始数据行数: {len(df)}")

# ==========================================
# 4. 核心步骤：截面清洗因子 (Cross-Sectional Processing)
# ==========================================
print("正在清洗因子 (MAD去极值 -> ZScore标准化)...")

# 关键知识点：因子处理必须在“每一天”内部进行！
# 不能把 2023年 和 2015年 的数据混在一起标准化。


def process_one_day(x):
    """
    输入 x: 某一天的因子 Series
    输出: 清洗后的 Series
    """
    # 1. 去极值 (先做！)
    x_win = winsorize_mad(x, n=3.0)

    # 2. 标准化 (后做！)
    x_std = standardize_zscore(x_win)

    return x_std


# 使用 groupby('date') 逐日处理
# group_keys=False 防止索引层级爆炸
df["factor_processed"] = df.groupby(level="date", group_keys=False)["factor_raw"].apply(
    process_one_day
)

# ==========================================
# 5. 可视化对比：看分布直方图 (Histogram)
# ==========================================
print("正在绘制因子分布对比图...")

# 我们随机选一天来看看效果
sample_date = df.index.get_level_values("date")[50]  # 第50个交易日
data_day = df.xs(sample_date, level="date")  # 取出这一天的数据

plt.figure(figsize=(12, 5))

# 左图：原始因子
plt.subplot(1, 2, 1)
plt.hist(data_day["factor_raw"], bins=30, color="gray", alpha=0.7)
plt.title(f"Raw Factor Distribution ({sample_date.date()})")
plt.xlabel("Factor Value")
plt.grid(True, alpha=0.3)

# 右图：清洗后的因子
plt.subplot(1, 2, 2)
plt.hist(data_day["factor_processed"], bins=30, color="blue", alpha=0.7)
plt.title(f"Processed (MAD+ZScore) ({sample_date.date()})")
plt.xlabel("Standard Deviations")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("factor_distribution_compare.png")
print(">> 分布对比图已保存: factor_distribution_compare.png")

# ==========================================
# 6. IC 对比测试
# ==========================================
print("\n正在对比 IC 表现...")

# 算原始因子的 IC
ic_raw = df.groupby(level="date").apply(
    lambda x: x["factor_raw"].corr(x["next_ret"], method="spearman")
)

# 算清洗后因子的 IC
# 注意：如果是 Spearman (Rank IC)，去极值和标准化的影响其实很小（因为排名没变）
# 但如果是 Pearson (普通 IC)，清洗后的效果会提升很明显
ic_processed = df.groupby(level="date").apply(
    lambda x: x["factor_processed"].corr(x["next_ret"], method="spearman")
)

print(f"原始因子 IC均值: {ic_raw.mean():.4f} | ICIR: {ic_raw.mean()/ic_raw.std():.4f}")
print(
    f"清洗因子 IC均值: {ic_processed.mean():.4f} | ICIR: {ic_processed.mean()/ic_processed.std():.4f}"
)

print("\n任务完成！请打开 factor_distribution_compare.png 看看你的因子变漂亮了吗？")
