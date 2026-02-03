#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib

# 设置绘图后端，防止服务器报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. 制造模拟数据 (Mock Data)
# ==========================================
print("正在生成模拟数据...")

# 50 只股票，100 天
tickers = [f"Stock_{i:02d}" for i in range(50)]
dates = pd.date_range(start="2023-01-01", periods=100)

index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
df = pd.DataFrame(index=index)

np.random.seed(42)
# 构造独立走势
df["noise"] = np.random.randn(len(df))
df["close"] = df.groupby(level="ticker")["noise"].transform(lambda x: x.cumsum() + 100)
del df["noise"]

# 构造因子
df["factor"] = np.random.randn(len(df))

# ==========================================
# 2. 数据预处理
# ==========================================
print("正在处理数据...")

# 计算下期收益
df["next_ret"] = df.groupby(level="ticker")["close"].pct_change().shift(-1)

# 删除空值
df = df.dropna()

# ==========================================
# 3. IC 分析
# ==========================================
print("正在计算 IC...")

try:
    ic_series = df.groupby(level="date").apply(
        lambda x: x["factor"].corr(x["next_ret"], method="spearman")
    )
except Exception as e:
    print(f"IC计算出错: {e}")
    exit()

print(f"\n=== IC 分析结果 ===")
print(f"IC 均值: {ic_series.mean():.4f}")
print(f"ICIR: {ic_series.mean() / ic_series.std():.4f}")

# 画图
plt.figure(figsize=(10, 4))
ic_series.plot(title="Factor IC Series")
plt.axhline(0, color="red", linestyle="--")
plt.tight_layout()
plt.savefig("result_ic.png")
print(">> IC图已保存: result_ic.png")
plt.close()

# ==========================================
# 4. 分层回测 (Layered Backtest)
# ==========================================
print("\n正在进行分层回测...")


def get_groups(x):
    if len(x) < 5:
        return pd.Series([np.nan] * len(x), index=x.index)

    # 使用 rank 解决 duplicates 报错
    return pd.qcut(x.rank(method="first"), 5, labels=["G1", "G2", "G3", "G4", "G5"])


# 【关键修复】 加上 group_keys=False，防止索引变成 3 层导致报错
df["group"] = df.groupby(level="date", group_keys=False)["factor"].apply(get_groups)

# 计算分组收益
group_ret = df.groupby(["date", "group"], observed=True)["next_ret"].mean().unstack()

# 计算累积净值
group_cum_ret = (1 + group_ret).cumprod()

print("\n=== 分层回测结果 (累积净值) ===")
print(group_cum_ret.iloc[-1])

# 画图
plt.figure(figsize=(10, 4))
for col in group_cum_ret.columns:
    plt.plot(group_cum_ret.index, group_cum_ret[col], label=col)

long_short = group_cum_ret["G5"] - group_cum_ret["G1"]
plt.plot(
    long_short.index,
    long_short,
    label="Long-Short",
    color="black",
    linestyle="--",
    linewidth=2,
)

plt.title("Layered Backtest")
plt.legend()
plt.tight_layout()
plt.savefig("result_backtest.png")
print(">> 回测图已保存: result_backtest.png")
plt.close()

print("\n全部完成！")
