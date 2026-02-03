#!/usr/bin/env python3
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib
import os
from datetime import datetime

# 设置绘图后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区域
# ==========================================
START_DATE = "20230101"
END_DATE = "20260105"
CACHE_FILE = "stock_data_cache.pkl"  # 本地缓存文件名

# ==========================================
# 2. 数据获取 (带缓存机制)
# ==========================================


def get_stock_data():
    """
    智能数据获取函数：
    1. 检查本地有没有缓存文件
    2. 有 -> 直接读取 (0.1秒)
    3. 无 -> 联网下载 (需要时间) -> 存入缓存
    """
    if os.path.exists(CACHE_FILE):
        print(f"⚡ 发现本地缓存 {CACHE_FILE}，正在读取...")
        df = pd.read_pickle(CACHE_FILE)
        # 方法：去索引(Index)里把 'ticker' 这一层的值取出来
        print(
            f"✅ 读取成功！包含 {len(df.index.get_level_values('ticker').unique())} 只股票，共 {len(df)} 行数据。"
        )
        return df

    print("🌐 本地无缓存，准备开始下载...")

    # --- 【选择股票池】 ---

    # 选项 A: 沪深300成分股 (推荐！300只，比较快，约5-10分钟)
    print("正在获取沪深300成分股列表...")
    stock_list_df = ak.index_stock_cons(symbol="000300")
    stock_list_df.info()
    # 格式化代码，akshare通常用6位数字
    target_codes = stock_list_df["品种代码"].tolist()

    # 选项 B: 全A股 (慎用！5000+只，耗时约1-2小时)
    # 如果你真想跑全市场，取消下面几行的注释，注释掉上面的 选项 A
    # print("正在获取全A股列表...")
    # stock_list_df = ak.stock_zh_a_spot_em()
    # target_codes = stock_list_df["代码"].tolist()

    print(f"目标股票数量: {len(target_codes)} 只")

    data_list = []
    total = len(target_codes)

    for i, code in enumerate(target_codes):
        # 打印进度条
        if i % 10 == 0:
            print(f"进度: {i}/{total} ...")

        try:
            # 下载日线行情 (前复权)
            df_temp = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="qfq",
            )

            # 只有下载到数据才处理
            if not df_temp.empty:
                # 重命名列
                df_temp = df_temp.rename(
                    columns={"日期": "date", "收盘": "close", "成交量": "volume"}
                )
                # 保留有用列
                df_temp = df_temp[["date", "close", "volume"]]
                # 加上 ticker 列
                df_temp["ticker"] = code

                data_list.append(df_temp)

        except Exception as e:
            # 个别股票下载失败很正常(停牌等)，跳过即可
            pass

    print("合并数据中...")
    df_all = pd.concat(data_list)

    # 格式整理
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.set_index(["date", "ticker"]).sort_index()

    # 存入缓存！下次就不用下载了
    print(f"💾 正在保存缓存到 {CACHE_FILE} ...")
    df_all.to_pickle(CACHE_FILE)

    return df_all


# 执行获取数据
df = get_stock_data()

# ==========================================
# 3. 因子计算 (Factor Calculation)
# ==========================================
print("正在计算因子 (20日动量)...")

# 真实数据通常会有停牌导致的空值，计算前可以先 dropna 一次，也可以计算后 drop
# 动量因子：(今天收盘 / 20天前收盘) - 1
df["factor"] = df.groupby(level="ticker")["close"].pct_change(20)

# ==========================================
# 4. 预处理 (计算收益、清洗)
# ==========================================
print("正在计算下期收益...")
df["next_ret"] = df.groupby(level="ticker")["close"].pct_change().shift(-1)

# 清洗数据
# 1. 删除因子计算产生的 NaN (前20天)
# 2. 删除 next_ret 产生的 NaN (最后1天)
df = df.dropna()

print(f"清洗后剩余数据行数: {len(df)}")

# ==========================================
# 5. IC 分析
# ==========================================
print("正在计算 IC...")

ic_series = df.groupby(level="date").apply(
    lambda x: x["factor"].corr(x["next_ret"], method="spearman")
)

print(f"\n=== IC 分析结果 ===")
print(f"IC 均值: {ic_series.mean():.4f}")
print(f"ICIR: {ic_series.mean() / ic_series.std():.4f}")
print(f"IC > 0 的比例: {(ic_series > 0).mean():.2%}")

# 画 IC 图
plt.figure(figsize=(10, 4))
# rolling(20) 是画一根20日的移动平均线，让趋势更清楚
ic_series.plot(alpha=0.4, label="Daily IC")
ic_series.rolling(20).mean().plot(color="red", label="20D MA", linewidth=2)
plt.axhline(0, color="black", linestyle="--")
plt.title("Momentum Factor IC (CSI 300)")
plt.legend()
plt.tight_layout()
plt.savefig("full_ic.png")
print(">> IC图已保存: full_ic.png")

# ==========================================
# 6. 分层回测 (Layered Backtest)
# ==========================================
print("\n正在进行分层回测...")


def get_groups(x):
    # 现在我们有300只股票了，完全可以放心地分 5 组，甚至 10 组
    # 如果数据量少于10只，返回空
    if len(x) < 10:
        return pd.Series([np.nan] * len(x), index=x.index)

    # rank method='first' 保证严格均匀分组
    return pd.qcut(
        x.rank(method="first"),
        30,
        labels=[f"G{i+1}" for i in range(30)],
    )


df["group"] = df.groupby(level="date", group_keys=False)["factor"].apply(get_groups)

# 计算分组收益
group_ret = df.groupby(["date", "group"], observed=True)["next_ret"].mean()
group_ret.info()
group_ret = group_ret.unstack()
group_ret.info()
# 计算累积净值 (Long Only)
group_cum_ret = (1 + group_ret).cumprod()

print("\n=== 分层回测结果 (累积净值) ===")
print(group_cum_ret.iloc[-1])

# 画回测图
plt.figure(figsize=(10, 4))
for col in group_cum_ret.columns:
    plt.plot(group_cum_ret.index, group_cum_ret[col], label=col)

# 多空对冲 (Long High - Short Low)
# 动量因子通常预期 High 涨得好，Low 涨得差
long_short = group_cum_ret["G30"] - group_cum_ret["G1"]
plt.plot(
    long_short.index,
    long_short,
    label="Long-Short",
    color="black",
    linestyle="--",
    linewidth=2,
)

plt.title("Backtest: Momentum on CSI 300")
plt.legend()
plt.tight_layout()
plt.savefig("full_backtest.png")
print(">> 回测图已保存: full_backtest.png")
