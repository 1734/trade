import akshare as ak
import pandas as pd
import mplfinance as mpf


# ==========================================
# 1. 获取数据
# ==========================================
def get_oil_data_2022():
    print("正在从新浪财经获取 NYMEX原油(CL) 历史数据...")

    # 获取外盘期货历史行情，"CL" 代表 WTI原油
    try:
        df = ak.futures_foreign_hist(symbol="CL")
    except Exception as e:
        print(f"数据获取失败，请检查网络: {e}")
        return None

    # --- 数据清洗 ---
    # 1. 重命名列以适配 mplfinance (必须是首字母大写的英文)
    df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    # 2. 设置日期索引
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # 3. 精准截取 2022 年数据
    # 使用 Pandas 的字符串切片功能，非常方便
    df_2022 = df.loc["2022-01-01":"2022-12-31"]

    print(f"成功获取 2022 年数据，共 {len(df_2022)} 个交易日。")
    return df_2022


# ==========================================
# 2. 绘图
# ==========================================
def plot_chart(df):
    if df is None or df.empty:
        print("无数据，无法绘图。")
        return

    # 设置符合国人习惯的配色：红涨绿跌
    mc = mpf.make_marketcolors(
        up="red", down="green", edge="i", wick="i", volume="in", inherit=True
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="--", y_on_right=True)

    print("正在绘图...")
    mpf.plot(
        df,
        type="candle",  # 蜡烛图
        volume=True,  # 【关键】开启成交量副图
        mav=(5, 20, 60),  # 添加 5/20/60 日均线
        title="\nNYMEX WTI Crude Oil (2022)",
        ylabel="Price ($)",
        ylabel_lower="Volume",  # 成交量Y轴标签
        style=style,  # 应用红涨绿跌风格
        figratio=(16, 9),  # 图片长宽比
        figscale=1.2,  # 图片放大
        datetime_format="%Y-%m-%d",  # 时间轴格式
        tight_layout=True,
        show_nontrading=False,  # 去除周末和节假日空隙，让K线连续
    )


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    data = get_oil_data_2022()

    # 2. 画图
    plot_chart(data)

    # 3. (可选) 打印前5行数据查看
    if data is not None:
        print("\n数据预览:")
        print(data.head())
