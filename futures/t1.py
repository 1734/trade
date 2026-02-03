import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime


# ==========================================
# 1. 回测引擎 (复用之前的逻辑，无需改动)
# ==========================================
class FuturesBacktest:
    def __init__(
        self,
        data,
        initial_capital=100000,
        commission_rate=0.0001,
        slippage=1,
        contract_multiplier=10,
    ):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.multiplier = contract_multiplier
        self.results = None

    def strategy_ma_crossover(self, short_window=20, long_window=60):
        # 简单的双均线策略
        self.data["Short_MA"] = self.data["Close"].rolling(window=short_window).mean()
        self.data["Long_MA"] = self.data["Close"].rolling(window=long_window).mean()

        self.data["Signal"] = 0
        # 短线上穿长线做多(1)，下穿做空(-1)
        self.data["Signal"] = np.where(
            self.data["Short_MA"] > self.data["Long_MA"], 1, -1
        )

        # 信号下移一格，避免未来函数
        self.data["Position"] = self.data["Signal"].shift(1)
        self.data.dropna(inplace=True)

    def run_backtest(self):
        df = self.data.copy()
        # 价格变化
        df["Price_Change"] = df["Close"] - df["Close"].shift(1)
        # 持仓盈亏 (Mark to Market)
        df["Daily_PnL_Raw"] = df["Price_Change"] * df["Position"] * self.multiplier

        # 交易成本
        df["Trade_Action"] = df["Position"].diff().abs()
        commission_cost = (
            df["Trade_Action"] * df["Close"] * self.multiplier * self.commission_rate
        )
        slippage_cost = df["Trade_Action"] * self.slippage * self.multiplier
        df["Total_Cost"] = commission_cost + slippage_cost

        # 净值计算
        df["Net_PnL"] = df["Daily_PnL_Raw"] - df["Total_Cost"]
        df["Equity"] = self.initial_capital + df["Net_PnL"].cumsum()

        # 简单的回撤计算
        df["Peak"] = df["Equity"].cummax()
        df["Drawdown"] = (df["Equity"] - df["Peak"]) / df["Peak"]

        self.results = df
        return df

    def plot_results(self):
        if self.results is None:
            return
        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.results.index, self.results["Equity"], color="red")
        ax1.set_title("Account Equity")
        ax1.grid(True)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(self.results.index, self.results["Close"], color="black", alpha=0.6)
        ax2.plot(
            self.results.index, self.results["Short_MA"], color="orange", alpha=0.8
        )
        ax2.plot(self.results.index, self.results["Long_MA"], color="green", alpha=0.8)
        ax2.set_title("Price & MA")
        ax2.grid(True)
        plt.show()


# ==========================================
# 2. AkShare 数据获取模块 (核心修改部分)
# ==========================================
def get_real_futures_data(symbol="rb0", start_date="20200101", end_date="20231231"):
    """
    获取期货主连数据并清洗格式
    :param symbol: 合约代码，例如 'rb0' (螺纹主连), 'm0' (豆粕主连)
    """
    print(f"正在从 AkShare 获取 {symbol} 的数据...")

    # 使用 ak.futures_main_sina 获取新浪期货主连数据
    # 注意：AkShare 接口更新较快，如果报错请检查 akshare 版本
    try:
        df = ak.futures_main_sina(symbol=symbol)
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

    # 1. 数据清洗：重命名列以匹配回测框架
    # AkShare 返回通常是中文列名：['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量', ...]
    rename_map = {
        "日期": "Date",
        "开盘价": "Open",
        "最高价": "High",
        "最低价": "Low",
        "收盘价": "Close",
        "成交量": "Volume",
    }
    df.rename(columns=rename_map, inplace=True)

    # 2. 处理日期索引
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # 3. 确保数据类型为 float
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 4. 筛选时间段
    # 转换输入的时间字符串为 datetime 对象
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df_final = df.loc[mask].copy()

    if df_final.empty:
        print("警告：筛选后数据为空，请检查日期范围。")
    else:
        print(f"成功获取数据：{len(df_final)} 条 K线")

    return df_final


# ==========================================
# 3. 主程序运行
# ==========================================
if __name__ == "__main__":
    # --- 配置参数 ---
    symbol = "rb0"  # 螺纹钢主连
    multiplier = 10  # 螺纹钢合约乘数 10吨/手
    initial_cash = 50000  # 初始资金 5万

    # 1. 拉取真实数据
    # 获取最近3年的数据
    today = datetime.date.today().strftime("%Y%m%d")
    df_real = get_real_futures_data(
        symbol=symbol, start_date="20230101", end_date=today
    )

    if df_real is not None:
        # 2. 初始化回测
        # 螺纹钢手续费通常是万分之1左右，滑点设为1元
        bt = FuturesBacktest(
            df_real,
            initial_capital=initial_cash,
            commission_rate=0.0001,
            slippage=1,
            contract_multiplier=multiplier,
        )

        # 3. 运行策略 (例如 10日和60日均线)
        bt.strategy_ma_crossover(short_window=10, long_window=20)

        # 4. 执行回测
        results = bt.run_backtest()

        # 5. 输出结果
        final_equity = results["Equity"].iloc[-1]
        ret = (final_equity - initial_cash) / initial_cash
        print(f"\n回测结束: {symbol}")
        print(f"初始资金: {initial_cash}")
        print(f"最终权益: {final_equity:.2f}")
        print(f"总收益率: {ret*100:.2f}%")

        # 6. 绘图
        bt.plot_results()
