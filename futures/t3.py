import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime


# ==========================================
# 0. 全局配置中心 (Dual Thrust 专用配置)
# ==========================================
class StrategyConfig:
    # --- 账户与回测设置 ---
    SYMBOL = "cu0"  # 合约代码 (螺纹钢主连)
    INITIAL_CAPITAL = 500000  # 初始资金 (8万)
    MULTIPLIER = 5  # 合约乘数 (10吨/手)
    MARGIN_RATE = 0.12  # 保证金率 (12%)
    COMMISSION = 0.0001  # 手续费率 (万分之1)
    SLIPPAGE = 1  # 滑点 (1元)

    # --- Dual Thrust 策略参数 ---
    N_DAYS = 4  # 回溯周期 (计算过去N天的Range)
    K1 = 0.5  # 上轨系数 (Buy Trigger)
    K2 = 0.5  # 下轨系数 (Sell Trigger)

    # --- 风控参数 ---
    ATR_PERIOD = 14  # ATR计算周期
    ATR_STOP_MULTIPLIER = 2.5  # 吊灯止损倍数 (Dual Thrust波动大，止损可稍紧)
    RISK_PER_TRADE = 0.02  # 单笔风险敞口 (总资金的2%)


# ==========================================
# 1. 核心回测引擎
# ==========================================
class ProFuturesBacktest:
    def __init__(self, data, config):
        self.data = data.copy()
        self.cfg = config
        self.trade_log = []
        self.daily_stats = []

    def calculate_safe_lots(self, equity, price, atr_value):
        """基于ATR波动率倒算仓位"""
        if price <= 0 or atr_value <= 0:
            return 0

        # 1. 风险限额模型
        stop_distance = self.cfg.ATR_STOP_MULTIPLIER * atr_value
        risk_money = equity * self.cfg.RISK_PER_TRADE
        if stop_distance == 0:
            return 0
        risk_lots = int(risk_money / (stop_distance * self.cfg.MULTIPLIER))

        # 2. 保证金限额模型
        value_per_lot = price * self.cfg.MULTIPLIER
        margin_per_lot = value_per_lot * self.cfg.MARGIN_RATE
        if margin_per_lot == 0:
            return 0
        max_lots_margin = int(equity / margin_per_lot)

        return max(0, min(risk_lots, max_lots_margin))

    def run_strategy(self):
        equity = self.cfg.INITIAL_CAPITAL
        position = 0
        entry_price = 0
        highest_price = 0
        lowest_price = 0

        print(
            f"开始回测... 初始资金: {equity}, 策略: Dual Thrust (N={self.cfg.N_DAYS}, K1={self.cfg.K1}, K2={self.cfg.K2})"
        )

        for row in self.data.itertuples():
            date = row.Index
            open_p, high_p, low_p, close_p = row.Open, row.High, row.Low, row.Close
            atr = getattr(row, "ATR", 0)

            trade_executed = False

            # --- 1. 盘中风控 (ATR 移动止损) ---
            if position != 0:
                # 多单处理
                if position > 0:
                    highest_price = max(highest_price, high_p)
                    stop_line = highest_price - (self.cfg.ATR_STOP_MULTIPLIER * atr)

                    if low_p <= stop_line:
                        exit_price = open_p if open_p < stop_line else stop_line
                        pnl = (
                            (exit_price - entry_price)
                            * abs(position)
                            * self.cfg.MULTIPLIER
                        )
                        cost = (
                            exit_price
                            * abs(position)
                            * self.cfg.MULTIPLIER
                            * self.cfg.COMMISSION
                        )
                        net_pnl = pnl - cost
                        equity += net_pnl

                        action_type = "TakeProfit(L)" if pnl > 0 else "StopLoss(L)"
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": action_type,
                                "Price": exit_price,
                                "PnL": net_pnl,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

                # 空单处理
                elif position < 0:
                    lowest_price = min(lowest_price, low_p)
                    stop_line = lowest_price + (self.cfg.ATR_STOP_MULTIPLIER * atr)

                    if high_p >= stop_line:
                        exit_price = open_p if open_p > stop_line else stop_line
                        pnl = (
                            (entry_price - exit_price)
                            * abs(position)
                            * self.cfg.MULTIPLIER
                        )
                        cost = (
                            exit_price
                            * abs(position)
                            * self.cfg.MULTIPLIER
                            * self.cfg.COMMISSION
                        )
                        net_pnl = pnl - cost
                        equity += net_pnl

                        action_type = "TakeProfit(S)" if pnl > 0 else "StopLoss(S)"
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": action_type,
                                "Price": exit_price,
                                "PnL": net_pnl,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

            # --- 2. 信号执行 ---
            signal = getattr(row, "Signal", 0)

            if not trade_executed and signal != 0:
                if position == 0 or np.sign(position) != np.sign(signal):

                    # 平旧仓
                    if position != 0:
                        pnl = (close_p - entry_price) * position * self.cfg.MULTIPLIER
                        cost = (
                            close_p
                            * abs(position)
                            * self.cfg.MULTIPLIER
                            * self.cfg.COMMISSION
                        )
                        equity += pnl - cost
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "Close(Rev)",
                                "Price": close_p,
                                "PnL": pnl - cost,
                                "Equity": equity,
                            }
                        )
                        position = 0

                    # 开新仓
                    lots = self.calculate_safe_lots(equity, close_p, atr_value=atr)

                    if lots > 0:
                        position = signal * lots
                        entry_price = close_p
                        highest_price = close_p
                        lowest_price = close_p

                        cost = (
                            close_p * lots * self.cfg.MULTIPLIER * self.cfg.COMMISSION
                        ) + (lots * self.cfg.SLIPPAGE * self.cfg.MULTIPLIER)
                        equity -= cost

                        action = "Buy" if signal == 1 else "Short"
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": action,
                                "Price": close_p,
                                "Lots": lots,
                                "PnL": -cost,
                                "Equity": equity,
                            }
                        )

            # --- 3. 每日结算 ---
            floating_pnl = 0
            if position != 0:
                floating_pnl = (close_p - entry_price) * position * self.cfg.MULTIPLIER

            self.daily_stats.append(
                {
                    "Date": date,
                    "Equity": equity + floating_pnl,
                    "Close": close_p,
                    "Position": position,
                }
            )

        self.df_results = pd.DataFrame(self.daily_stats).set_index("Date")
        self.df_trades = pd.DataFrame(self.trade_log)

        self.df_results["Peak"] = self.df_results["Equity"].cummax()
        self.df_results["Drawdown"] = (
            self.df_results["Equity"] - self.df_results["Peak"]
        ) / self.df_results["Peak"]

        return self.df_results

    def plot_performance(self):
        if not hasattr(self, "df_results"):
            return
        df = self.df_results

        plt.figure(figsize=(16, 12))

        # 子图1: 权益曲线
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df.index, df["Equity"], color="#c0392b", linewidth=2)
        ax1.fill_between(
            df.index, df["Equity"], self.cfg.INITIAL_CAPITAL, alpha=0.1, color="red"
        )
        ax1.set_title(
            f'Equity Curve (Final: {df["Equity"].iloc[-1]:,.0f})', fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # 子图2: 价格与Dual Thrust轨道
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df.index, df["Close"], color="black", alpha=0.5, label="Price")

        # 画 Dual Thrust 上下轨
        if "Buy_Line" in self.data.columns:
            ax2.plot(
                self.data.index,
                self.data["Buy_Line"],
                color="green",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Buy Line (Upper)",
            )
        if "Sell_Line" in self.data.columns:
            ax2.plot(
                self.data.index,
                self.data["Sell_Line"],
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Sell Line (Lower)",
            )

        if not self.df_trades.empty:
            buys = self.df_trades[self.df_trades["Type"] == "Buy"]
            shorts = self.df_trades[self.df_trades["Type"] == "Short"]
            stops = self.df_trades[
                self.df_trades["Type"].str.contains("StopLoss", case=False)
            ]
            profits = self.df_trades[
                self.df_trades["Type"].str.contains("TakeProfit", case=False)
            ]
            closes = self.df_trades[
                self.df_trades["Type"].str.contains("Close", case=False)
            ]

            if not buys.empty:
                ax2.scatter(
                    buys["Date"],
                    buys["Price"],
                    marker="^",
                    color="red",
                    s=100,
                    label="Buy Entry",
                    zorder=5,
                )
            if not shorts.empty:
                ax2.scatter(
                    shorts["Date"],
                    shorts["Price"],
                    marker="v",
                    color="green",
                    s=100,
                    label="Short Entry",
                    zorder=5,
                )
            if not stops.empty:
                ax2.scatter(
                    stops["Date"],
                    stops["Price"],
                    marker="x",
                    color="black",
                    s=80,
                    linewidths=2,
                    label="Stop Loss",
                    zorder=5,
                )
            if not profits.empty:
                ax2.scatter(
                    profits["Date"],
                    profits["Price"],
                    marker="*",
                    color="purple",
                    s=150,
                    label="Take Profit",
                    zorder=5,
                )
            if not closes.empty:
                ax2.scatter(
                    closes["Date"],
                    closes["Price"],
                    marker="o",
                    color="blue",
                    s=60,
                    label="Signal Close",
                    zorder=5,
                )

        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # 子图3: 回撤
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.fill_between(df.index, df["Drawdown"], 0, color="blue", alpha=0.2)
        ax3.set_title("Drawdown")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==========================================
# 2. 策略逻辑：Dual Thrust
# ==========================================
def prepare_strategy(df, config):
    """
    Dual Thrust 策略计算
    """
    data = df.copy()

    N = config.N_DAYS
    K1 = config.K1
    K2 = config.K2

    # 1. 计算过去N天的极值 (不含今天，shift(1))
    data["HH"] = data["High"].rolling(N).max().shift(1)
    data["HC"] = data["Close"].rolling(N).max().shift(1)
    data["LC"] = data["Close"].rolling(N).min().shift(1)
    data["LL"] = data["Low"].rolling(N).min().shift(1)

    # 2. 计算 Range
    data["Range"] = np.maximum(data["HH"] - data["LC"], data["HC"] - data["LL"])

    # 3. 计算上下轨 (基于今日开盘价)
    data["Buy_Line"] = data["Open"] + K1 * data["Range"]
    data["Sell_Line"] = data["Open"] - K2 * data["Range"]

    # 4. 计算 ATR (用于风控)
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    data["ATR"] = (
        pd.concat([high_low, high_close, low_close], axis=1)
        .max(axis=1)
        .rolling(config.ATR_PERIOD)
        .mean()
    )

    # 5. 生成信号
    data["Signal"] = 0

    # 突破上轨做多
    long_cond = data["Close"] > data["Buy_Line"]
    # 跌破下轨做空
    short_cond = data["Close"] < data["Sell_Line"]

    data.loc[long_cond, "Signal"] = 1
    data.loc[short_cond, "Signal"] = -1

    data.dropna(inplace=True)
    return data


# ==========================================
# 3. 数据获取
# ==========================================
def get_data(symbol):
    print(f"正在获取 {symbol} 数据...")
    try:
        df = ak.futures_main_sina(symbol=symbol)
        df = df[["日期", "开盘价", "最高价", "最低价", "收盘价", "成交量"]].rename(
            columns={
                "日期": "Date",
                "开盘价": "Open",
                "最高价": "High",
                "最低价": "Low",
                "收盘价": "Close",
                "成交量": "Volume",
            }
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.astype(float)

        start_date = (
            datetime.datetime.now() - datetime.timedelta(days=365 * 3)
        ).strftime("%Y-%m-%d")
        df = df[df.index >= start_date]
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    raw_data = get_data(StrategyConfig.SYMBOL)

    if raw_data is not None:
        # 2. 准备策略
        strategy_data = prepare_strategy(raw_data, StrategyConfig)

        # 3. 初始化回测
        bt = ProFuturesBacktest(strategy_data, StrategyConfig)

        # 4. 运行回测
        results = bt.run_strategy()

        # 5. 输出结果
        final_equity = results["Equity"].iloc[-1]
        ret = (
            final_equity - StrategyConfig.INITIAL_CAPITAL
        ) / StrategyConfig.INITIAL_CAPITAL
        mdd = results["Drawdown"].min()

        print("\n" + "=" * 40)
        print(f"回测品种: {StrategyConfig.SYMBOL}")
        print(f"初始资金: {StrategyConfig.INITIAL_CAPITAL}")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {ret*100:.2f}%")
        print(f"最大回撤: {mdd*100:.2f}%")
        print(f"交易次数: {len(bt.trade_log)}")
        print("=" * 40 + "\n")

        # 6. 绘图
        bt.plot_performance()
