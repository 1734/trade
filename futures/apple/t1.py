import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime


# ==========================================
# 0. 全局配置
# ==========================================
class StrategyConfig:
    SYMBOL = "ap0"  # 苹果
    INITIAL_CAPITAL = 100000
    MULTIPLIER = 10
    MARGIN_RATE = 0.15
    COMMISSION = 0.0001

    # 季节性窗口
    LONG_MONTHS = [4, 8]  # 做多月
    SHORT_MONTHS = [10, 12]  # 做空月

    # 参数
    MA_WINDOW = 20  # 趋势过滤
    ATR_WINDOW = 14
    ATR_STOP_MULT = 2.0  # 止损幅度


# ==========================================
# 1. 核心回测引擎 (通用版)
# ==========================================
class ProFuturesBacktest:
    def __init__(self, data, config):
        self.data = data.copy()
        self.cfg = config

        self.trade_log = []
        self.daily_stats = []

    def run(self):
        print("开始回测...")

        cash = self.cfg.INITIAL_CAPITAL
        position = 0
        entry_price = 0
        stop_price = 0  # 动态止损价

        for row in self.data.itertuples():
            date = row.Index

            # --- 1. 盘中风控 (止损优先) ---
            # 即使策略信号让你持有，但如果价格打到止损线，必须强平
            stop_triggered = False

            if position != 0:
                # 多单止损
                if position > 0 and row.Low <= stop_price:
                    exit_price = min(row.Open, stop_price)  # 模拟成交
                    pnl = (exit_price - entry_price) * position * self.cfg.MULTIPLIER
                    cash += pnl
                    self.trade_log.append(
                        {
                            "Date": date,
                            "Type": "StopLoss(L)",
                            "Price": exit_price,
                            "PnL": pnl,
                        }
                    )
                    position = 0
                    stop_triggered = True

                # 空单止损
                elif position < 0 and row.High >= stop_price:
                    exit_price = max(row.Open, stop_price)
                    pnl = (
                        (entry_price - exit_price) * abs(position) * self.cfg.MULTIPLIER
                    )
                    cash += pnl
                    self.trade_log.append(
                        {
                            "Date": date,
                            "Type": "StopLoss(S)",
                            "Price": exit_price,
                            "PnL": pnl,
                        }
                    )
                    position = 0
                    stop_triggered = True

            # --- 2. 策略信号执行 ---
            # 获取预先计算好的信号 (1:多, -1:空, 0:空仓)
            target_signal = getattr(row, "Signal", 0)

            # 如果刚刚止损了，今天就不再开仓，冷静一天
            if not stop_triggered:

                # 情况A: 信号归零 (季节窗口结束/趋势坏了) -> 平仓
                if target_signal == 0 and position != 0:
                    pnl = (row.Close - entry_price) * position * self.cfg.MULTIPLIER
                    cash += pnl
                    self.trade_log.append(
                        {
                            "Date": date,
                            "Type": "TimeExit/Close",
                            "Price": row.Close,
                            "PnL": pnl,
                        }
                    )
                    position = 0

                # 情况B: 开仓/换仓
                # 只有当当前持仓 != 目标信号时才操作
                elif target_signal != 0 and target_signal != np.sign(position):
                    # 先平旧仓 (如果有)
                    if position != 0:
                        pnl = (row.Close - entry_price) * position * self.cfg.MULTIPLIER
                        cash += pnl
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "Close(Rev)",
                                "Price": row.Close,
                                "PnL": pnl,
                            }
                        )
                        position = 0

                    # 开新仓
                    # 资金管理：固定2手
                    vol = 2
                    margin = (
                        row.Close * vol * self.cfg.MULTIPLIER * self.cfg.MARGIN_RATE
                    )

                    if cash >= margin:
                        position = target_signal * vol
                        entry_price = row.Close

                        # 设定初始止损价
                        atr = getattr(row, "ATR", 0)
                        if target_signal == 1:
                            stop_price = entry_price - (self.cfg.ATR_STOP_MULT * atr)
                            action = "Buy (Season)"
                        else:
                            stop_price = entry_price + (self.cfg.ATR_STOP_MULT * atr)
                            action = "Short (Season)"

                        self.trade_log.append(
                            {"Date": date, "Type": action, "Price": row.Close}
                        )

            # --- 3. 结算 ---
            equity = cash
            if position != 0:
                equity += (row.Close - entry_price) * position * self.cfg.MULTIPLIER

            self.daily_stats.append(
                {"Date": date, "Equity": equity, "Close": row.Close}
            )

        return pd.DataFrame(self.daily_stats).set_index("Date")

    def plot(self, results):
        plt.figure(figsize=(12, 8))

        # 权益
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(results.index, results["Equity"], color="red")
        ax1.set_title(
            f'Seasonal Strategy Equity (Final: {results["Equity"].iloc[-1]:.0f})'
        )
        ax1.grid(True)

        # 价格与信号
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(self.data.index, self.data["Close"], color="black", alpha=0.6)

        # 画出季节性背景色 (红色=多头月，绿色=空头月)
        # 为了不让图太乱，我们只标记交易点
        trades = pd.DataFrame(self.trade_log)
        if not trades.empty:
            buys = trades[trades["Type"].str.contains("Buy")]
            shorts = trades[trades["Type"].str.contains("Short")]
            stops = trades[trades["Type"].str.contains("Stop")]
            exits = trades[trades["Type"].str.contains("TimeExit")]

            ax2.scatter(
                buys["Date"], buys["Price"], marker="^", color="red", s=80, label="Buy"
            )
            ax2.scatter(
                shorts["Date"],
                shorts["Price"],
                marker="v",
                color="green",
                s=80,
                label="Short",
            )
            ax2.scatter(
                stops["Date"],
                stops["Price"],
                marker="x",
                color="black",
                s=60,
                label="StopLoss",
            )
            ax2.scatter(
                exits["Date"],
                exits["Price"],
                marker="o",
                color="blue",
                s=40,
                label="MonthEnd Exit",
            )

        ax2.legend()
        ax2.grid(True)
        plt.show()


# ==========================================
# 2. 策略逻辑 (专门计算 Signal)
# ==========================================
def prepare_strategy(df, config):
    """
    在这里集中计算所有信号
    Signal = 1 (持有做多)
    Signal = -1 (持有做空)
    Signal = 0 (空仓)
    """
    data = df.copy()

    # 1. 计算指标
    data["MA"] = data["Close"].rolling(config.MA_WINDOW).mean()

    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    data["ATR"] = (
        pd.concat([high_low, high_close, low_close], axis=1)
        .max(axis=1)
        .rolling(config.ATR_WINDOW)
        .mean()
    )

    # 2. 提取时间特征
    data["Month"] = data.index.month

    # 3. 生成信号 (向量化逻辑)
    data["Signal"] = 0

    # 条件A: 做多窗口 (月份符合 AND 价格在均线上)
    long_cond = (data["Month"].isin(config.LONG_MONTHS)) & (data["Close"] > data["MA"])

    # 条件B: 做空窗口 (月份符合 AND 价格在均线下)
    short_cond = (data["Month"].isin(config.SHORT_MONTHS)) & (
        data["Close"] < data["MA"]
    )

    # 赋值
    data.loc[long_cond, "Signal"] = 1
    data.loc[short_cond, "Signal"] = -1

    # 4. 处理月末强制平仓 (可选)
    # 如果你想严格执行“月底走人”，其实上面的逻辑已经包含了。
    # 因为一旦月份变成 5月(不在LONG_MONTHS里)，Signal 自动变为 0，回测引擎就会平仓。
    # 所以这里不需要额外写代码，逻辑已经闭环。

    data.dropna(inplace=True)
    return data


# ==========================================
# 3. 数据获取
# ==========================================
def get_data(symbol):
    print(f"正在获取 {symbol} 数据...")
    try:
        df = ak.futures_main_sina(symbol=symbol)
        df = df[["日期", "开盘价", "最高价", "最低价", "收盘价"]].rename(
            columns={
                "日期": "Date",
                "开盘价": "Open",
                "最高价": "High",
                "最低价": "Low",
                "收盘价": "Close",
            }
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    raw_data = get_data(StrategyConfig.SYMBOL)

    if raw_data is not None:
        # 2. 计算信号 (Prepare)
        strategy_data = prepare_strategy(raw_data, StrategyConfig)

        # 3. 初始化回测
        bt = ProFuturesBacktest(strategy_data, StrategyConfig)

        # 4. 运行 (Run)
        results = bt.run()

        # 5. 统计
        final_equity = results["Equity"].iloc[-1]
        ret = (
            final_equity - StrategyConfig.INITIAL_CAPITAL
        ) / StrategyConfig.INITIAL_CAPITAL

        print("\n" + "=" * 40)
        print(f"苹果季节性策略 (标准架构版)")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {ret*100:.2f}%")
        print(f"交易次数: {len(bt.trade_log)}")
        print("=" * 40)

        # 6. 绘图
        bt.plot(results)

        # 打印最近交易
        print("\n最近 5 笔交易:")
        print(pd.DataFrame(bt.trade_log).tail(5))
