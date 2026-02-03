import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime


# ==========================================
# 1. 核心回测引擎 (带风控与资金管理)
# ==========================================
class RobustFuturesBacktest:
    def __init__(
        self,
        data,
        initial_capital=500000,
        contract_multiplier=10,
        commission_rate=0.0001,
        slippage=1,
        margin_rate=0.12,  # 期货公司保证金率 (12%)
        limit_move_ratio=0.08,
    ):  # 涨跌停幅度 (8%)

        self.data = data.copy()
        self.initial_capital = initial_capital
        self.multiplier = contract_multiplier
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.margin_rate = margin_rate
        self.limit_move_ratio = limit_move_ratio

        # 结果容器
        self.trade_log = []
        self.daily_stats = []

    def calculate_safe_lots(self, equity, price, risk_ratio=0.5):
        if price <= 0:
            return 0
        value_per_lot = price * self.multiplier
        max_loss_per_lot = value_per_lot * self.limit_move_ratio
        margin_per_lot = value_per_lot * self.margin_rate
        max_allowed_loss = equity * risk_ratio
        safe_lots = int(max_allowed_loss / max_loss_per_lot)
        max_lots_margin = int(equity / margin_per_lot)
        return max(0, min(safe_lots, max_lots_margin))

    def run_strategy(self, sl_pct=0.02, tp_pct=0.05, risk_ratio=0.5):
        equity = self.initial_capital
        position = 0
        entry_price = 0

        print("开始回测...")

        for row in self.data.itertuples():
            date = row.Index
            open_p, high_p, low_p, close_p = row.Open, row.High, row.Low, row.Close

            trade_executed = False

            # --- 1. 盘中风控 ---
            if position != 0:
                if position > 0:  # 多单
                    sl_price = entry_price * (1 - sl_pct)
                    tp_price = entry_price * (1 + tp_pct)

                    if low_p <= sl_price:
                        exit_price = open_p if open_p < sl_price else sl_price
                        pnl = (
                            (exit_price - entry_price) * abs(position) * self.multiplier
                        )
                        cost = (
                            exit_price
                            * abs(position)
                            * self.multiplier
                            * self.commission_rate
                        )
                        equity += pnl - cost
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "StopLoss(L)",
                                "Price": exit_price,
                                "PnL": pnl - cost,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

                    elif high_p >= tp_price:
                        exit_price = open_p if open_p > tp_price else tp_price
                        pnl = (
                            (exit_price - entry_price) * abs(position) * self.multiplier
                        )
                        cost = (
                            exit_price
                            * abs(position)
                            * self.multiplier
                            * self.commission_rate
                        )
                        equity += pnl - cost
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "TakeProfit(L)",
                                "Price": exit_price,
                                "PnL": pnl - cost,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

                elif position < 0:  # 空单
                    sl_price = entry_price * (1 + sl_pct)
                    tp_price = entry_price * (1 - tp_pct)

                    if high_p >= sl_price:
                        exit_price = open_p if open_p > sl_price else sl_price
                        pnl = (
                            (entry_price - exit_price) * abs(position) * self.multiplier
                        )
                        cost = (
                            exit_price
                            * abs(position)
                            * self.multiplier
                            * self.commission_rate
                        )
                        equity += pnl - cost
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "StopLoss(S)",
                                "Price": exit_price,
                                "PnL": pnl - cost,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

                    elif low_p <= tp_price:
                        exit_price = open_p if open_p < tp_price else tp_price
                        pnl = (
                            (entry_price - exit_price) * abs(position) * self.multiplier
                        )
                        cost = (
                            exit_price
                            * abs(position)
                            * self.multiplier
                            * self.commission_rate
                        )
                        equity += pnl - cost
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "TakeProfit(S)",
                                "Price": exit_price,
                                "PnL": pnl - cost,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

            # --- 2. 信号执行 ---
            signal = getattr(row, "Signal", 0)

            if not trade_executed and signal != 0:
                if position == 0 or np.sign(position) != np.sign(signal):
                    if position != 0:
                        pnl = (close_p - entry_price) * position * self.multiplier
                        cost = (
                            close_p
                            * abs(position)
                            * self.multiplier
                            * self.commission_rate
                        )
                        equity += pnl - cost
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": "Close",
                                "Price": close_p,
                                "PnL": pnl - cost,
                                "Equity": equity,
                            }
                        )
                        position = 0

                    lots = self.calculate_safe_lots(
                        equity, close_p, risk_ratio=risk_ratio
                    )

                    if lots > 0:
                        position = signal * lots
                        entry_price = close_p
                        cost = (
                            close_p * lots * self.multiplier * self.commission_rate
                        ) + (lots * self.slippage * self.multiplier)
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
                floating_pnl = (close_p - entry_price) * position * self.multiplier

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

        # =========================================
        # 【修复点】在此处计算回撤，确保返回结果包含 Drawdown
        # =========================================
        self.df_results["Peak"] = self.df_results["Equity"].cummax()
        self.df_results["Drawdown"] = (
            self.df_results["Equity"] - self.df_results["Peak"]
        ) / self.df_results["Peak"]

        return self.df_results

    def plot_performance(self):
        if not hasattr(self, "df_results"):
            return
        df = self.df_results

        plt.figure(figsize=(16, 10))

        # 子图1: 权益曲线
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(
            df.index, df["Equity"], color="#c0392b", linewidth=2, label="Total Equity"
        )
        ax1.fill_between(
            df.index, df["Equity"], self.initial_capital, alpha=0.1, color="red"
        )
        ax1.set_title(
            f'Equity Curve (Final: {df["Equity"].iloc[-1]:,.0f})',
            fontsize=12,
            fontweight="bold",
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 子图2: 价格与交易点
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df.index, df["Close"], color="black", alpha=0.6, label="Price")

        if not self.df_trades.empty:
            buys = self.df_trades[self.df_trades["Type"] == "Buy"]
            shorts = self.df_trades[self.df_trades["Type"] == "Short"]
            stops = self.df_trades[self.df_trades["Type"].str.contains("StopLoss")]
            takes = self.df_trades[self.df_trades["Type"].str.contains("TakeProfit")]

            ax2.scatter(
                buys["Date"],
                buys["Price"],
                marker="^",
                color="red",
                s=80,
                label="Buy",
                zorder=5,
            )
            ax2.scatter(
                shorts["Date"],
                shorts["Price"],
                marker="v",
                color="green",
                s=80,
                label="Short",
                zorder=5,
            )
            ax2.scatter(
                stops["Date"],
                stops["Price"],
                marker="x",
                color="black",
                s=60,
                label="StopLoss",
                zorder=5,
            )
            ax2.scatter(
                takes["Date"],
                takes["Price"],
                marker="o",
                color="gold",
                s=60,
                label="TakeProfit",
                zorder=5,
            )

        ax2.set_title("Trades Visualization")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # 子图3: 回撤
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(df.index, df["Drawdown"], color="blue", linewidth=1)
        ax3.fill_between(df.index, df["Drawdown"], 0, color="blue", alpha=0.2)
        ax3.set_title(f'Max Drawdown: {df["Drawdown"].min()*100:.2f}%')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==========================================
# 2. 数据获取与策略定义
# ==========================================
def get_data(symbol="rb0"):
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
        start_date = (
            datetime.datetime.now() - datetime.timedelta(days=365 * 3)
        ).strftime("%Y-%m-%d")
        df = df[df.index >= start_date]
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None


def prepare_strategy(df):
    data = df.copy()
    # 示例策略：唐奇安通道突破
    window = 20
    data["Donchian_High"] = data["High"].rolling(window).max().shift(1)
    data["Donchian_Low"] = data["Low"].rolling(window).min().shift(1)
    data["Signal"] = 0

    long_cond = data["Close"] > data["Donchian_High"]
    short_cond = data["Close"] < data["Donchian_Low"]

    data.loc[long_cond, "Signal"] = 1
    data.loc[short_cond, "Signal"] = -1
    data.dropna(inplace=True)
    return data


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    SYMBOL = "rb0"
    MULTIPLIER = 10
    MARGIN_RATE = 0.12
    SLIPPAGE = 1
    CAPITAL = 500000

    raw_data = get_data(SYMBOL)

    if raw_data is not None:
        strategy_data = prepare_strategy(raw_data)

        bt = RobustFuturesBacktest(
            strategy_data,
            initial_capital=CAPITAL,
            contract_multiplier=MULTIPLIER,
            margin_rate=MARGIN_RATE,
            slippage=SLIPPAGE,
        )

        results = bt.run_strategy(sl_pct=0.02, tp_pct=0.06, risk_ratio=0.5)

        final_equity = results["Equity"].iloc[-1]
        total_return = (final_equity - CAPITAL) / CAPITAL
        print("\n" + "=" * 30)
        print(f"回测品种: {SYMBOL}")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {total_return*100:.2f}%")
        # 现在这里不会报错了
        print(f"最大回撤: {results['Drawdown'].min()*100:.2f}%")
        print(f"交易次数: {len(bt.df_trades)}")
        print("=" * 30 + "\n")

        bt.plot_performance()
