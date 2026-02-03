import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime
import os
import time

# ==========================================
# 0. 全局配置与品种数据库
# ==========================================

# 基础配置
INITIAL_CAPITAL = 80000
RESULT_DIR = "backtest_results"  # 图片保存文件夹

# --- 中国期货主流品种数据库 (代码: {乘数, 保证金率}) ---
# 注意：保证金率按期货公司标准估算 (通常比交易所高2-3个点)
FUTURES_DB = {
    # --- 黑色系 ---
    "rb0": {"name": "螺纹钢", "mult": 10, "margin": 0.13},
    "hc0": {"name": "热卷", "mult": 10, "margin": 0.13},
    "i0": {"name": "铁矿石", "mult": 100, "margin": 0.15},
    "j0": {"name": "焦炭", "mult": 100, "margin": 0.15},
    "jm0": {"name": "焦煤", "mult": 60, "margin": 0.15},
    "zc0": {"name": "动力煤", "mult": 100, "margin": 0.30},  # 波动极大，保证金高
    "ss0": {"name": "不锈钢", "mult": 5, "margin": 0.10},
    # --- 有色金属 ---
    "cu0": {"name": "沪铜", "mult": 5, "margin": 0.10},
    "al0": {"name": "沪铝", "mult": 5, "margin": 0.10},
    "zn0": {"name": "沪锌", "mult": 5, "margin": 0.10},
    "au0": {"name": "黄金", "mult": 1000, "margin": 0.10},
    "ag0": {"name": "白银", "mult": 15, "margin": 0.12},
    # --- 化工能源 ---
    "sc0": {"name": "原油", "mult": 1000, "margin": 0.15},
    "fu0": {"name": "燃油", "mult": 10, "margin": 0.15},
    "pg0": {"name": "LPG", "mult": 20, "margin": 0.12},
    "ta0": {"name": "PTA", "mult": 5, "margin": 0.10},
    "ma0": {"name": "甲醇", "mult": 10, "margin": 0.12},
    "pp0": {"name": "聚丙烯", "mult": 5, "margin": 0.10},
    "l0": {"name": "塑料", "mult": 5, "margin": 0.10},
    "v0": {"name": "PVC", "mult": 5, "margin": 0.10},
    "fg0": {"name": "玻璃", "mult": 20, "margin": 0.15},  # 波动大
    "sa0": {"name": "纯碱", "mult": 20, "margin": 0.15},  # 波动大
    # --- 农产品 ---
    "m0": {"name": "豆粕", "mult": 10, "margin": 0.10},
    "y0": {"name": "豆油", "mult": 10, "margin": 0.10},
    "p0": {"name": "棕榈油", "mult": 10, "margin": 0.12},
    "sr0": {"name": "白糖", "mult": 10, "margin": 0.10},
    "cf0": {"name": "棉花", "mult": 5, "margin": 0.10},
    "c0": {"name": "玉米", "mult": 10, "margin": 0.09},
    "jd0": {"name": "鸡蛋", "mult": 500, "margin": 0.10},
    "lh0": {"name": "生猪", "mult": 16, "margin": 0.15},
    "ap0": {"name": "苹果", "mult": 10, "margin": 0.15},
    # --- 金融期货 ---
    "IF0": {"name": "沪深300", "mult": 300, "margin": 0.12},
    "IC0": {"name": "中证500", "mult": 200, "margin": 0.12},
    "IH0": {"name": "上证50", "mult": 300, "margin": 0.12},
    "IM0": {"name": "中证1000", "mult": 200, "margin": 0.12},
    "T0": {"name": "十年国债", "mult": 10000, "margin": 0.03},  # 国债保证金极低
}


class StrategyConfig:
    # 动态参数，将在循环中被修改
    SYMBOL = ""
    NAME = ""
    INITIAL_CAPITAL = INITIAL_CAPITAL
    MULTIPLIER = 10
    MARGIN_RATE = 0.12

    # 固定参数
    COMMISSION = 0.0001
    SLIPPAGE = 1

    # Dual Thrust 参数
    N_DAYS = 4
    K1 = 0.6  # 做多难一点
    K2 = 0.4  # 做空容易一点

    # 风控参数
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.5
    RISK_PER_TRADE = 0.02


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
        if price <= 0 or atr_value <= 0:
            return 0

        # 1. 风险限额
        stop_distance = self.cfg.ATR_STOP_MULTIPLIER * atr_value
        risk_money = equity * self.cfg.RISK_PER_TRADE
        if stop_distance == 0:
            return 0
        risk_lots = int(risk_money / (stop_distance * self.cfg.MULTIPLIER))

        # 2. 保证金限额
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

        for row in self.data.itertuples():
            date = row.Index
            open_p, high_p, low_p, close_p = row.Open, row.High, row.Low, row.Close
            atr = getattr(row, "ATR", 0)

            trade_executed = False

            # --- 风控 ---
            if position != 0:
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
                        action = "TakeProfit(L)" if pnl > 0 else "StopLoss(L)"
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": action,
                                "Price": exit_price,
                                "PnL": net_pnl,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True
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
                        action = "TakeProfit(S)" if pnl > 0 else "StopLoss(S)"
                        self.trade_log.append(
                            {
                                "Date": date,
                                "Type": action,
                                "Price": exit_price,
                                "PnL": net_pnl,
                                "Equity": equity,
                            }
                        )
                        position = 0
                        trade_executed = True

            # --- 信号 ---
            signal = getattr(row, "Signal", 0)
            if not trade_executed and signal != 0:
                if position == 0 or np.sign(position) != np.sign(signal):
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

            # --- 结算 ---
            floating_pnl = 0
            if position != 0:
                floating_pnl = (close_p - entry_price) * position * self.cfg.MULTIPLIER

            self.daily_stats.append(
                {"Date": date, "Equity": equity + floating_pnl, "Close": close_p}
            )

        self.df_results = pd.DataFrame(self.daily_stats).set_index("Date")
        self.df_trades = pd.DataFrame(self.trade_log)
        self.df_results["Peak"] = self.df_results["Equity"].cummax()
        self.df_results["Drawdown"] = (
            self.df_results["Equity"] - self.df_results["Peak"]
        ) / self.df_results["Peak"]
        return self.df_results

    def save_plot(self, filename):
        """保存图片到文件，不显示"""
        if not hasattr(self, "df_results"):
            return
        df = self.df_results

        # 关闭交互模式，防止弹窗
        plt.ioff()
        fig = plt.figure(figsize=(16, 12))

        # 1. 权益
        ax1 = plt.subplot(3, 1, 1)
        final_eq = df["Equity"].iloc[-1]
        ret = (final_eq - self.cfg.INITIAL_CAPITAL) / self.cfg.INITIAL_CAPITAL
        color = "#c0392b" if ret > 0 else "green"

        ax1.plot(df.index, df["Equity"], color=color, linewidth=2)
        ax1.set_title(
            f"{self.cfg.NAME} ({self.cfg.SYMBOL}) - Return: {ret*100:.2f}% - Final: {final_eq:,.0f}",
            fontweight="bold",
            fontsize=14,
        )
        ax1.grid(True, alpha=0.3)

        # 2. 价格与轨道
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df.index, df["Close"], color="black", alpha=0.5, label="Price")
        if "Buy_Line" in self.data.columns:
            ax2.plot(
                self.data.index,
                self.data["Buy_Line"],
                color="green",
                linestyle="--",
                alpha=0.5,
            )
            ax2.plot(
                self.data.index,
                self.data["Sell_Line"],
                color="red",
                linestyle="--",
                alpha=0.5,
            )

        # 标记交易
        if not self.df_trades.empty:
            buys = self.df_trades[self.df_trades["Type"] == "Buy"]
            shorts = self.df_trades[self.df_trades["Type"] == "Short"]
            stops = self.df_trades[
                self.df_trades["Type"].str.contains("StopLoss", case=False)
            ]
            profits = self.df_trades[
                self.df_trades["Type"].str.contains("TakeProfit", case=False)
            ]

            if not buys.empty:
                ax2.scatter(
                    buys["Date"], buys["Price"], marker="^", color="red", s=80, zorder=5
                )
            if not shorts.empty:
                ax2.scatter(
                    shorts["Date"],
                    shorts["Price"],
                    marker="v",
                    color="green",
                    s=80,
                    zorder=5,
                )
            if not stops.empty:
                ax2.scatter(
                    stops["Date"],
                    stops["Price"],
                    marker="x",
                    color="black",
                    s=60,
                    zorder=5,
                )
            if not profits.empty:
                ax2.scatter(
                    profits["Date"],
                    profits["Price"],
                    marker="*",
                    color="purple",
                    s=120,
                    zorder=5,
                )

        ax2.legend(["Price", "Buy Line", "Sell Line"], loc="upper left")
        ax2.grid(True, alpha=0.3)

        # 3. 回撤
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.fill_between(df.index, df["Drawdown"], 0, color="blue", alpha=0.2)
        ax3.set_title(f'Max Drawdown: {df["Drawdown"].min()*100:.2f}%')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        # 保存文件
        plt.savefig(filename)
        plt.close(fig)  # 必须关闭，否则内存溢出


# ==========================================
# 2. 策略逻辑
# ==========================================
def prepare_strategy(df, config):
    data = df.copy()
    N, K1, K2 = config.N_DAYS, config.K1, config.K2

    data["HH"] = data["High"].rolling(N).max().shift(1)
    data["HC"] = data["Close"].rolling(N).max().shift(1)
    data["LC"] = data["Close"].rolling(N).min().shift(1)
    data["LL"] = data["Low"].rolling(N).min().shift(1)
    data["Range"] = np.maximum(data["HH"] - data["LC"], data["HC"] - data["LL"])

    data["Buy_Line"] = data["Open"] + K1 * data["Range"]
    data["Sell_Line"] = data["Open"] - K2 * data["Range"]

    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    data["ATR"] = (
        pd.concat([high_low, high_close, low_close], axis=1)
        .max(axis=1)
        .rolling(config.ATR_PERIOD)
        .mean()
    )

    data["Signal"] = 0
    data.loc[data["Close"] > data["Buy_Line"], "Signal"] = 1
    data.loc[data["Close"] < data["Sell_Line"], "Signal"] = -1
    data.dropna(inplace=True)
    return data


def get_data(symbol):
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
    except:
        return None


# ==========================================
# 3. 批量运行主程序
# ==========================================
if __name__ == "__main__":
    # 1. 创建结果文件夹
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"创建文件夹: {RESULT_DIR}")

    summary_list = []

    print(f"开始批量回测，共 {len(FUTURES_DB)} 个品种...")
    print("-" * 60)
    print(f"{'代码':<6} {'名称':<6} {'总收益率':<10} {'最大回撤':<10} {'最终权益':<12}")
    print("-" * 60)

    # 2. 循环遍历所有品种
    for symbol, info in FUTURES_DB.items():
        # 更新配置
        StrategyConfig.SYMBOL = symbol
        StrategyConfig.NAME = info["name"]
        StrategyConfig.MULTIPLIER = info["mult"]
        StrategyConfig.MARGIN_RATE = info["margin"]

        # 获取数据
        raw_data = get_data(symbol)

        if raw_data is not None and not raw_data.empty:
            try:
                # 运行策略
                strategy_data = prepare_strategy(raw_data, StrategyConfig)
                bt = ProFuturesBacktest(strategy_data, StrategyConfig)
                results = bt.run_strategy()

                # 统计结果
                final_equity = results["Equity"].iloc[-1]
                ret = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
                mdd = results["Drawdown"].min()

                # 打印单行结果
                print(
                    f"{symbol:<6} {info['name']:<6} {ret*100:>9.2f}% {mdd*100:>9.2f}% {final_equity:>12,.0f}"
                )

                # 保存图片
                file_path = os.path.join(RESULT_DIR, f"{symbol}_{info['name']}.png")
                bt.save_plot(file_path)

                # 收集汇总
                summary_list.append(
                    {
                        "Symbol": symbol,
                        "Name": info["name"],
                        "Return": ret,
                        "MDD": mdd,
                        "Equity": final_equity,
                    }
                )

            except Exception as e:
                print(f"{symbol} 回测出错: {e}")
        else:
            print(f"{symbol} 获取数据失败或为空")

        # 防止请求过快被封，稍微停顿
        time.sleep(0.5)

    # 3. 最终汇总排序
    print("\n" + "=" * 60)
    print("回测结束！收益率排名 (Top 10):")
    print("=" * 60)

    # 按收益率从高到低排序
    summary_list.sort(key=lambda x: x["Return"], reverse=True)

    for item in summary_list[:10]:
        print(f"{item['Name']}({item['Symbol']}): {item['Return']*100:.2f}%")

    print(f"\n所有结果图片已保存在 '{RESULT_DIR}' 文件夹中。")
