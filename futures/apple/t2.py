import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime


# ==========================================
# 0. 全局配置
# ==========================================
class StrategyConfig:
    # --- 标的与资金 ---
    SYMBOL = "ap0"  # 苹果主连
    INITIAL_CAPITAL = 100000  # 10万
    MULTIPLIER = 10  # 10吨/手
    MARGIN_RATE = 0.15  # 保证金
    COMMISSION = 0.0001

    # --- 回测时间范围 ---
    # 默认从 2010-01-01 (足够久远，涵盖苹果2017年上市) 到 今天
    START_DATE = "2010-01-01"
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    # --- 季节性窗口 ---
    LONG_MONTHS = [4, 8]  # 做多月
    SHORT_MONTHS = [10, 12]  # 做空月

    # --- 策略参数 ---
    MA_WINDOW = 20  # 趋势过滤 (20日均线)
    BREAKOUT_WINDOW = 5  # 突破扳机 (5日高低点)
    ATR_WINDOW = 14  # ATR周期
    ATR_STOP_MULT = 2.0  # 止损幅度


# ==========================================
# 1. 数据获取 (支持时间过滤)
# ==========================================
def get_data(symbol, config):
    print(f"正在获取 {symbol} 数据...")
    try:
        # 1. 下载全量历史数据
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

        # 2. 根据配置筛选时间范围
        start_dt = pd.to_datetime(config.START_DATE)
        end_dt = pd.to_datetime(config.END_DATE)

        # 筛选
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        df = df.loc[mask].copy()

        if df.empty:
            print(f"警告：在 {config.START_DATE} 至 {config.END_DATE} 期间无数据！")
            return None

        print(
            f"数据获取成功: {len(df)} 条 ({df.index[0].date()} ~ {df.index[-1].date()})"
        )
        return df

    except Exception as e:
        print(f"数据获取失败: {e}")
        return None


# ==========================================
# 2. 策略逻辑 (计算信号)
# ==========================================
def prepare_strategy(df, config):
    data = df.copy()

    # 1. 基础指标
    data["MA"] = data["Close"].rolling(config.MA_WINDOW).mean()

    # 2. 突破指标 (唐奇安通道)
    # 过去 N 天的最高/最低价 (不含今天)
    data["Ref_High"] = data["High"].rolling(config.BREAKOUT_WINDOW).max().shift(1)
    data["Ref_Low"] = data["Low"].rolling(config.BREAKOUT_WINDOW).min().shift(1)

    # 3. ATR
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    data["ATR"] = (
        pd.concat([high_low, high_close, low_close], axis=1)
        .max(axis=1)
        .rolling(config.ATR_WINDOW)
        .mean()
    )

    # 4. 月份
    data["Month"] = data.index.month

    # 5. 基础意愿信号 (季节 + 均线)
    data["Signal"] = 0

    # 做多意愿: 季节符合 & 价格在均线上
    long_cond = (data["Month"].isin(config.LONG_MONTHS)) & (data["Close"] > data["MA"])
    # 做空意愿: 季节符合 & 价格在均线下
    short_cond = (data["Month"].isin(config.SHORT_MONTHS)) & (
        data["Close"] < data["MA"]
    )

    data.loc[long_cond, "Signal"] = 1
    data.loc[short_cond, "Signal"] = -1

    data.dropna(inplace=True)
    return data


# ==========================================
# 3. 回测引擎 (带突破扳机)
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
        stop_price = 0

        for row in self.data.itertuples():
            date = row.Index

            # --- 1. 盘中风控 ---
            stop_triggered = False
            if position != 0:
                # 多单止损
                if position > 0 and row.Low <= stop_price:
                    exit_price = min(row.Open, stop_price)
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

            # --- 2. 信号执行 ---
            # 基础信号 (1:多头意愿, -1:空头意愿, 0:休息)
            base_signal = getattr(row, "Signal", 0)

            # 突破位
            ref_high = getattr(row, "Ref_High", 999999)
            ref_low = getattr(row, "Ref_Low", 0)

            if not stop_triggered:
                # A. 平仓逻辑: 意愿消失 (季节过了/均线破了)
                if base_signal == 0 and position != 0:
                    pnl = (row.Close - entry_price) * position * self.cfg.MULTIPLIER
                    cash += pnl
                    self.trade_log.append(
                        {
                            "Date": date,
                            "Type": "TimeExit/TrendEnd",
                            "Price": row.Close,
                            "PnL": pnl,
                        }
                    )
                    position = 0

                # B. 开仓逻辑: (空仓) AND (有意愿) AND (触发突破扳机)
                elif position == 0 and base_signal != 0:
                    can_open = False

                    # 做多: 意愿1 且 突破前高
                    if base_signal == 1 and row.Close > ref_high:
                        can_open = True
                    # 做空: 意愿-1 且 跌破前低
                    elif base_signal == -1 and row.Close < ref_low:
                        can_open = True

                    if can_open:
                        vol = 2  # 固定2手
                        margin = (
                            row.Close * vol * self.cfg.MULTIPLIER * self.cfg.MARGIN_RATE
                        )

                        if cash >= margin:
                            position = base_signal * vol
                            entry_price = row.Close

                            # 设置止损
                            atr = getattr(row, "ATR", 0)
                            if base_signal == 1:
                                stop_price = entry_price - (
                                    self.cfg.ATR_STOP_MULT * atr
                                )
                                action = "Buy (Breakout)"
                            else:
                                stop_price = entry_price + (
                                    self.cfg.ATR_STOP_MULT * atr
                                )
                                action = "Short (Breakout)"

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
        if results.empty:
            return
        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(results.index, results["Equity"], color="red")
        ax1.set_title(f"Equity Curve ({self.cfg.START_DATE} ~ {self.cfg.END_DATE})")
        ax1.grid(True)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(self.data.index, self.data["Close"], color="black", alpha=0.6)

        trades = pd.DataFrame(self.trade_log)
        if not trades.empty:
            buys = trades[trades["Type"].str.contains("Buy")]
            shorts = trades[trades["Type"].str.contains("Short")]
            stops = trades[trades["Type"].str.contains("Stop")]

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

        ax2.legend()
        ax2.grid(True)
        plt.show()


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据 (传入配置)
    raw_data = get_data(StrategyConfig.SYMBOL, StrategyConfig)

    if raw_data is not None:
        # 2. 计算信号
        strategy_data = prepare_strategy(raw_data, StrategyConfig)

        # 3. 回测
        bt = ProFuturesBacktest(strategy_data, StrategyConfig)
        results = bt.run()

        # 4. 统计
        final_equity = results["Equity"].iloc[-1]
        ret = (
            final_equity - StrategyConfig.INITIAL_CAPITAL
        ) / StrategyConfig.INITIAL_CAPITAL

        print("\n" + "=" * 40)
        print(f"回测品种: {StrategyConfig.SYMBOL}")
        print(f"时间范围: {StrategyConfig.START_DATE} 至 {StrategyConfig.END_DATE}")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {ret*100:.2f}%")
        print(f"交易次数: {len(bt.trade_log)}")
        print("=" * 40)

        # 5. 绘图
        bt.plot(results)

        # 打印最近交易
        if len(bt.trade_log) > 0:
            print("\n最近 5 笔交易:")
            print(pd.DataFrame(bt.trade_log).tail(5))
