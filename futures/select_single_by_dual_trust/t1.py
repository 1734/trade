import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import datetime
import time
import os


# ==========================================
# 0. 全局配置
# ==========================================
class StrategyConfig:
    # --- 账户设置 ---
    INITIAL_CAPITAL = 200000  # 20万 (只做一手，资金充裕)

    # --- Dual Thrust 参数 ---
    N_DAYS = 4  # N日区间
    K1 = 0.5  # 上轨系数
    K2 = 0.5  # 下轨系数

    # --- 风控参数 ---
    ATR_PERIOD = 14
    RISK_PER_TRADE = 0.05  # 这里的风控主要靠轮动，单笔仓位给宽一点


# 品种数据库
FUTURES_DB = {
    "rb0": {"name": "螺纹", "mult": 10, "margin": 0.13},
    "i0": {"name": "铁矿", "mult": 100, "margin": 0.15},
    "cu0": {"name": "沪铜", "mult": 5, "margin": 0.10},
    "ta0": {"name": "PTA", "mult": 5, "margin": 0.10},
    "ma0": {"name": "甲醇", "mult": 10, "margin": 0.12},
    "m0": {"name": "豆粕", "mult": 10, "margin": 0.10},
    "p0": {"name": "棕榈", "mult": 10, "margin": 0.12},
    "sr0": {"name": "白糖", "mult": 10, "margin": 0.10},
    "fg0": {"name": "玻璃", "mult": 20, "margin": 0.15},
    "sa0": {"name": "纯碱", "mult": 20, "margin": 0.15},
}


# ==========================================
# 1. 数据管理器
# ==========================================
class DataManager:
    def __init__(self, symbol_dict):
        self.symbol_dict = symbol_dict
        self.data_dict = {}

    def download_data(self):
        print("正在下载数据...")
        for symbol, info in self.symbol_dict.items():
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

                # 截取最近3年
                start_date = (
                    datetime.datetime.now() - datetime.timedelta(days=365 * 3)
                ).strftime("%Y-%m-%d")
                df = df[df.index >= start_date]

                # 预计算策略指标 (Dual Thrust + ATR)
                self._calc_indicators(df)

                self.data_dict[symbol] = df
                print(f"[{info['name']}] 准备就绪")
                time.sleep(0.3)
            except Exception as e:
                print(f"[{symbol}] 失败: {e}")

    def _calc_indicators(self, df):
        """预计算 Dual Thrust 轨道和 ATR"""
        N = StrategyConfig.N_DAYS
        K1 = StrategyConfig.K1
        K2 = StrategyConfig.K2

        # Dual Thrust Range
        hh = df["High"].rolling(N).max().shift(1)
        hc = df["Close"].rolling(N).max().shift(1)
        lc = df["Close"].rolling(N).min().shift(1)
        ll = df["Low"].rolling(N).min().shift(1)
        df["Range"] = np.maximum(hh - lc, hc - ll)

        # 上下轨
        df["Buy_Line"] = df["Open"] + K1 * df["Range"]
        df["Sell_Line"] = df["Open"] - K2 * df["Range"]

        # ATR (用于归一化分值)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        df["ATR"] = (
            pd.concat([high_low, high_close, low_close], axis=1)
            .max(axis=1)
            .rolling(14)
            .mean()
        )


# ==========================================
# 2. 单标的轮动回测引擎
# ==========================================
class SingleRotationBacktest:
    def __init__(self, data_manager, config):
        self.dm = data_manager
        self.cfg = config

        self.cash = config.INITIAL_CAPITAL
        self.current_symbol = None  # 当前持有的品种 (None表示空仓)
        self.current_pos = {}  # {'vol': 0, 'entry': 0, 'dir': 0}

        self.equity_curve = []
        self.trade_log = []
        self.holding_history = []  # 记录每天持有什么，用于画图

        # 生成统一的时间轴
        all_dates = sorted(
            list(set().union(*[df.index for df in self.dm.data_dict.values()]))
        )
        self.dates = pd.to_datetime(all_dates)

    def run(self):
        print("\n开始回测 (同一时间仅持有一只)...")

        for date in self.dates:
            # 1. --- 每日结算 (Mark to Market) ---
            equity = self.cash
            if self.current_symbol:
                symbol = self.current_symbol
                if date in self.dm.data_dict[symbol].index:
                    bar = self.dm.data_dict[symbol].loc[date]
                    pos = self.current_pos
                    # 浮动盈亏
                    pnl = (
                        (bar["Close"] - pos["entry"])
                        * pos["vol"]
                        * FUTURES_DB[symbol]["mult"]
                        * pos["dir"]
                    )
                    equity += pnl

            self.equity_curve.append({"Date": date, "Equity": equity})
            self.holding_history.append(
                {
                    "Date": date,
                    "Symbol": self.current_symbol if self.current_symbol else "Cash",
                }
            )

            # 2. --- 选品逻辑 (计算分值) ---
            best_symbol = None
            best_score = 0
            best_direction = 0  # 1做多, -1做空

            for symbol, df in self.dm.data_dict.items():
                if date not in df.index:
                    continue
                row = df.loc[date]

                if pd.isna(row["ATR"]) or row["ATR"] == 0:
                    continue

                # === 核心：归一化分值计算 ===
                # 分值 = (收盘价 - 突破线) / ATR
                # 含义：突破了多少个ATR的距离？

                score = 0
                direction = 0

                # 检查做多突破
                if row["Close"] > row["Buy_Line"]:
                    score = (row["Close"] - row["Buy_Line"]) / row["ATR"]
                    direction = 1

                # 检查做空突破
                elif row["Close"] < row["Sell_Line"]:
                    score = (row["Sell_Line"] - row["Close"]) / row[
                        "ATR"
                    ]  # 取正值方便比较幅度
                    direction = -1

                # 记录最强的那个
                if score > best_score:
                    best_score = score
                    best_symbol = symbol
                    best_direction = direction

            # 3. --- 交易执行 (轮动) ---
            # 只有当分值 > 0 (即发生了突破) 时才考虑持有

            # 情况A: 需要换仓 (当前持有的不是最好的，或者当前没持仓但有好的)
            if best_symbol is not None and best_symbol != self.current_symbol:
                # 1. 平掉旧的
                if self.current_symbol:
                    self._close_position(date, "Switch")

                # 2. 开新的
                self._open_position(best_symbol, date, best_direction, equity)

            # 情况B: 最好的就是当前持有的 -> 检查方向是否反转
            elif best_symbol is not None and best_symbol == self.current_symbol:
                if best_direction != self.current_pos["dir"]:
                    # 同品种反手
                    self._close_position(date, "Reverse")
                    self._open_position(best_symbol, date, best_direction, equity)
                else:
                    # 继续持有，不动
                    pass

            # 情况C: 全市场都没有突破 (best_symbol is None) -> 空仓休息
            elif best_symbol is None and self.current_symbol:
                self._close_position(date, "No_Signal")

        return pd.DataFrame(self.equity_curve).set_index("Date")

    def _close_position(self, date, reason):
        symbol = self.current_symbol
        if date not in self.dm.data_dict[symbol].index:
            return  # 停牌无法交易

        bar = self.dm.data_dict[symbol].loc[date]
        pos = self.current_pos

        # 结算盈亏
        pnl = (
            (bar["Close"] - pos["entry"])
            * pos["vol"]
            * FUTURES_DB[symbol]["mult"]
            * pos["dir"]
        )
        self.cash += pnl

        self.trade_log.append(
            {
                "Date": date,
                "Type": "Close",
                "Symbol": FUTURES_DB[symbol]["name"],
                "PnL": pnl,
                "Reason": reason,
            }
        )

        self.current_symbol = None
        self.current_pos = {}

    def _open_position(self, symbol, date, direction, equity):
        if date not in self.dm.data_dict[symbol].index:
            return
        bar = self.dm.data_dict[symbol].loc[date]

        # 资金管理：每次只开 1 手 (为了方便手动，不搞复杂仓位)
        # 或者：按资金利用率开仓，比如占用 30% 资金
        # 这里演示：固定 1 手，简单明了
        vol = 1

        # 检查保证金够不够
        margin_req = (
            bar["Close"]
            * vol
            * FUTURES_DB[symbol]["mult"]
            * FUTURES_DB[symbol]["margin"]
        )
        if self.cash < margin_req:
            print(f"资金不足开仓 {symbol}")
            return

        self.current_symbol = symbol
        self.current_pos = {"vol": vol, "entry": bar["Close"], "dir": direction}

        action = "Buy" if direction == 1 else "Short"
        self.trade_log.append(
            {
                "Date": date,
                "Type": action,
                "Symbol": FUTURES_DB[symbol]["name"],
                "Price": bar["Close"],
                "Score": f"Breakout",
            }
        )


# ==========================================
# 3. 运行与可视化
# ==========================================
if __name__ == "__main__":
    # 1. 下载数据
    dm = DataManager(FUTURES_DB)
    dm.download_data()

    # 2. 运行回测
    bt = SingleRotationBacktest(dm, StrategyConfig)
    results = bt.run()

    # 3. 统计
    final_equity = results["Equity"].iloc[-1]
    ret = (
        final_equity - StrategyConfig.INITIAL_CAPITAL
    ) / StrategyConfig.INITIAL_CAPITAL

    print("\n" + "=" * 40)
    print(f"单标的轮动策略 (Dual Thrust Score)")
    print(f"最终权益: {final_equity:,.2f}")
    print(f"总收益率: {ret*100:.2f}%")
    print(f"交易次数: {len(bt.trade_log)}")
    print("=" * 40)

    # 4. 绘图 (权益 + 持仓分布)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 权益曲线
    ax1.plot(results.index, results["Equity"], color="red")
    ax1.set_title("Equity Curve")
    ax1.grid(True)

    # 持仓历史 (用散点图表示每天持有什么)
    history_df = pd.DataFrame(bt.holding_history).set_index("Date")
    # 将品种名称映射为数字以便画图
    symbols = list(FUTURES_DB.keys()) + ["Cash"]
    y_map = {sym: i for i, sym in enumerate(symbols)}

    # 准备绘图数据
    dates = history_df.index
    y_values = [y_map.get(s, -1) for s in history_df["Symbol"]]

    ax2.scatter(dates, y_values, s=10, c="blue", marker="|")
    ax2.set_yticks(range(len(symbols)))
    # 替换Y轴标签为中文名
    y_labels = [FUTURES_DB[s]["name"] if s in FUTURES_DB else "空仓" for s in symbols]
    ax2.set_yticklabels(y_labels)
    ax2.set_title("Daily Holding Position")
    ax2.grid(True, axis="x")

    plt.tight_layout()
    plt.show()

    # 打印最近交易日志
    print("\n最近 10 笔操作:")
    print(pd.DataFrame(bt.trade_log).tail(10))
