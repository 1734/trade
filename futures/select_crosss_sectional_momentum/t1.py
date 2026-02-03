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
class StrategyConfig:
    # --- 账户设置 ---
    INITIAL_CAPITAL = 80000  # 50万 (多品种需要资金量稍大)
    MAX_LONG_POS = 2  # 最多做多 2 个品种
    MAX_SHORT_POS = 2  # 最多做空 2 个品种

    # --- 选品参数 (动量) ---
    MOMENTUM_WINDOW = 20  # 看过去20天的涨跌幅来排名
    HOLDING_PERIOD = 5  # 调仓周期 (每5天重新排名换仓，避免频繁交易)

    # --- 风控参数 ---
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 3.5  # 3倍ATR止损
    RISK_PER_TRADE = 0.02  # 单个品种亏损限额 2%


# 品种数据库 (代码: {名称, 乘数, 保证金})
FUTURES_DB = {
    "rb0": {"name": "螺纹", "mult": 10, "margin": 0.13},
    "i0": {"name": "铁矿", "mult": 100, "margin": 0.15},
    "cu0": {"name": "沪铜", "mult": 5, "margin": 0.10},
    "ta0": {"name": "PTA", "mult": 5, "margin": 0.10},
    "ma0": {"name": "甲醇", "mult": 10, "margin": 0.12},
    "m0": {"name": "豆粕", "mult": 10, "margin": 0.10},
    "p0": {"name": "棕榈", "mult": 10, "margin": 0.12},
    "sr0": {"name": "白糖", "mult": 10, "margin": 0.10},
    "ru0": {"name": "橡胶", "mult": 10, "margin": 0.15},
    "fg0": {"name": "玻璃", "mult": 20, "margin": 0.15},
}


# ==========================================
# 1. 数据管理器 (下载并对齐数据)
# ==========================================
class DataManager:
    def __init__(self, symbol_dict):
        self.symbol_dict = symbol_dict
        self.data_dict = {}  # {symbol: dataframe}
        self.panel_data = None  # 对齐后的数据

    def download_and_align(self):
        print("正在下载并清洗多品种数据...")
        close_dict = {}

        for symbol, info in self.symbol_dict.items():
            try:
                # 下载数据
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

                # 计算 ATR (用于风控)
                high_low = df["High"] - df["Low"]
                high_close = (df["High"] - df["Close"].shift()).abs()
                low_close = (df["Low"] - df["Close"].shift()).abs()
                df["ATR"] = (
                    pd.concat([high_low, high_close, low_close], axis=1)
                    .max(axis=1)
                    .rolling(14)
                    .mean()
                )

                # 截取最近3年
                start_date = (
                    datetime.datetime.now() - datetime.timedelta(days=365 * 3)
                ).strftime("%Y-%m-%d")
                df = df[df.index >= start_date]

                self.data_dict[symbol] = df
                close_dict[symbol] = df["Close"]
                print(f"[{info['name']}] 数据就绪")
                time.sleep(0.3)

            except Exception as e:
                print(f"[{symbol}] 下载失败: {e}")

        # 生成 Close 价格矩阵 (用于横向比较)
        # fillna(method='ffill') 处理停牌数据
        self.panel_close = pd.DataFrame(close_dict).fillna(method="ffill").dropna()
        print(f"数据对齐完成，共 {len(self.panel_close)} 个交易日")


# ==========================================
# 2. 投资组合回测引擎
# ==========================================
class PortfolioBacktest:
    def __init__(self, data_manager, config):
        self.dm = data_manager
        self.cfg = config

        self.cash = config.INITIAL_CAPITAL
        self.positions = (
            {}
        )  # {symbol: {'vol': 1, 'entry_price': 3000, 'direction': 1, 'stop_loss': 2900}}
        self.equity_curve = []
        self.trade_log = []

    def calculate_lots(self, symbol, price, atr, equity):
        """ATR倒算仓位"""
        info = FUTURES_DB[symbol]
        if atr == 0 or price == 0:
            return 0

        # 风险平价
        stop_dist = self.cfg.ATR_STOP_MULTIPLIER * atr
        risk_money = equity * self.cfg.RISK_PER_TRADE
        risk_lots = int(risk_money / (stop_dist * info["mult"]))

        # 保证金限制
        margin_per_lot = price * info["mult"] * info["margin"]
        max_lots = int(equity * 0.3 / margin_per_lot)  # 单品种最多占用30%资金

        return max(0, min(risk_lots, max_lots))

    def run(self):
        print("\n开始多品种轮动回测...")
        dates = self.dm.panel_close.index
        rebalance_counter = 0

        for date in dates:
            # 1. --- 每日结算 (Mark to Market) ---
            current_equity = self.cash
            margin_used = 0

            # 检查持仓盈亏和止损
            symbols_to_close = []

            for symbol, pos in self.positions.items():
                # 获取当日数据
                if date not in self.dm.data_dict[symbol].index:
                    continue
                bar = self.dm.data_dict[symbol].loc[date]

                # 计算浮动盈亏
                pnl = (
                    (bar["Close"] - pos["entry_price"])
                    * pos["vol"]
                    * FUTURES_DB[symbol]["mult"]
                    * pos["direction"]
                )
                current_equity += pnl

                # 计算保证金占用
                margin_used += (
                    bar["Close"]
                    * pos["vol"]
                    * FUTURES_DB[symbol]["mult"]
                    * FUTURES_DB[symbol]["margin"]
                )

                # --- 盘中风控 (ATR止损) ---
                # 多单止损
                if pos["direction"] == 1 and bar["Low"] <= pos["stop_loss"]:
                    exit_price = min(bar["Open"], pos["stop_loss"])  # 简单模拟
                    realized_pnl = (
                        (exit_price - pos["entry_price"])
                        * pos["vol"]
                        * FUTURES_DB[symbol]["mult"]
                    )
                    self.cash += realized_pnl
                    self.trade_log.append(
                        {
                            "Date": date,
                            "Symbol": symbol,
                            "Type": "StopLoss(L)",
                            "PnL": realized_pnl,
                        }
                    )
                    symbols_to_close.append(symbol)

                # 空单止损
                elif pos["direction"] == -1 and bar["High"] >= pos["stop_loss"]:
                    exit_price = max(bar["Open"], pos["stop_loss"])
                    realized_pnl = (
                        (pos["entry_price"] - exit_price)
                        * pos["vol"]
                        * FUTURES_DB[symbol]["mult"]
                    )
                    self.cash += realized_pnl
                    self.trade_log.append(
                        {
                            "Date": date,
                            "Symbol": symbol,
                            "Type": "StopLoss(S)",
                            "PnL": realized_pnl,
                        }
                    )
                    symbols_to_close.append(symbol)

            # 执行止损平仓
            for s in symbols_to_close:
                del self.positions[s]

            # 记录权益
            self.equity_curve.append({"Date": date, "Equity": current_equity})

            # 2. --- 选品与调仓 (每N天一次) ---
            rebalance_counter += 1
            if rebalance_counter >= self.cfg.HOLDING_PERIOD:
                rebalance_counter = 0
                self._rebalance(date, current_equity)

        return pd.DataFrame(self.equity_curve).set_index("Date")

    def _rebalance(self, date, current_equity):
        """核心选品逻辑：截面动量排名"""

        # 1. 计算所有品种过去N天的涨跌幅 (Momentum)
        # 获取 date 这一行的数据
        try:
            # 过去N天的价格
            idx = self.dm.panel_close.index.get_loc(date)
            if idx < self.cfg.MOMENTUM_WINDOW:
                return

            prices_now = self.dm.panel_close.iloc[idx]
            prices_prev = self.dm.panel_close.iloc[idx - self.cfg.MOMENTUM_WINDOW]

            # 计算收益率
            returns = (prices_now - prices_prev) / prices_prev

            # 2. 排序
            # 剔除停牌或无数据的
            valid_returns = returns.dropna()
            if len(valid_returns) < 4:
                return  # 品种太少不交易

            # 排序：从高到低
            ranked = valid_returns.sort_values(ascending=False)

            # 选出最强和最弱
            # 逻辑：只有当涨幅 > 0 才做多，跌幅 < 0 才做空 (趋势过滤)
            long_candidates = ranked.head(self.cfg.MAX_LONG_POS)
            long_targets = long_candidates[long_candidates > 0].index.tolist()

            short_candidates = ranked.tail(self.cfg.MAX_SHORT_POS)
            short_targets = short_candidates[short_candidates < 0].index.tolist()

            # 3. 执行调仓
            # A. 平掉不在目标列表里的旧仓位
            current_holdings = list(self.positions.keys())
            for symbol in current_holdings:
                # 如果持有多单，但不在新的多单目标里 -> 平仓
                if (
                    self.positions[symbol]["direction"] == 1
                    and symbol not in long_targets
                ):
                    self._close_position(symbol, date, "Rotation_Exit")
                # 如果持有空单，但不在新的空单目标里 -> 平仓
                elif (
                    self.positions[symbol]["direction"] == -1
                    and symbol not in short_targets
                ):
                    self._close_position(symbol, date, "Rotation_Exit")

            # B. 开新仓
            # 开多单
            for symbol in long_targets:
                if symbol not in self.positions:
                    self._open_position(symbol, date, 1, current_equity)

            # 开空单
            for symbol in short_targets:
                if symbol not in self.positions:
                    self._open_position(symbol, date, -1, current_equity)

        except Exception as e:
            print(f"调仓出错: {e}")

    def _close_position(self, symbol, date, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        bar = self.dm.data_dict[symbol].loc[date]
        price = bar["Close"]  # 轮动通常按收盘价结算

        pnl = (
            (price - pos["entry_price"])
            * pos["vol"]
            * FUTURES_DB[symbol]["mult"]
            * pos["direction"]
        )
        self.cash += pnl
        self.trade_log.append(
            {"Date": date, "Symbol": symbol, "Type": reason, "PnL": pnl}
        )
        del self.positions[symbol]

    def _open_position(self, symbol, date, direction, equity):
        bar = self.dm.data_dict[symbol].loc[date]
        price = bar["Close"]
        atr = bar["ATR"]

        # 计算手数
        lots = self.calculate_lots(symbol, price, atr, equity)
        if lots == 0:
            return

        # 计算止损价
        if direction == 1:
            stop_loss = price - (self.cfg.ATR_STOP_MULTIPLIER * atr)
        else:
            stop_loss = price + (self.cfg.ATR_STOP_MULTIPLIER * atr)

        self.positions[symbol] = {
            "vol": lots,
            "entry_price": price,
            "direction": direction,
            "stop_loss": stop_loss,
        }
        action = "Buy" if direction == 1 else "Short"
        # 开仓不扣钱，只占保证金，平仓才结算盈亏到 cash
        self.trade_log.append(
            {
                "Date": date,
                "Symbol": symbol,
                "Type": action,
                "Price": price,
                "Lots": lots,
            }
        )


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 下载数据
    dm = DataManager(FUTURES_DB)
    dm.download_and_align()

    # 2. 运行回测
    bt = PortfolioBacktest(dm, StrategyConfig)
    results = bt.run()

    # 3. 统计与绘图
    final_equity = results["Equity"].iloc[-1]
    ret = (
        final_equity - StrategyConfig.INITIAL_CAPITAL
    ) / StrategyConfig.INITIAL_CAPITAL

    print("\n" + "=" * 40)
    print(
        f"多品种轮动策略 (Top {StrategyConfig.MAX_LONG_POS} Long / Top {StrategyConfig.MAX_SHORT_POS} Short)"
    )
    print(f"最终权益: {final_equity:,.2f}")
    print(f"总收益率: {ret*100:.2f}%")
    print(f"交易记录数: {len(bt.trade_log)}")
    print("=" * 40)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["Equity"], color="red", label="Portfolio Equity")
    plt.title("Cross-Sectional Momentum Strategy (Futures)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 打印最近的持仓
    print("\n当前持仓:")
    for s, p in bt.positions.items():
        d = "多" if p["direction"] == 1 else "空"
        print(f"{FUTURES_DB[s]['name']}: {d}单 {p['vol']}手, 成本 {p['entry_price']}")
