import backtrader as bt
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# --- 基础设置 ---
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 策略配置区 =================
START_CASH = 100000.0
START_DATE = "20250101"
END_DATE = datetime.date.today().strftime("%Y%m%d")

POOL_SIZE = 50  # 沪深300主板头部50只
HOLD_NUM = 3  # 持有3只
REBALANCE_DAYS = 5  # 5天调仓

# --- 估值红线 (实战过滤用) ---
MAX_PE = 60.0  # 市盈率上限 (超过60视为泡沫)
MAX_PB = 8.0  # 市净率上限
# ============================================


# 1. 动量因子
class VolatilityAdjustedMomentum(bt.Indicator):
    lines = ("score",)
    params = (("period", 20),)

    def __init__(self):
        roc = bt.indicators.ROC(self.data, period=self.p.period)
        std = bt.indicators.StdDev(self.data, period=self.p.period)
        self.lines.score = roc / (std + 0.0001)


# 2. 策略类
class ValueMomentumStrategy(bt.Strategy):
    params = (
        ("momentum_period", 20),
        ("trail_stop", 0.10),
    )

    def __init__(self):
        self.timer = 0
        self.inds = {}
        self.bench = self.getdatabyname("bench")
        self.bench_ma = bt.indicators.SMA(self.bench.close, period=20)

        for d in self.datas:
            if d._name == "bench":
                continue
            self.inds[d] = {
                "score": VolatilityAdjustedMomentum(
                    d.close, period=self.p.momentum_period
                ),
                "ma20": bt.indicators.SMA(d.close, period=20),
                "high_since_entry": 0.0,
            }

    def next(self):
        # A. 移动止损
        for d in self.datas:
            if d._name == "bench":
                continue
            pos = self.getposition(d)
            if pos.size > 0:
                if d.high[0] > self.inds[d]["high_since_entry"]:
                    self.inds[d]["high_since_entry"] = d.high[0]
                high_price = self.inds[d]["high_since_entry"]
                if high_price > 0:
                    drawdown = (d.close[0] - high_price) / high_price
                    if drawdown < -self.p.trail_stop:
                        self.close(d)
                        self.inds[d]["high_since_entry"] = 0

        # B. 调仓周期
        self.timer += 1
        if self.timer % REBALANCE_DAYS != 0:
            return

        # C. 大盘风控
        if self.bench.close[0] < self.bench_ma[0]:
            for d in self.datas:
                if d._name != "bench" and self.getposition(d).size > 0:
                    self.order_target_percent(d, target=0.0)
            return

        # D. 选股 (这里主要基于价格，因为历史PE获取慢，PE过滤放在实战预测环节)
        candidates = []
        for d in self.datas:
            if d._name == "bench":
                continue
            if len(d) < 60:
                continue
            if d.close[0] < self.inds[d]["ma20"][0]:
                continue

            score = self.inds[d]["score"][0]
            candidates.append((d, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        target_stocks = [x[0] for x in candidates[:HOLD_NUM]]

        target_weight = 0.95 / HOLD_NUM

        for d in self.datas:
            if d._name == "bench":
                continue
            if d in target_stocks:
                if self.getposition(d).size == 0:
                    self.inds[d]["high_since_entry"] = d.high[0]
                self.order_target_percent(d, target=target_weight)
            else:
                if self.getposition(d).size > 0:
                    self.order_target_percent(d, target=0.0)


# ================= 数据引擎 =================
def get_data_engine():
    feeds = []
    print("🚀 初始化数据引擎...")

    # 1. 大盘
    try:
        bench_df = ak.stock_zh_index_daily(symbol="sh000300")
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bench_df.set_index("date", inplace=True)
        bench_df = bench_df.loc[START_DATE:END_DATE]
        feeds.append(bt.feeds.PandasData(dataname=bench_df, name="bench"))
    except:
        print("大盘数据获取失败")
        return []

    # 2. 股票池 (剔除 300/688/北交所)
    print(f"📡 扫描沪深300主板核心资产 (前{POOL_SIZE}只)...")
    try:
        all_cons = ak.index_stock_cons(symbol="000300")
        valid_stocks = []
        for i, row in all_cons.iterrows():
            code = row["品种代码"]
            if code.startswith(("688", "300", "8", "4")):
                continue
            valid_stocks.append(row)

        valid_cons = pd.DataFrame(valid_stocks).head(POOL_SIZE)

        total = len(valid_cons)
        count = 0
        for i, row in valid_cons.iterrows():
            code = row["品种代码"]
            name = row["品种名称"]
            count += 1
            print(f"   [{count}/{total}] 下载: {name} ...", end="\r")

            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust="qfq",
                )
                if df.empty:
                    continue
                df.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                    },
                    inplace=True,
                )
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                feeds.append(bt.feeds.PandasData(dataname=df, name=name))
            except:
                continue

        print("\n✅ 数据准备完毕")
        return feeds
    except Exception as e:
        print(f"错误: {e}")
        return []


# ================= 核心：获取实时估值数据 =================
def get_current_valuation(stock_names_list):
    """
    获取全市场实时估值，并匹配我们的候选股
    """
    print("\n🔍 正在拉取实时 PE/PB 数据进行基本面体检...")
    try:
        # 获取全市场实时行情（包含PE/PB）
        spot_df = ak.stock_zh_a_spot_em()
        # 建立 名字 -> (代码, PE, PB) 的映射
        # spot_df 列名: 代码, 名称, 市盈率-动态, 市净率
        val_map = {}
        for i, row in spot_df.iterrows():
            name = row["名称"]
            pe = row["市盈率-动态"]
            pb = row["市净率"]
            code = row["代码"]
            val_map[name] = {"code": code, "pe": pe, "pb": pb}

        return val_map
    except Exception as e:
        print(f"估值数据获取失败: {e}")
        return {}


# ================= 主程序 =================
if __name__ == "__main__":
    data_feeds = get_data_engine()
    if not data_feeds:
        exit()

    cerebro = bt.Cerebro()
    for d in data_feeds:
        cerebro.adddata(d)

    cerebro.addstrategy(ValueMomentumStrategy)
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(commission=0.0003)

    print(f"\n💰 初始本金: {START_CASH}")
    print("=" * 60)
    results = cerebro.run()
    print("=" * 60)

    final_val = cerebro.broker.getvalue()
    ret = ((final_val - START_CASH) / START_CASH) * 100
    print(f"🏆 最终资产: {final_val:.2f} (收益率: {ret:.2f}%)")

    # ================= 实战预测 (加入估值过滤) =================
    print("\n🔮 [明日实战指引 - 双重过滤版]")
    print("-" * 50)

    strat = results[0]
    bench_data = strat.bench

    # 1. 大盘过滤器
    if bench_data.close[0] < sum(bench_data.close.get(ago=0, size=20)) / 20:
        print("🔴 市场状态：【弱势】(大盘跌破20日线)")
        print("👉 操作建议：【空仓休息】。")
    else:
        print("🟢 市场状态：【强势】(大盘趋势向上)")

        # 2. 动量初选
        candidates = []
        for d in strat.datas:
            if d._name == "bench":
                continue
            try:
                score = strat.inds[d]["score"][0]
                close = d.close[0]
                ma20 = strat.inds[d]["ma20"][0]
                if close > ma20:
                    candidates.append({"name": d._name, "score": score, "close": close})
            except:
                continue

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_momentum = candidates[:10]  # 先取前10名候选

        # 3. 估值决选 (获取实时 PE/PB)
        val_map = get_current_valuation([x["name"] for x in top_momentum])

        print(f"\n👉 候选股体检报告 (剔除 PE>{MAX_PE} 或 PB>{MAX_PB} 的泡沫股):")
        print(
            f"{'股票名称':<8} | {'动量分':<6} | {'PE(市盈)':<8} | {'PB(市净)':<8} | {'结论'}"
        )
        print("-" * 60)

        valid_targets = []

        for stock in top_momentum:
            name = stock["name"]
            info = val_map.get(name)

            if info:
                pe = info["pe"]
                pb = info["pb"]

                # 检查逻辑
                is_safe = True
                status = "✅ 建议买入"

                if pe > MAX_PE:
                    is_safe = False
                    status = f"❌ 估值过高 (PE>{MAX_PE})"
                elif pe < 0:
                    is_safe = False
                    status = "❌ 业绩亏损 (PE<0)"
                elif pb > MAX_PB:
                    is_safe = False
                    status = f"❌ 市净率高 (PB>{MAX_PB})"

                print(
                    f"{name:<8} | {stock['score']:.1f}   | {pe:<8} | {pb:<8} | {status}"
                )

                if is_safe:
                    stock["pe"] = pe
                    valid_targets.append(stock)
            else:
                print(f"{name:<8} | 数据缺失，跳过")

        # 4. 最终输出
        print("-" * 60)
        print(f"🔥 最终优选名单 (建议明日 09:25 挂单):")

        final_picks = valid_targets[:3]  # 只买前3
        if not final_picks:
            print("   (无符合条件的股票，建议空仓)")
        else:
            for pick in final_picks:
                hands = int((START_CASH / 3) / pick["close"] / 100) * 100
                print(f"   🚀 {pick['name']} \t(PE: {pick['pe']}) \t-> 买入 {hands} 股")
