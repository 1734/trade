import backtrader as bt
import akshare as ak
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# --- 基础设置 ---
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 策略配置区 =================
START_CASH = 100000.0
START_DATE = "20220101"
END_DATE = datetime.date.today().strftime("%Y%m%d")

POOL_SIZE = 50
HOLD_NUM = 3
MA_PERIOD = 20
# ============================================


class SingleMATrendStrategy(bt.Strategy):
    params = (("ma_period", MA_PERIOD),)

    def __init__(self):
        # Data0 是大盘，作为时间基准
        self.bench = self.datas[0]
        self.bench_ma = bt.indicators.SMA(self.bench.close, period=self.p.ma_period)

        self.inds = {}
        # 遍历所有数据（包括大盘和个股）
        for d in self.datas:
            if d._name == "bench":
                continue

            # 关键点：即使这只股票还没上市，我们也可以预先定义指标
            # Backtrader 会自动处理，等数据来了指标才会有值
            self.inds[d] = {
                "ma": bt.indicators.SMA(d.close, period=self.p.ma_period),
                "roc": bt.indicators.RateOfChange(d.close, period=10),
            }

    def next(self):
        # 1. 大盘风控
        # 必须确保大盘数据是足量的
        if len(self.bench) < self.p.ma_period:
            return

        if self.bench.close[0] < self.bench_ma[0]:
            if any(
                [self.getposition(d).size > 0 for d in self.datas if d._name != "bench"]
            ):
                print(f"[{self.datas[0].datetime.date(0)}] 🌩️ 大盘风控: 全仓止损")
            for d in self.datas:
                if d._name != "bench" and self.getposition(d).size > 0:
                    self.close(d)
            return

        # 2. 个股交易逻辑
        candidates = []

        for d in self.datas:
            if d._name == "bench":
                continue

            # --- 核心修改：动态判断个股是否上市 ---
            # 如果当前时间点，这只股票还没有数据（未上市），或者上市不足20天
            # len(d) 会返回当前已有的K线条数
            if len(d) < self.p.ma_period:
                continue

            # 个股止损
            if self.getposition(d).size > 0:
                if d.close[0] < self.inds[d]["ma"][0]:
                    print(f"[{self.datas[0].datetime.date(0)}] ✂️ 止损: {d._name}")
                    self.close(d)

            # 选股逻辑
            if d.close[0] > self.inds[d]["ma"][0]:
                candidates.append((d, self.inds[d]["roc"][0]))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[1], reverse=True)
        target_stocks = [x[0] for x in candidates[:HOLD_NUM]]
        target_value = self.broker.get_value() / HOLD_NUM

        for d in target_stocks:
            if self.getposition(d).size == 0:
                if self.broker.getcash() > target_value * 0.8:
                    print(f"[{self.datas[0].datetime.date(0)}] 🚀 买入: {d._name}")
                    self.order_target_value(d, target=target_value)


# ================= 数据引擎 =================
def get_data_engine():
    feeds = []
    print("🚀 初始化数据引擎...")

    # 1. 必须先添加大盘 (Data0)，它决定了回测的起止时间！
    try:
        bench_df = ak.stock_zh_index_daily_em(symbol="sh000300")
        bench_df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bench_df.set_index("date", inplace=True)
        bench_df = bench_df.loc[START_DATE:END_DATE]
        print(
            f"   >>> 时间轴锁定: {bench_df.index[0].date()} 至 {bench_df.index[-1].date()}"
        )

        # 这里的 name='bench' 很重要，策略里通过它识别大盘
        feeds.append(bt.feeds.PandasData(dataname=bench_df, name="bench"))
    except Exception as e:
        print(f"❌ 大盘数据失败: {e}")
        return []

    # 2. 股票池 (不剔除新股)
    print(f"2. 扫描沪深300主板成分股...")
    try:
        all_cons = ak.index_stock_cons(symbol="000300")
        valid_cons = []
        # 依然只剔除科创/创业，保留主板，哪怕它是昨天才上市的
        for i, row in all_cons.iterrows():
            if not row["品种代码"].startswith(("688", "300", "8", "4")):
                valid_cons.append(row)

        valid_cons = pd.DataFrame(valid_cons).head(POOL_SIZE)
        total = len(valid_cons)

        for i, row in valid_cons.iterrows():
            code = row["品种代码"]
            name = row["品种名称"]

            try:
                # 下载数据
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

                # --- 这里删除了之前的剔除代码 ---
                # 哪怕 df 只有几行数据（刚上市），也照样加进去！

                print(
                    f"   [下载] {name} (最早日期: {df.index[0].date()}) ...", end="\r"
                )

                # 关键技巧：告诉Backtrader数据的有效起止时间
                # fromdate 设置为 20220101，即使股票2024年才有数据，BT也会处理成前面的为空
                feeds.append(bt.feeds.PandasData(dataname=df, name=name))
            except:
                continue

    except Exception as e:
        print(f"❌ 股票池失败: {e}")
        return []

    print("\n✅ 数据准备完毕")
    return feeds


# ================= 主程序 =================
if __name__ == "__main__":
    data_feeds = get_data_engine()
    if not data_feeds:
        exit()

    cerebro = bt.Cerebro()

    # 技巧：设置 cheat_on_open=True 可以避免某些新股数据对齐的边缘Bug，
    # 但对于日线策略通常不需要。这里保持默认。

    for d in data_feeds:
        cerebro.adddata(d)

    cerebro.addstrategy(SingleMATrendStrategy)
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(commission=0.0003)

    print(f"\n💰 回测开始 | {START_DATE} -> {END_DATE}")
    print("=" * 60)
    results = cerebro.run()
    print("=" * 60)

    final_val = cerebro.broker.getvalue()
    ret = ((final_val - START_CASH) / START_CASH) * 100
    print(f"🏆 最终资产: {final_val:.2f} (收益率: {ret:.2f}%)")

    # ... (预测部分代码与之前相同，此处省略以节省篇幅，直接复制上一段的即可) ...
    # 为了完整性，我把预测部分的结尾补上
    print(f"\n🔮 [明日实战指引]")
    strat = results[0]
    bench_data = strat.bench
    # 安全获取最后一个数据
    idx = len(bench_data) - 1
    if idx >= 0:
        last_close = bench_data.close[0]
        # 手动计算均线
        ma_period = MA_PERIOD
        if len(bench_data) >= ma_period:
            vals = bench_data.close.get(ago=0, size=ma_period)
            last_ma = sum(vals) / len(vals)

            if last_close < last_ma:
                print("🔴 市场环境：【空头】 -> 空仓休息")
            else:
                print("🟢 市场环境：【多头】 -> 关注线上强势股：")
                cands = []
                for d in strat.datas:
                    if d._name == "bench":
                        continue
                    if len(d) < ma_period:
                        continue
                    if d.close[0] > strat.inds[d]["ma"][0]:
                        roc = strat.inds[d]["roc"][0]
                        cands.append(
                            {
                                "name": d._name,
                                "c": d.close[0],
                                "ma": strat.inds[d]["ma"][0],
                                "roc": roc,
                            }
                        )
                cands.sort(key=lambda x: x["roc"], reverse=True)
                for s in cands[:3]:
                    print(f"🔥 {s['name']:<8} | 现价:{s['c']:.2f}")
