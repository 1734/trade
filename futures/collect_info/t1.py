# -*- coding: utf-8 -*-
import os
import pandas as pd
import akshare as ak
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
#                                   用户配置区域 (CONFIG)
# ==============================================================================
CONFIG = {
    # 1. 网络代理 (仅用于 US 数据)
    "us_proxy": "http://127.0.0.1:7897",
    # 2. 保存路径
    "save_dir": "./futures_data_v2",
    # 3. 时间范围 (格式: YYYY-MM-DD)
    "start_date": "2024-01-01",
    "end_date": "2026-12-31",
    # 4. 目标列表
    # interval 支持: "1m"..."60m"(分钟), "1d"(日), "1w"(周), "1M"(月), "1Q"(季)
    "targets": [
        # --- 示例 1: 豆粕 15分钟线 ---
        {"type": "CN", "symbol": "m2605", "name": "豆粕2605_15分钟", "interval": "15m"},
        # --- 示例 2: 豆粕 日线 ---
        {"type": "CN", "symbol": "m2605", "name": "豆粕2605_日线", "interval": "1d"},
        # --- 示例 3: 原油 周线 (自动合成) ---
        {"type": "CN", "symbol": "sc2605", "name": "上海原油_周线", "interval": "1w"},
        {
            "type": "US",
            "symbol": "CL=F",  # 美原油
            "name": "美原油_月线",
            "interval": "1d",  # 自动合成月线
        },
    ],
}

# ==============================================================================
#                                   核心逻辑类
# ==============================================================================


class FuturesResampler:
    def __init__(self, config):
        self.cfg = config
        if not os.path.exists(self.cfg["save_dir"]):
            os.makedirs(self.cfg["save_dir"])

    def _toggle_proxy(self, enable=False):
        if enable and self.cfg["us_proxy"]:
            os.environ["https_proxy"] = self.cfg["us_proxy"]
            os.environ["http_proxy"] = self.cfg["us_proxy"]
        else:
            os.environ.pop("https_proxy", None)
            os.environ.pop("http_proxy", None)

    def run(self):
        print(">>> 任务开始...")
        for target in self.cfg["targets"]:
            self.process_target(target)
        print(f"\n>>> 全部完成。查看目录: {os.path.abspath(self.cfg['save_dir'])}")

    def process_target(self, target):
        interval = target["interval"]
        print(
            f"\n正在处理 [{target['type']}] {target['name']} ({target['symbol']}) - 周期: {interval} ..."
        )

        df = None
        try:
            if target["type"] == "CN":
                self._toggle_proxy(False)
                df = self._fetch_cn_data(target)
            elif target["type"] == "US":
                self._toggle_proxy(True)
                df = self._fetch_us_data(target)
        except Exception as e:
            print(f"  ❌ 获取失败: {e}")
        finally:
            self._toggle_proxy(False)

        if df is None or df.empty:
            print("  ⚠️ 无数据返回")
            return

        # 重采样
        if interval in ["1w", "1M", "1Q"]:
            df = self._resample_data(df, interval)
            if df.empty:
                print("  ⚠️ 重采样后无数据")
                return

        # 时间过滤 (分钟线除外)
        if "m" not in interval:
            df = self._filter_date(df)

        # 计算指标
        df = self._calc_indicators(df)

        # 保存与绘图
        self._save_and_plot(df, target)

    def _fetch_cn_data(self, target):
        symbol = target["symbol"]
        interval = target["interval"]

        if "m" in interval:
            period = interval.replace("m", "")
            try:
                df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
            except Exception as e:
                print(f"  -> 分钟接口报错: {e}")
                return None

            if df is None or df.empty:
                return None
            rename_map = {
                "datetime": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "hold": "OpenInterest",
            }
            df.rename(columns=rename_map, inplace=True)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)
            return df
        else:
            try:
                df = ak.futures_zh_daily_sina(symbol=symbol)
            except Exception as e:
                print(f"  -> 日线接口报错: {e}")
                return None

            if df is None or df.empty:
                return None
            rename_map = {
                "date": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "hold": "OpenInterest",
            }
            df.rename(columns=rename_map, inplace=True)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)
            return df

    def _fetch_us_data(self, target):
        ticker = yf.Ticker(target["symbol"])
        interval = target["interval"]
        yf_interval = interval

        if interval == "1Q":
            yf_interval = "1d"
        elif interval == "1w":
            yf_interval = "1wk"
        elif interval == "1M":
            yf_interval = "1mo"

        try:
            if "m" in interval:
                df = ticker.history(period="1mo", interval=yf_interval)
            else:
                df = ticker.history(
                    start=self.cfg["start_date"],
                    end=self.cfg["end_date"],
                    interval=yf_interval,
                )
        except:
            return None

        if df.empty:
            return None
        df.reset_index(inplace=True)
        date_col = "Date" if "Date" in df.columns else "Datetime"
        df.rename(
            columns={date_col: "Datetime", "Open Interest": "OpenInterest"},
            inplace=True,
        )

        if "OpenInterest" not in df.columns:
            df["OpenInterest"] = 0
        df["OpenInterest"] = df["OpenInterest"].fillna(0)
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
        df.set_index("Datetime", inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume", "OpenInterest"]]

    def _resample_data(self, df, interval):
        rule_map = {"1w": "W-FRI", "1M": "ME", "1Q": "QE"}
        rule = rule_map.get(interval)
        if not rule:
            return df

        print(f"  -> 正在重采样为: {interval} ...")
        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "OpenInterest": "last",
        }
        try:
            df_resampled = df.resample(rule).agg(agg_dict)
            df_resampled.dropna(subset=["Close"], inplace=True)
            return df_resampled
        except Exception as e:
            print(f"  ⚠️ 重采样失败: {e}")
            return df

    def _filter_date(self, df):
        if not self.cfg["start_date"] or not self.cfg["end_date"]:
            return df
        start = pd.to_datetime(self.cfg["start_date"])
        end = pd.to_datetime(self.cfg["end_date"]) + timedelta(days=1)
        return df[(df.index >= start) & (df.index < end)]

    def _calc_indicators(self, df):
        df["OI_Change"] = df["OpenInterest"].diff().fillna(0)
        return df

    def _save_and_plot(self, df, target):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_name = f"{target['name']}_{target['interval']}"
        csv_path = os.path.join(self.cfg["save_dir"], f"{safe_name}.csv")
        df.to_csv(csv_path)
        print(f"  ✅ CSV保存: {csv_path}")
        self._plot_wenhua(df, target, safe_name)

    def _plot_wenhua(self, df, target, filename):
        """
        绘制图表 (已调整间距)
        """
        c_up, c_down = "#ef5350", "#26a69a"

        # --- 关键修改 1: 调整 vertical_spacing (垂直间距) ---
        # 0.02 -> 0.08 (拉大间距)
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,  # <--- 这里改大了，图表之间会更开阔
            row_heights=[0.5, 0.15, 0.2, 0.15],  # 调整各子图高度比例
            subplot_titles=(
                f"{target['name']} ({target['interval']})",
                "成交量",
                "持仓量",
                "仓差",
            ),
        )

        # 1. K线
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K线",
                increasing_line_color=c_up,
                decreasing_line_color=c_down,
            ),
            row=1,
            col=1,
        )
        # 均线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"].rolling(5).mean(),
                line=dict(color="white", width=1),
                name="MA5",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"].rolling(20).mean(),
                line=dict(color="yellow", width=1),
                name="MA20",
            ),
            row=1,
            col=1,
        )

        # 2. 成交量
        colors_vol = [
            c_up if c >= o else c_down for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=colors_vol, name="成交量"),
            row=2,
            col=1,
        )

        # 3. 持仓量
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["OpenInterest"],
                mode="lines",
                line=dict(color="#ffd700", width=1.5),
                name="持仓量",
            ),
            row=3,
            col=1,
        )

        # 4. 仓差
        colors_oi = [c_up if v >= 0 else c_down for v in df["OI_Change"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["OI_Change"], marker_color=colors_oi, name="仓差"),
            row=4,
            col=1,
        )

        # --- 关键修改 2: 调整 height (总高度) ---
        # 1000 -> 1500 (增加总高度，防止子图被压扁)
        fig.update_layout(
            template="plotly_dark",
            height=1500,  # <--- 这里改大了
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50),  # 边距
        )

        html_path = os.path.join(self.cfg["save_dir"], f"{filename}.html")
        fig.write_html(html_path)
        print(f"  ✅ 图表生成: {html_path}")


if __name__ == "__main__":
    FuturesResampler(CONFIG).run()
