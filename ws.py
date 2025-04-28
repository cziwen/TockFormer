from polygon import WebSocketClient, RESTClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from typing import List, Optional
import numpy as np
import pandas as pd
import pytz

from datetime import datetime, timedelta, timezone

# CSVModifier中引用的方法，假设它们已经实现：
from CSVModifier import (
    add_factors,
    clean_outliers,
    aggregate_high_freq_to_low,
    aggregate_ohlcv
)

# 全局变量: 存储分钟OHLCV数据和秒级OHLCV数据
data_buffer_sec = []
data_buffer_min = []

RESOLUTION_MINUTE = 1  # 每条数据间隔的分钟数
SEQUENTIAL_LENGTH = 32  # 序列长度
WINDOW_SIZE_MIN = SEQUENTIAL_LENGTH * RESOLUTION_MINUTE  # 分钟滑动窗口大小
WINDOW_SIZE_SEC = WINDOW_SIZE_MIN * 60  # 秒级滑动窗口大小

has_fetched_latest_data = False


########################
# 辅助函数：使用 Polygon RESTClient 获取历史数据
########################
def fetch_polygon_history (rest_client, symbol, timespan="minute", bars=50):
    """
    从 Polygon 获取最近 N 条历史数据，并返回一个带有 timestamp 的 DataFrame。
    timespan 可以是 'minute', 'second' 等；
    bars 是要获取的数量（最多 bars * 2 条会被抓取，然后在函数外部筛选）。
    """

    # 当前时间 (UTC-aware)
    now_utc = datetime.now (timezone.utc)

    # 时间区间回溯（根据 timespan 类型确定）
    if timespan == "minute":
        delta = timedelta (minutes=bars * 2)
    elif timespan == "second":
        delta = timedelta (seconds=bars * 2)
    else:
        delta = timedelta (days=1)

    from_utc = now_utc - delta

    # 转为符合 Polygon 要求的时间字符串
    from_str = from_utc.strftime ("%Y-%m-%d")
    to_str = now_utc.strftime ("%Y-%m-%d")

    # 调用 Polygon API，ticker 传 symbol
    aggs = rest_client.get_aggs (
        ticker=symbol,
        multiplier=1,
        timespan=timespan,
        from_=from_str,
        to=to_str,
        limit=bars * 2,
        sort="desc",  # 倒序返回（新数据在前）
        adjusted=False  # 是否复权
    )

    # 转为 DataFrame（不同版本返回的对象可能不同）
    df = pd.DataFrame (aggs)

    # 处理时间戳字段
    if not df.empty:
        df["timestamp"] = pd.to_datetime (df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert (pytz.timezone ('US/Eastern')) # 转换为美东时间

    return df


########################
# WebSocket 消息处理
########################
def process_message (m: WebSocketMessage):
    """
    将 WebSocketMessage 解析为包含 OHLCV 等字段的字典。
    假设消息包含以下属性 (Polygon Aggregates):
      - m.start_timestamp / m.s: 开始时间毫秒
      - m.open / m.o
      - m.high / m.h
      - m.low / m.l
      - m.close / m.c
      - m.volume / m.v
      - m.vwap
      - m.event_type / m.ev
    若字段或格式不同，请根据实际情况进行调整。
    """
    try:
        ohlcv = {
            "timestamp": datetime.fromtimestamp (m.start_timestamp / 1000.0),  # k线开始的时间
            "open": m.open,
            "high": m.high,
            "low": m.low,
            "vwap": m.vwap,
            "close": m.close,
            "volume": m.volume
        }
        return ohlcv, m.event_type
    except Exception as e:
        print ("Error processing message:", e)
        return None, None


def preprocess_data () -> pd.DataFrame:
    """
    将 OHLCV 字典列表转换为 Pandas DataFrame，进行数据预处理。
    - 分钟数据 => 聚合成 5min
    - 秒数据 => 聚合成 5min
    - 合并后加因子
    """
    global data_buffer_sec, data_buffer_min

    # 1) 分钟数据 => 5min
    df_min = pd.DataFrame (data_buffer_min)
    if df_min.empty:
        return pd.DataFrame ()

    df_5min = aggregate_ohlcv (df_min, freq="5min")
    df_5min = clean_outliers (df_5min, columns=["open", "high", "low", "close"], z_thresh=8)

    # 2) 秒数据 => 5min
    df_sec = pd.DataFrame (data_buffer_sec)
    if df_sec.empty:
        return pd.DataFrame ()

    df_sec_5min_agg = aggregate_high_freq_to_low (df_sec, freq="5min", timestamp="timestamp")
    df_sec_5min_agg = clean_outliers (df_sec_5min_agg, columns=["open", "high", "low", "close"], z_thresh=8)

    # 3) 加因子
    df_5min = add_factors (df_5min)

    # 4) 合并
    df_merged = pd.merge (df_sec_5min_agg, df_5min, on="timestamp", how="inner")
    return df_merged


def predict (model_input: pd.DataFrame) -> float:
    """
    用预处理后的数据做预测。预测可以是下一个价格、涨跌信号等等。
    这里简单示例：最后一个 close 略微上升。
    """
    dummy_prediction = model_input["close"].iloc[-1] * 1.001
    return dummy_prediction


def handle_msg (msg: List[WebSocketMessage]):
    """
    WebSocket 收到新消息后的回调。
    """
    global data_buffer_sec, data_buffer_min, has_fetched_latest_data

    for m in msg:
        ohlcv, ev = process_message (m)
        if ohlcv is not None:
            # ev == "A" 代表 秒级数据（A.<symbol>）
            # ev == "AM" 代表 分钟级数据（AM.<symbol>）
            if ev == "A":
                data_buffer_sec.append (ohlcv)
                # 控制秒级滑动窗口大小
                if len (data_buffer_sec) > WINDOW_SIZE_SEC:
                    data_buffer_sec = data_buffer_sec[-WINDOW_SIZE_SEC:]
            elif ev == "AM":
                data_buffer_min.append (ohlcv)
                # 控制分钟级滑动窗口大小
                if len (data_buffer_min) > WINDOW_SIZE_MIN:
                    data_buffer_min = data_buffer_min[-WINDOW_SIZE_MIN:]
            else:
                print (f"Unexpected event type: {ev}")

        print (
            f"[{datetime.now ().time ()}] 成功处理完 ohlcv: {ohlcv}, 当前秒级缓存数={len (data_buffer_sec)}, 分钟级缓存数={len (data_buffer_min)}")

    # 如果分钟数据大于一定数量，且尚未获取过历史数据，则调用 RESTClient 获取
    if len (data_buffer_min) > 0 and not has_fetched_latest_data:
        print ("开始获取往期数据...")

        # ------------------------------------------------------------------
        # fetch_polygon_history 返回的 DataFrame（或可转换为 DataFrame 的数据）
        # 已经是"时间降序"的，所以只需要 head(...) 获取最新的几行
        # ------------------------------------------------------------------
        past_min_data = fetch_polygon_history (rest_client, symbol, timespan="minute", bars=WINDOW_SIZE_MIN * 2)
        past_sec_data = fetch_polygon_history (rest_client, symbol, timespan="second", bars=WINDOW_SIZE_SEC * 2)

        # 转为 DataFrame
        df_past_min = pd.DataFrame (past_min_data.to_dict (orient="records"))
        df_past_sec = pd.DataFrame (past_sec_data.to_dict (orient="records"))

        # 仅取最新的 N 条记录（因为是时间降序，head(n) 即可）
        df_past_min = df_past_min.head (WINDOW_SIZE_MIN * 2)
        df_past_sec = df_past_sec.head (WINDOW_SIZE_SEC * 2)

        # timestamp 转为 datetime
        df_past_min["timestamp"] = pd.to_datetime (df_past_min["timestamp"])
        df_past_sec["timestamp"] = pd.to_datetime (df_past_sec["timestamp"])

        # 再将其按时间升序排一下
        df_past_min.sort_values (by="timestamp", inplace=True)
        df_past_sec.sort_values (by="timestamp", inplace=True)

        # 合并历史数据和实时 buffer
        # 注意这里是先把 buffer 转为 DataFrame，然后做 concat 或者直接加到列表
        df_min_combined = pd.DataFrame (data_buffer_min + df_past_min.to_dict (orient="records"))
        df_sec_combined = pd.DataFrame (data_buffer_sec + df_past_sec.to_dict (orient="records"))

        # 去重
        df_min_combined.drop_duplicates (subset="timestamp", keep="last", inplace=True)
        df_sec_combined.drop_duplicates (subset="timestamp", keep="last", inplace=True)

        # 再次按时间升序
        df_min_combined.sort_values (by="timestamp", inplace=True)
        df_sec_combined.sort_values (by="timestamp", inplace=True)

        # 截断到固定窗口大小
        df_min_combined = df_min_combined.tail (WINDOW_SIZE_MIN)
        df_sec_combined = df_sec_combined.tail (WINDOW_SIZE_SEC)

        # 进行“每条间隔约 1 分钟”校验（分钟数据）
        valid_min_data = False
        if len (df_min_combined) > 1:
            # 计算相邻行的时间差（秒）
            time_diffs = df_min_combined["timestamp"].diff ().dt.total_seconds ().iloc[1:]

            # 设置一个容差，比如 ±1 秒
            lower_bound = 58
            upper_bound = 62

            # 检查是否所有的时间差都在 [58, 62] 区间内
            if time_diffs.between (lower_bound, upper_bound).all ():
                valid_min_data = True

        # 如果分钟数据有效，则更新全局 buffer；否则给出提示
        if valid_min_data:
            data_buffer_min = df_min_combined.to_dict (orient="records")
            data_buffer_sec = df_sec_combined.to_dict (orient="records")

            has_fetched_latest_data = True
            print ("成功获取并合并往期数据，且分钟数据间隔校验通过。")
        else:
            has_fetched_latest_data = True
            print ("警告：合并后的历史分钟数据不连续，未设置 has_fetched_latest_data。")

    # 当数据量达到预设滑动窗口大小后，进行预处理和预测
    if len (data_buffer_min) >= WINDOW_SIZE_MIN:
        print ("开始数据预处理...")
        preprocessed_df = preprocess_data ()
        if not preprocessed_df.empty:
            print (f"预处理的数据维度: {preprocessed_df.shape}")
            # 这里可插入你的模型预测逻辑
            # prediction = predict(preprocessed_df)
            # print(f"预测结果: {prediction}")
        print ("预处理结束。")


########################
# 主入口
########################
if __name__ == "__main__":
    # 你的 Polygon API Key
    API_KEY = "oilTTMMexxTBTmjivaMq3R0Y9ZS1BKbK"

    # 初始化 Polygon RESTClient
    rest_client = RESTClient (api_key=API_KEY)

    # 初始化 Polygon WebSocket 客户端（示例使用 Delayed Feed，可按需求改为 RealTime)
    ws = WebSocketClient (
        api_key=API_KEY,
        feed=Feed.Delayed,
        market=Market.Stocks
    )

    symbol = "SPY"

    # 订阅 SPY 的分钟和秒聚合数据
    ws.subscribe (f"AM.{symbol}")
    ws.subscribe (f"A.{symbol}")

    # 启动 WebSocket 客户端
    ws.run (handle_msg=handle_msg)
