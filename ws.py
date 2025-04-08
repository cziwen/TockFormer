import time
import json
import joblib
import torch
import pandas as pd
import requests
import websocket
import pytz
import sys
from datetime import datetime, timedelta

from CSVModifier import clean_outliers, add_factors
from TransformerModel import TimeSeriesTransformer
from BiasCorrector import BiasCorrector
from Util import create_prediction_sequence

# 全局变量：存储已完成的蜡烛数据和当前未完成的蜡烛
aggregated_candles = []  # 已完成的5分钟蜡烛，每个元素为一个 dict
current_candle = None  # 当前正在聚合的蜡烛
historical_data_patched = False  # 标记是否已补充过历史数据
ws_start_time = None

API_KEY = "cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00"  # 替换为你的 API Key

# 参数设置
symbol = "SPY"  # 订阅的股票代码
seq_length = 32  # 模型要求的蜡烛数量
new_column_order = [
    'time', 'open', 'high', 'low', 'close', 'vwap', 'volume',
    'EMA5_open', 'EMA10_open', 'EMA20_open',
    'EMA5_high', 'EMA10_high', 'EMA20_high',
    'EMA5_low', 'EMA10_low', 'EMA20_low',
    'EMA5_close', 'EMA10_close', 'EMA20_close',
    'RSI_open', 'MACD_value_open', 'MACD_signal_open', 'MACD_histogram_open', 'ROC_open',
    'RSI_high', 'MACD_value_high', 'MACD_signal_high', 'MACD_histogram_high', 'ROC_high',
    'RSI_low', 'MACD_value_low', 'MACD_signal_low', 'MACD_histogram_low', 'ROC_low',
    'RSI_close', 'MACD_value_close', 'MACD_signal_close', 'MACD_histogram_close', 'ROC_close',
    'Stoch_K', 'Stoch_D', 'Volume_SMA5', 'Volume_SMA10', 'Volume_ROC', 'OBV',
    'open_volatility', 'high_volatility', 'low_volatility', 'close_volatility', 'volume_volatility'
]

# 加载模型、scaler 和 bias_corrector
device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
model_path = "models/transformer_regression_5min_lam.pth"
scaler_path = "models/scaler_regression_5min_lam.pkl"
bias_corrector_path = "models/corrector_regression_5min_lam.pkl"

model = TimeSeriesTransformer (input_dim=49, model_dim=64, num_heads=4, num_layers=2,
                               dropout=0.2, seq_length=seq_length, output_dim=4)
model.load_state_dict (torch.load (model_path, map_location=device))
model.to (device)

bias_corrector = BiasCorrector.load (bias_corrector_path)
scaler = joblib.load (scaler_path)

# --- 新增：全局变量用于显示 ---
current_status_msg = ""
current_prediction_msg = ""


def update_display ():
    """
    通过 ANSI 转义序列更新终端中固定区域（两行）的显示内容：
      第一行：预测信息；
      第二行：状态信息。
    """
    # 移动光标到两行预留区域的起始位置，并更新
    sys.stdout.write ("\033[2F")  # 向上移动两行
    sys.stdout.write ("\033[K")  # 清除当前行
    sys.stdout.write ("Prediction: " + current_prediction_msg + "\n")
    sys.stdout.write ("\033[K")
    sys.stdout.write ("Status    : " + current_status_msg + "\n")
    sys.stdout.flush ()


def fetch_data (symbol, interval='5min', num_bars=50):
    """
    使用 Finhub REST API 获取指定股票的 K 线数据。
    """
    interval_mapping = {
        '1min': '1',
        '5min': '5',
        '10min': '10',  # 注意: Finhub API 可能不支持 10 分钟K线
        '15min': '15',
        '30min': '30',
        '1h': '60',
        '1d': 'D'
    }
    interval_seconds = {
        '1min': 60,
        '5min': 5 * 60,
        '10min': 10 * 60,
        '15min': 15 * 60,
        '30min': 30 * 60,
        '1h': 60 * 60,
        '1d': 24 * 60 * 60
    }
    if interval not in interval_mapping:
        raise ValueError ("不支持的时间间隔。请使用: " + ", ".join (interval_mapping.keys ()))
    resolution = interval_mapping[interval]
    safety_factor = int (8 * 24 * 60 * 60)  # 确保能够获取足够数据
    duration = interval_seconds[interval] * num_bars + safety_factor

    end_time = int (time.time ())
    start_time = end_time - duration

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': start_time,
        'to': end_time,
        'extended': True,
        'token': API_KEY
    }

    response = requests.get (url, params=params)
    if response.status_code != 200:
        print ("响应内容：", response.text)
        raise Exception (f"请求失败: {response.status_code}")

    data = response.json ()
    if data.get ('s') != 'ok':
        print ("未获取到数据:", data)
        return pd.DataFrame ()

    df = pd.DataFrame ({
        'time': pd.to_datetime (data['t'], unit='s', utc=True).tz_convert ('America/New_York'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })

    if not df.empty:
        df = df.sort_values (by='time')
        if len (df) > num_bars:
            df = df.tail (num_bars)
            df = df.reset_index (drop=True)
    return df


def patch_historical_data ():
    """
    通过 REST API 获取历史 5 分钟数据，并将其拼接到 WS 聚合数据的前面。
    """
    global aggregated_candles, historical_data_patched
    print ("开始通过 REST API 补充历史数据...")
    rest_df = fetch_data (symbol, interval='5min', num_bars=seq_length + 10)
    if rest_df.empty:
        print ("无法通过 REST API 补充历史数据")
        return
    rest_candles = rest_df.to_dict (orient='records')
    if aggregated_candles:
        ws_start_time_local = aggregated_candles[0]['time']
        rest_candles = [c for c in rest_candles if c['time'] < ws_start_time_local]
    aggregated_candles[:] = rest_candles + aggregated_candles
    aggregated_candles.sort (key=lambda x: x['time'])
    historical_data_patched = True
    # df = pd.DataFrame (aggregated_candles)
    # print (df)
    print ("历史数据补充完成，共计蜡烛数：", len (aggregated_candles))
    print ("\n")


def display_current_and_prediction (current_row, pred, resolution_minutes=5):
    """
    构造预测信息摘要，并更新预测显示区域（单行）。
    """
    global current_prediction_msg
    current_time = current_row['time']
    # 确保当前时间为纽约时区
    if current_time.tzinfo is None or current_time.tzinfo.utcoffset (current_time) is None:
        current_time = pd.to_datetime (current_time, utc=True).tz_convert ('America/New_York')
    else:
        current_time = current_time.tz_convert ('America/New_York')
    predicted_time = current_time + timedelta (minutes=resolution_minutes)
    pred = pred.flatten ()
    predicted_price = {
        "open": round (pred[0], 4),
        "high": round (pred[1], 4),
        "low": round (pred[2], 4),
        "close": round (pred[3], 4)
    }
    # 构造一行摘要信息
    current_prediction_msg = (f"{current_time.strftime ('%Y-%m-%d %H:%M:%S')} | "
                              f"Current: O:{current_row['open']:.2f} H:{current_row['high']:.2f} "
                              f"L:{current_row['low']:.2f} C:{current_row['close']:.2f} -> "
                              f"Pred {predicted_time.strftime ('%H:%M:%S')}: "
                              f"O:{predicted_price['open']:.2f} H:{predicted_price['high']:.2f} "
                              f"L:{predicted_price['low']:.2f} C:{predicted_price['close']:.2f}")
    update_display ()


def run_prediction_if_possible ():
    """
    当累计完整蜡烛数达到要求时，构建 DataFrame、预测，并展示预测结果。
    同时限制 aggregated_candles 只保留最新的 seq_length 个蜡烛。
    """
    global aggregated_candles
    if len (aggregated_candles) >= seq_length:
        # 限制缓存数据，只保留最近 seq_length 个蜡烛
        if len (aggregated_candles) > seq_length:
            aggregated_candles[:] = aggregated_candles[-seq_length:]
        df = pd.DataFrame (aggregated_candles)
        df = df.sort_values (by="time").reset_index (drop=True)
        if new_column_order:
            df = df[new_column_order]
        df = add_factors (df)
        clean_outliers (df, columns=['open', 'high', 'low', 'close'], z_thresh=10)

        x, target_indices = create_prediction_sequence (df, seq_length=seq_length,
                                                        scaler=scaler,
                                                        target_col=['open', 'high', 'low', 'close'])
        preds = model.predict_model (x, scaler=scaler, target_indices=target_indices,
                                     bias_corrector=bias_corrector)
        current_row = df.iloc[-1]
        display_current_and_prediction (current_row, preds, resolution_minutes=5)


def aggregate_trade (trade):
    """
    将单笔交易聚合到当前5分钟蜡烛中。如果检测到进入新时间区间，
    并且当前蜡烛已经足够完整（即 trade_time >= 当前蜡烛开始时间+5分钟），
    则将该蜡烛认为完整并保存到 aggregated_candles 中，
    然后启动新的蜡烛，并尝试预测。
    """
    global current_candle, aggregated_candles, historical_data_patched, ws_start_time

    price = trade['p']
    trade_volume = trade['v']
    # 使用 timezone-aware 的 fromtimestamp，将毫秒时间戳转换为纽约时区时间
    trade_time = datetime.fromtimestamp (trade['t'] / 1000, tz=pytz.UTC)
    trade_time = trade_time.astimezone (pytz.timezone ("America/New_York"))
    # 向下取整到最近的 5 分钟整点
    floored_minute = trade_time.minute - (trade_time.minute % 5)
    candle_time = trade_time.replace (minute=floored_minute, second=0, microsecond=0)

    if current_candle is None:
        current_candle = {
            "time": candle_time,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": trade_volume
        }
    else:
        # 更新当前蜡烛（同一时间区间内）
        if candle_time == current_candle["time"]:
            current_candle["high"] = max (current_candle["high"], price)
            current_candle["low"] = min (current_candle["low"], price)
            current_candle["close"] = price
            current_candle["volume"] += trade_volume
        # 进入新的时间区间，判断当前蜡烛是否足够完整
        elif candle_time > current_candle["time"]:
            # 修改后的判断条件：交易时间超过当前蜡烛开始时间+5分钟
            if current_candle["time"] >= ws_start_time and trade_time >= current_candle["time"] + timedelta (minutes=5):
                aggregated_candles.append (current_candle)
            else:
                # 不足5分钟则不保存当前蜡烛
                pass

            # 开启新的蜡烛
            current_candle = {
                "time": candle_time,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": trade_volume
            }

            # 第一次新蜡烛产生时补充历史数据（仅一次）
            if not historical_data_patched and len (aggregated_candles) > 0:
                patch_historical_data ()

            # 每次新蜡烛产生后尝试预测
            run_prediction_if_possible ()

    return current_candle["time"].replace (microsecond=0).strftime ("%Y-%m-%d %H:%M:%S")


def on_message (ws, message):
    """
    WebSocket 接收消息回调：处理交易数据，更新状态，并调用聚合函数。
    """
    global current_status_msg
    data = json.loads (message)
    if data.get ('type') == 'trade':
        trades = data.get ('data', [])
        for trade in trades:
            if trade.get ('s') == symbol:  # 只处理指定股票
                current_candle_time = aggregate_trade (trade)
                current_status_msg = (f"[{datetime.now ().strftime ('%H:%M:%S')}] "
                                      f"Aggregating for {current_candle_time} | "
                                      f"Trade: price={trade.get ('p')}, volume={trade.get ('v')} | "
                                      f"Aggregated candles: {len (aggregated_candles)}")
                update_display ()


def on_open (ws):
    """
    WebSocket 连接建立后订阅指定股票的交易数据，并初始化显示区域。
    """
    global ws_start_time
    ws_start_time = datetime.now (pytz.timezone ("America/New_York"))
    subscribe_message = {"type": "subscribe", "symbol": symbol}
    ws.send (json.dumps (subscribe_message))
    print (f"Subscribed to {symbol}. WS 启动时间: {ws_start_time.strftime ('%Y-%m-%d %H:%M:%S')}")
    # 预留两行用于实时更新（Prediction 与 Status）
    print ("Prediction: ")
    print ("Status    : ")


def on_close (ws, close_status_code, close_msg):
    print ("WebSocket Closed")


def start_ws ():
    """
    启动 WebSocket 客户端，连接至 Finhub 的 WebSocket API。
    """
    token = API_KEY
    socket_url = f"wss://ws.finnhub.io?token={token}"
    ws = websocket.WebSocketApp (socket_url,
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_close=on_close)
    ws.run_forever ()


if __name__ == "__main__":
    start_ws ()
