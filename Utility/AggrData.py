"""
实时股票交易数据采集与聚合脚本
- 先同步回填最近两天数据，再订阅 WebSocket 实时数据
- 按分钟统一批量向量化统计，支持多只标的
- 写入前对每批次结果按 timestamp 排序，保证有序
"""

import json
import threading
import websocket
import time
import csv
import os
import pytz
import logging
import signal
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd

# UTIL & PREPROCESSING
from Util import safeLoadCSV
from DataRequest import fetch_tick_data_last_day
from Preprocessing import aggregate_tick_to_minute  # 针对单只标的的函数

# ---------------------------------------- 配置参数 ----------------------------------------
SYMBOLS = ["AAPL", "GOOGL", "TSLA"]
INTERVAL_KEY = "1min"
FINNHUB_TOKEN = "your api"
WS_URL = f"wss://ws.finnhub.io?token={FINNHUB_TOKEN}"
DATA_DIR = "./realtimeData/raw"
LOG_DIR = "./realtimeData/logs"

# ---------------------------------------- 日志与目录初始化 ----------------------------------------
os.makedirs (DATA_DIR, exist_ok=True)
os.makedirs (LOG_DIR, exist_ok=True)
logging.basicConfig (
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler (f"{LOG_DIR}/stock_data.log", encoding='utf-8'),
        logging.StreamHandler ()
    ]
)
logger = logging.getLogger (__name__)

# ---------------------------------------- 全局状态 ----------------------------------------
should_exit = False

# raw_ticks[minute_ts][symbol] = list of {'timestamp','price','volume'}
raw_ticks = defaultdict (lambda: defaultdict (list))
raw_ticks_lock = threading.Lock ()


# ---------------------------------------- 时间处理工具 ----------------------------------------
def get_eastern_time ():
    tz = pytz.timezone ('America/New_York')
    return datetime.now (pytz.UTC).astimezone (tz)


def get_next_minute_time ():
    return (get_eastern_time () + timedelta (minutes=1)).replace (second=0, microsecond=0)


def get_previous_minute_time ():
    return (get_eastern_time () - timedelta (minutes=1)).replace (second=0, microsecond=0)


# ---------------------------------------- 回填函数（同步） ----------------------------------------
def backfill_initial_interval (symbols, api_key, write_callback):
    logger.info ("【Backfill】开始回填初始两天数据…")
    for symbol in symbols:
        try:
            df = fetch_tick_data_last_day (symbol=symbol, api_key=api_key)
            df_minute = aggregate_tick_to_minute (df)
            if df_minute.empty:
                logger.warning (f"【Backfill】{symbol}：未获取到任何数据")
                continue
            # 局部排序，保证按 timestamp 升序
            df_minute = df_minute.sort_values ('timestamp')
            write_callback (symbol, df_minute)
            logger.info (f"【Backfill】{symbol} 回填完毕，共 {len (df_minute)} 条")
        except Exception as e:
            logger.error (f"【Backfill】{symbol} 出错: {e}")
    logger.info ("【Backfill】完成。")


def write_callback (symbol, df_minute):
    """将回填结果写入各自 CSVc"""
    filename = f"{DATA_DIR}/{symbol}_{INTERVAL_KEY}.csv"
    new_file = not os.path.isfile (filename)
    with open (filename, 'a', newline='') as f:
        fieldnames = list (df_minute.columns)
        writer = csv.DictWriter (f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader ()
        for _, row in df_minute.iterrows ():
            row_dict = row.to_dict ()
            # 格式化 timestamp
            row_dict['timestamp'] = row['timestamp'].strftime ('%Y-%m-%d %H:%M:00')
            writer.writerow (row_dict)


# ---------------------------------------- WebSocket 回调 ----------------------------------------
def on_message (ws, message):
    msg = json.loads (message)
    if msg.get ('type') != 'trade':
        return
    for trade in msg['data']:
        sym, price, vol, ts = trade['s'], trade['p'], trade['v'], trade['t']
        if sym not in SYMBOLS:
            continue
        # 转换为美东时间，并 floor 到分钟
        dt = datetime.fromtimestamp (ts / 1000, pytz.UTC) \
            .astimezone (pytz.timezone ('America/New_York'))
        minute_ts = dt.replace (second=0, microsecond=0) \
            .strftime ('%Y-%m-%d %H:%M:00')

        with raw_ticks_lock:
            raw_ticks[minute_ts][sym].append ({
                'timestamp': dt,
                'price': price,
                'volume': vol
            })


def on_error (ws, error):
    logger.error (f"WS 错误: {error}")


def on_close (ws, code, msg):
    global should_exit
    logger.error (f"WS 关闭: {code} {msg}")
    if not should_exit:
        time.sleep (5)
        start_websocket ()


def on_open (ws):
    def run ():
        for sym in SYMBOLS:
            ws.send (json.dumps ({'type': 'subscribe', 'symbol': sym}))
            logger.info (f"已订阅 {sym}")
            time.sleep (0.1)

    threading.Thread (target=run, daemon=True).start ()


def start_websocket ():
    if should_exit:
        return
    ws_app = websocket.WebSocketApp (
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws_app.run_forever (ping_interval=30)


# ---------------------------------------- 分钟批量聚合 ----------------------------------------
def aggregate_minute_data ():
    prev_min = get_previous_minute_time ().strftime ('%Y-%m-%d %H:%M:00')

    # 确保 raw_ticks 中即使没有这一分钟也不会报 key 错误
    minute_dict = raw_ticks.get (prev_min, {})

    for sym in SYMBOLS:
        ticks = minute_dict.get (sym, [])

        if ticks:
            # 正常聚合流程
            df_ticks = pd.DataFrame (ticks)
            df_minute = aggregate_tick_to_minute (df_ticks)
            df_minute = df_minute.sort_values ('timestamp')

            # 写入 CSV
            filename = f"{DATA_DIR}/{sym}_{INTERVAL_KEY}.csv"
            new_file = not os.path.isfile (filename)
            with open (filename, 'a', newline='') as f:
                writer = csv.DictWriter (f, fieldnames=list (df_minute.columns))
                if new_file:
                    writer.writeheader ()
                for _, row in df_minute.iterrows ():
                    row_dict = row.to_dict ()
                    row_dict['timestamp'] = row['timestamp'].strftime ('%Y-%m-%d %H:%M:00')
                    writer.writerow (row_dict)

            logger.info (f"写入 {sym}@{prev_min} {len(ticks)} ticks")

        else:
            # 没有任何 tick，记录且跳过
            logger.info (f"无数据写入 {sym}@{prev_min}")

    # 清理旧分钟数据
    if prev_min in raw_ticks:
        with raw_ticks_lock:
            del raw_ticks[prev_min]


# ---------------------------------------- 信号处理 & 定时器 ----------------------------------------
def signal_handler (sig, frame):
    global should_exit
    logger.info ("收到退出信号，准备关闭…")
    should_exit = True
    sys.exit (0)


def start_minute_timer ():
    def runner ():
        # 等待下一个整分钟
        nxt = get_next_minute_time ()
        time.sleep ((nxt - get_eastern_time ()).total_seconds ())
        while not should_exit:
            aggregate_minute_data ()
            nxt = get_next_minute_time ()
            time.sleep ((nxt - get_eastern_time ()).total_seconds ())

    threading.Thread (target=runner, daemon=True).start ()


# ---------------------------------------- 主入口 ----------------------------------------
def main ():
    signal.signal (signal.SIGINT, signal_handler)
    signal.signal (signal.SIGTERM, signal_handler)

    # 1) 同步回填两天数据
    backfill_initial_interval (SYMBOLS, FINNHUB_TOKEN, write_callback)

    # 2) 启动分钟级批量聚合定时器
    start_minute_timer ()

    # 3) 启动 WebSocket 实时接收
    start_websocket ()


if __name__ == '__main__':
    main ()
