import pandas as pd
import requests
import time

from fontTools.misc.plistlib import end_date

# API_K-EY = "cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00"


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
    safety_factor = int (8 * 24 * 60 * 60)  # 确保能够获取足够的数据
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


def fetch_polygon_timeseries (symbol, start_date, end_date, api_key, timespan="minute", multiplier=1):
    """
    自动分页拉取 Polygon.io 的指定标的、指定时间段、指定频率的聚合数据。

    参数:
    - symbol: str, 标的代码，比如 'AAPL'
    - start_date: str, 开始日期，格式 'YYYY-MM-DD'
    - end_date: str, 结束日期，格式 'YYYY-MM-DD'
    - api_key: str, 你的 Polygon API Key
    - timespan: str, 时间粒度 ('minute', 'hour', 'day', etc.)
    - multiplier: int, 例如 5 代表 '5分钟K线'

    返回:
    - pandas.DataFrame
    """

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

    all_data = []
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }

    while True:
        response = requests.get (url, params=params)
        if response.status_code != 200:
            print (f"Error: {response.status_code} {response.text}")
            time.sleep (2)
            continue  # 自动重试

        data = response.json ()
        if "results" not in data or not data["results"]:
            break

        all_data.extend (data["results"])

        # 检查是否还有下一页
        if not data.get ("next_url"):
            break
        else:
            url = data["next_url"]
            params = {"apiKey": api_key}  # next_url已经包含了其他参数，不需要再带别的了

        time.sleep (0.3)  # 防止请求过快被限流

    if not all_data:
        raise ValueError ("No data fetched.")

    df = pd.DataFrame (all_data)
    df['timestamp'] = pd.to_datetime (df['t'], unit='ms')
    df = df.rename (columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'vw': 'vwap',
        'v': 'volume',
        'n': 'num_transactions'
    })


    return df[['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'num_transactions']]


apikey = "oilTTMMexxTBTmjivaMq3R0Y9ZS1BKbK"

df = fetch_polygon_timeseries (symbol="SPY", timespan="minute", multiplier=1, api_key=apikey,
                               start_date="2025-04-01", end_date="2025-04-27")
# df.to_csv("rawdata/SPY_1min_test-april.csv", index=False)