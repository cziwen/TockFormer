import pandas as pd
import requests
import time


# API_KE-Y = "cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00"


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


fetch_data('SPY', interval='5min')