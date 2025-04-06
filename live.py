import time
import requests
import pandas as pd
from CSVModifier import *


def fetch_data (symbol, interval='5min', num_bars=50):
    """
    使用 Finhub REST API 获取指定股票的K线数据。

    参数:
      symbol (str): 股票代码，如 'AAPL'
      interval (str): 时间间隔，如 '5min', '10min', '1h'
      num_bars (int): 从当前时间往回取 num_bars 个时间点的数据，
                      返回最新的 num_bars 条数据（即使当前为非交易时段）。
    返回:
      pd.DataFrame: 包含K线数据的 DataFrame；如果获取失败或无数据则返回空 DataFrame。
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

    # 每个时间间隔对应的秒数
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
    safety_factor = int(8 * 24 * 60 * 60) # 确保能够获取足够的数据, 秒数
    duration = interval_seconds[interval] * num_bars + safety_factor

    end_time = int (time.time ())
    start_time = end_time - duration

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': start_time,
        'to': end_time,
        # 'token': 'cvop3lhr01qihjtq3uvgcv叽里呱啦op3lhr01qihjtq3v00'  # 替换为你的 AP-I K-ey
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
        'time': pd.to_datetime (data['t'], unit='s'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })

    # 按时间升序排序，并取最新的 num_bars 条记录
    if not df.empty:
        df = df.sort_values (by='time')
        if len (df) > num_bars:
            df = df.tail (num_bars)
            df = df.reset_index (drop=True)

    return df


def main_workflow (symbol, num_bars_prediction=50):
    """
    工作流：获取多时间周期的预测数据和1分钟数据，
    利用1分钟数据计算5min、30min和1h周期内的波动率（std）。

    num_bars_prediction: 跟模型的 sequential length 一个意思。
    """
    # 获取用于预测的不同时间周期数据
    print ("获取5分钟数据...")
    df_5min = fetch_data (symbol, interval='5min', num_bars=num_bars_prediction)

    print ("获取30分钟数据...")
    df_30min = fetch_data (symbol, interval='30min', num_bars=num_bars_prediction)

    print ("获取1小时数据...")
    df_1h = fetch_data (symbol, interval='1h', num_bars=num_bars_prediction)

    # 获取1分钟数据
    num_bars_1min = num_bars_prediction * 60
    print ("获取1分钟数据...")
    df_1min = fetch_data (symbol, interval='1min', num_bars=num_bars_1min)

    if df_1min.empty:
        raise Exception ("1分钟数据为空，工作流停止")

    # 计算波动率
    vol_5min = vol_30min = vol_1h = None
    if df_5min.shape[0] >= num_bars_prediction:
        vol_5min = aggregate_high_freq_to_low (df_1min, '5min', timestamp='time')

    if df_30min.shape[0] >= num_bars_prediction:
        vol_30min = aggregate_high_freq_to_low (df_1min, '30min', timestamp='time')

    if df_1h.shape[0] >= num_bars_prediction:
        vol_1h = aggregate_high_freq_to_low (df_1min, '1h', timestamp='time')

    # 输出
    if df_5min.shape[0] >= num_bars_prediction  and vol_5min is not None:
        df_5min = add_factors (df_5min)  # 计算因子
        df_merged_5min = pd.merge (df_5min, vol_5min, on='time', how='left')  # join
        clean_outliers (df_merged_5min, columns=['open', 'high', 'low', 'close'], z_thresh=10)

        # 创建序列



    if df_30min.shape[0] >= num_bars_prediction and vol_30min is not None:
        df_30min = add_factors (df_30min)  # 计算因子
        df_merged_30min = pd.merge (df_30min, vol_30min, on='time', how='left')  # join
        clean_outliers (df_merged_30min, columns=['open', 'high', 'low', 'close'], z_thresh=10)


    if df_1h.shape[0] >= num_bars_prediction and vol_1h is not None:
        df_1h = add_factors (df_1h)  # 计算因子
        df_merged_1h = pd.merge (df_1h, vol_1h, on='time', how='left')  # join
        clean_outliers (df_merged_1h, columns=['open', 'high', 'low', 'close'], z_thresh=10)





if __name__ == "__main__":
    symbol = "SPY"  # 替换为你需要的股票代码
    main_workflow (symbol, num_bars_prediction=100)
