import time
import joblib
import requests
import torch
import pandas as pd
from datetime import datetime, timedelta

from CSVModifier import *
from TransformerModel import TimeSeriesTransformer
from BiasCorrector import BiasCorrector
from Util import *


def fetch_data(symbol, interval='5min', num_bars=50):
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
        raise ValueError("不支持的时间间隔。请使用: " + ", ".join(interval_mapping.keys()))

    resolution = interval_mapping[interval]
    safety_factor = int(8 * 24 * 60 * 60)  # 确保能够获取足够的数据, 秒数
    duration = interval_seconds[interval] * num_bars + safety_factor

    end_time = int(time.time())
    start_time = end_time - duration

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': start_time,
        'to': end_time,
        'extended': True,
        'token': 'cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00'  # 替换为你的 API Key
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("响应内容：", response.text)
        raise Exception(f"请求失败: {response.status_code}")

    data = response.json()
    if data.get('s') != 'ok':
        print("未获取到数据:", data)
        return pd.DataFrame()

    df = pd.DataFrame ({
        'time': pd.to_datetime (data['t'], unit='s', utc=True).tz_convert ('America/New_York'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })

    # 按时间升序排序，并取最新的 num_bars 条记录
    if not df.empty:
        df = df.sort_values(by='time')
        if len(df) > num_bars:
            df = df.tail(num_bars)
            df = df.reset_index(drop=True)

    return df


def display_current_and_prediction (current_row, pred, resolution_minutes=5):
    """
    同时展示当前最新行情和预测的未来价格，打印内容会覆盖原有内容。

    参数：
      - current_row: pd.Series，包含当前最新数据行，必须含有 'time', 'open', 'high', 'low', 'close'
      - pred: numpy array 或 list，形状为 [1, 4]，预测的 open, high, low, close
      - resolution_minutes: int，预测步长的分钟数，默认为 5 分钟
    """

    # 清除终端屏幕（使用 ANSI 转义序列），使新的输出覆盖原有内容
    print ("\033[2J\033[H", end="")  # 或者使用 os.system('cls' if os.name == 'nt' else 'clear')

    # 获取当前时间
    current_time = current_row['time']
    # 如果当前时间没有时区信息，假定其为 UTC 后转换为纽约时区；如果已有时区信息，则直接转换
    if current_time.tzinfo is None or current_time.tzinfo.utcoffset (current_time) is None:
        current_time = pd.to_datetime (current_time, utc=True).tz_convert ('America/New_York')
    else:
        current_time = current_time.tz_convert ('America/New_York')

    # 预测时间为当前时间加上给定的分钟数
    predicted_time = current_time + timedelta (minutes=resolution_minutes)

    # 当前价格信息
    current_price = {
        "open": current_row['open'],
        "high": current_row['high'],
        "low": current_row['low'],
        "close": current_row['close']
    }

    # 将预测结果拉平成一维数组，并构建预测价格信息
    pred = pred.flatten ()
    predicted_price = {
        "open": round (pred[0], 4),
        "high": round (pred[1], 4),
        "low": round (pred[2], 4),
        "close": round (pred[3], 4)
    }

    print ("=" * 40)
    print (f"当前时间（东部）：{current_time.strftime ('%Y-%m-%d %H:%M:%S')}")
    print ("当前价格（USD）：")
    for k, v in current_price.items ():
        print (f"  • {k:<6}: {v:.2f}")
    print ("-" * 40)
    print (f"预测时间（东部）：{predicted_time.strftime ('%Y-%m-%d %H:%M:%S')}")
    print ("预测价格（USD）：")
    for k, v in predicted_price.items ():
        print (f"  • {k:<6}: {v:.2f}")
    print ("=" * 40)

def main_workflow(symbol, num_bars_prediction=50, scaler_5min=None, model_5min=None,
                  bias_corrector_5min=None, new_colum_order=None):
    """
    工作流：获取数据、计算因子、生成预测序列、并进行预测。
    修改后在预测完成后返回最新合并后的 DataFrame 和预测结果。

    参数：
      - symbol: 股票代码
      - num_bars_prediction: 序列长度，与模型输入长度一致
      - scaler_5min, model_5min, bias_corrector_5min: 分别为 scaler, 模型和 BiasCorrector 对象
      - new_colum_order: 列顺序，如果需要重新排列 DataFrame 列

    返回：
      - df_merged_5min: 用于预测的最新数据（DataFrame）
      - preds_5min: 模型预测结果（numpy 数组，形状 [1, 4]）
    """
    # print("获取5分钟数据...")
    df_5min = fetch_data(symbol, interval='5min', num_bars=num_bars_prediction + 100)

    # print("获取1分钟数据...")
    num_bars_1min = num_bars_prediction * 60
    df_1min = fetch_data(symbol, interval='1min', num_bars=num_bars_1min + 600)

    if df_1min.empty:
        raise Exception("1分钟数据为空，工作流停止")

    vol_5min = None
    if df_5min.shape[0] >= num_bars_prediction:
        vol_5min = aggregate_high_freq_to_low(df_1min, '5min', timestamp='time')

    if df_5min.shape[0] >= num_bars_prediction and vol_5min is not None:
        df_5min = add_factors(df_5min)  # 计算因子
        df_merged_5min = pd.merge(df_5min, vol_5min, on='time', how='left')  # join
        if new_colum_order:
            df_merged_5min = df_merged_5min[new_colum_order]
        clean_outliers(df_merged_5min, columns=['open', 'high', 'low', 'close'], z_thresh=10)

        # 创建预测序列
        x, target_indices = create_prediction_sequence(df_merged_5min, seq_length=num_bars_prediction,
                                                         scaler=scaler_5min, target_col=['open', 'high', 'low', 'close'])

        # 预测
        preds_5min = model_5min.predict_model(x, scaler=scaler_5min, target_indices=target_indices,
                                              bias_corrector=bias_corrector_5min)

        return df_merged_5min, preds_5min

    else:
        print("数据不足以进行预测。")
        return None, None


def run_periodically(interval_minutes=5):
    """
    每隔固定分钟执行一次 main_workflow，并打印当前价格和预测价格。
    """
    symbol = "SPY"
    seq_length = 32
    model_path = "models/transformer_regression_5min_lam.pth"
    scaler_path = "models/scaler_regression_5min_lam.pkl"
    bias_corrector_path = "models/corrector_regression_5min_lam.pkl"

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(input_dim=49, model_dim=64, num_heads=4, num_layers=2,
                                  dropout=0.2, seq_length=seq_length, output_dim=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    bias_corrector = BiasCorrector.load(bias_corrector_path)
    scaler = joblib.load(scaler_path)

    while True:
        print("\n" + "=" * 60)
        print(f"开始运行 main_workflow at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        df_merged, preds = main_workflow(symbol, num_bars_prediction=seq_length, scaler_5min=scaler,
                                         model_5min=model, bias_corrector_5min=bias_corrector,
                                         new_colum_order=new_column_order)
        if df_merged is not None and preds is not None:
            # 取最新一行作为当前价格数据
            current_row = df_merged.iloc[-1]
            display_current_and_prediction(current_row, preds, resolution_minutes=5)
        else:
            print("未能获取足够数据。")
        print(f"等待 {interval_minutes} 分钟后继续...\n")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    run_periodically(interval_minutes=0.05)