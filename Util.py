from typing import Optional

import pandas as pd
import numpy as np
import pytz
import torch
import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def create_sequences (data, seq_length, target_cols=None, scaler=None, scale=True):
    """
    将 DataFrame 数据转换为模型的序列样本。时间会默认排除在外

    参数：
      - data: pd.DataFrame，包含特征列（数据假设已按时间顺序排列）
      - seq_length: 每个序列的长度
      - target_cols: 指定用于作为目标的列名，默认为 None，表示使用所有列
      - scaler: 可选 MinMaxScaler，用于共享训练数据的缩放器
      - scale: 是否进行归一化（默认 True）

    返回：
      - X_Tensor (samples, seq_length, F)
      - y_Tensor (samples, T)
      - scaler（MinMaxScaler 实例或 None）
      - target_indices（用于 inverse_transform）
    """
    feature_columns = data.columns.tolist ()[1:]  # 排除第一列时间戳
    df_copy = data.copy ()
    df_copy[df_copy.columns[1:]] = df_copy[df_copy.columns[1:]].astype (np.float32)

    # ======== 仅在 scale=True 时执行缩放 ========
    if scale:
        print ("数据被缩放")
        if scaler is None:
            scaler = MinMaxScaler (feature_range=(0, 1))
            scaler.fit (df_copy.iloc[:, 1:])  # 只 fit 特征列
        df_copy.iloc[:, 1:] = scaler.transform (df_copy.iloc[:, 1:]).astype (np.float32)
    else:
        print ("数据不缩放")
        scaler = None  # 不缩放时，不返回 scaler
    # ==================================================

    data_array = df_copy[feature_columns].values

    if target_cols is None:
        target_indices = list (range (len (feature_columns)))
    elif isinstance (target_cols, str):
        target_indices = [df_copy.columns.get_loc (target_cols) - 1]  # -1 因为去掉了时间列
    elif isinstance (target_cols, list):
        target_indices = [df_copy.columns.get_loc (col) - 1 for col in target_cols]
    else:
        raise ValueError ("target_cols 参数必须为 None, str 或 list")

    X, y = [], []
    for i in range (len (data_array) - seq_length):
        X.append (data_array[i: i + seq_length])
        y.append (data_array[i + seq_length][target_indices])

    X_tensor = torch.tensor (np.array (X), dtype=torch.float32)
    y_tensor = torch.tensor (np.array (y), dtype=torch.float32)

    return X_tensor, y_tensor, scaler, target_indices


def create_prediction_sequence (data, seq_length, scaler, target_col):
    """
    将 DataFrame 数据转换为预测序列样本，仅返回 X_tensor 和 target_indices。
    假设第一列为时间戳，其余列为特征，使用传入的 scaler 对特征进行归一化。

    参数：
      - data: pd.DataFrame，已按时间顺序排列，第一列为时间戳
      - seq_length: 输入序列长度
      - scaler: 已 fit 的 MinMaxScaler，用于归一化
      - target_col: 单个列名或列名列表，用于后续逆缩放预测值

    返回：
      - X_tensor: torch.Tensor，形状为 [1, seq_length, F]
      - target_indices: List[int]，在 scaler 中对应的目标列索引
    """
    # 提取特征列名（排除时间戳）
    feature_columns = data.columns.tolist ()[1:]
    df_copy = data.copy ()
    df_copy[df_copy.columns[1:]] = df_copy[df_copy.columns[1:]].astype (np.float32)

    # 执行归一化
    df_copy.iloc[:, 1:] = scaler.transform (df_copy.iloc[:, 1:]).astype (np.float32)
    data_array = df_copy[feature_columns].values

    if len (data_array) < seq_length:
        raise ValueError ("数据长度小于序列长度")

    # 构造输入序列
    X = data_array[-seq_length:]
    X_tensor = torch.tensor (X, dtype=torch.float32).unsqueeze (0)  # [1, seq_length, F]

    # 获取 target_indices（在 scaler 特征中的列位置）
    if isinstance (target_col, str):
        target_indices = [df_copy.columns.get_loc (target_col) - 1]  # -1 是因为排除了时间戳
    elif isinstance (target_col, list):
        target_indices = [df_copy.columns.get_loc (col) - 1 for col in target_col]
    else:
        raise ValueError ("target_col 必须是字符串或字符串列表")

    return X_tensor, target_indices


def plot_metric (values, y_label='Value', title='Training Metric', color='blue', show=True):
    """
    绘制折线图

    参数：
      - values: 数值列表或数组（y 轴数据），例如训练损失或准确率。列表中第 0 个值对应第 1 个 epoch，以此类推。
      - y_label: y 轴标签，默认为 'Value'
      - title: 图像标题，默认为 'Training Metric'
      - color: 线条颜色，默认为 'blue'
      - show: 是否显示图像，默认为 True

    示例：
    >>> loss_values = [0.9, 0.8, 0.7, 0.65, 0.6]
    >>> plot_metric(loss_values, y_label='Loss', title='Training Loss over Epochs', color='red')
    """
    epochs = range (1, len (values) + 1)  # 构造 x 轴，从 1 到 len(values)
    plt.figure (figsize=(8, 6))
    plt.plot (epochs, values, marker='o', linestyle='-', color=color)
    plt.xlabel ('Num Epoch')
    plt.ylabel (y_label)
    plt.title (title)
    plt.grid (True)

    if show:
        plt.show ()


def plot_multiple_curves (accuracy_curve_dict, x_label='period', y_label='Value', title='Training Metric'):
    """
    绘制多个曲线，并叠加在同一个图上。

    参数:
        accuracy_curve_dict (dict):
            - key: 代表不同实验或模型的标签 (str)
            - value: 对应的 accuracy 曲线数据 (list 或 np.array)，长度等于训练 epoch 数
    """
    plt.figure (figsize=(8, 5))

    # 生成颜色序列
    colors = plt.cm.tab20 (np.linspace (0, 1, len (accuracy_curve_dict)))

    for i, (label, acc_curve) in enumerate (accuracy_curve_dict.items ()):
        plt.plot (range (1, len (acc_curve) + 1),
                  acc_curve,
                  # marker='o',
                  # markersize=8,  # 增大标记大小
                  linestyle='-',
                  linewidth=1,  # 增加线宽
                  label=label,
                  color=colors[i % 10])

    plt.xlabel (x_label)
    plt.ylabel (y_label)
    plt.title (title)
    plt.legend ()
    plt.grid (True)
    plt.show ()


def safeLoadCSV (df):
    # 检查data前几个数据是否有nan，如果有，则舍弃前n行。
    if df.isna ().any ().any ():  # 检查是否有 NaN
        first_valid_index = df.dropna ().index[0]  # 找到第一个非 NaN 数据的索引
        df = df.loc[first_valid_index:]  # 丢弃 NaN 之前的数据
        df = df.dropna ().reset_index (drop=True)  # 重新索引，确保连续性

    return df


def display_prediction (pred, base_time=None, resolution_minutes=5):
    """
    美观打印模型预测的 OHLC 价格。

    参数：
      - pred: numpy array 或 list，形状为 [1, 4]，包含预测的 open, high, low, close
      - base_time: datetime，可选，预测基准时间，若为 None 则使用当前时间
      - resolution_minutes: int，预测步长的分钟数，默认为 5 分钟

    输出：
      - 美观的预测信息打印
    """
    # 处理时间
    if base_time is None:
        base_time = datetime.now ()
    prediction_time = base_time + timedelta (minutes=resolution_minutes)

    # 处理预测值
    pred = pred.flatten ()
    price_dict = {
        "open": round (pred[0], 4),
        "high": round (pred[1], 4),
        "low": round (pred[2], 4),
        "close": round (pred[3], 4),
    }

    # 打印
    print ("=" * 40)
    print (f"📅 预测时间：{prediction_time.strftime ('%Y-%m-%d %H:%M:%S')}")
    print ("📈 预测价格（单位：USD）：")
    for k, v in price_dict.items ():
        print (f"  • {k:<6}: {v:.2f}")
    print ("=" * 40)


def fetch_latest_agg_data (
        ticker: str,
        timespan: str = "second",
        limit: int = 10,
        api_key: str = "your_api_key",
        delayed: bool = False
) -> Optional[pd.DataFrame]:
    """
    获取最新 N 条聚合数据，支持 Free Plan（15分钟延迟数据）。

    参数:
        ticker: 股票代码
        timespan: 聚合周期（second, minute, hour, day）
        limit: 返回条数
        api_key: Polygon API Key
        delayed: 如果为 True，则使用美国东部时间，并回退 15 分钟

    返回:
        聚合数据 DataFrame，包含 timestamp, open, high, low, close, volume, vwap
    """
    eastern = pytz.timezone ("America/New_York")
    now_et = datetime.now (eastern)

    if delayed:
        to_time = now_et - timedelta (minutes=15)
        print (f"⚠ 使用延迟数据模式，to_time（纽约时间）= {to_time}")
    else:
        to_time = now_et

    # 计算 from_time
    if timespan == "second":
        from_time = to_time - timedelta (seconds=limit)
    elif timespan == "minute":
        from_time = to_time - timedelta (minutes=limit)
    elif timespan == "hour":
        from_time = to_time - timedelta (hours=limit)
    elif timespan == "day":
        from_time = to_time - timedelta (days=limit)
    else:
        raise ValueError (f"Unsupported timespan: {timespan}")

    # 转换为 ISO 格式（UTC 时间，Polygon 接收 ISO8601）
    from_utc = from_time.astimezone (pytz.utc).strftime ("%Y-%m-%dT%H:%M:%S")
    to_utc = to_time.astimezone (pytz.utc).strftime ("%Y-%m-%dT%H:%M:%S")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_utc}/{to_utc}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key
    }

    response = requests.get (url, params=params)
    if response.status_code != 200:
        print (f"请求失败: {response.status_code} - {response.text}")
        return None

    results = response.json ().get ("results", [])
    if not results:
        print ("未获取到聚合数据")
        return None

    df = pd.DataFrame (results)
    df["timestamp"] = pd.to_datetime (df["t"], unit="ms")
    df.rename (columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap"
    }, inplace=True)

    return df[["timestamp", "open", "high", "low", "close", "volume", "vwap"]]


symbol = 'SPY'
API_KEY = "oilTTMMexxTBTmjivaMq3R0Y9ZS1BKbK"

df_min_past = fetch_latest_agg_data (ticker=symbol, timespan="minute", limit=32,
                                     api_key=API_KEY, delayed=True)
# df_sec_past = fetch_latest_agg_data (ticker=symbol, timespan="second", limit=32,
#                                      api_key=API_KEY, delayed=True)
