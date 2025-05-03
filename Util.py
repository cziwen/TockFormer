import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import requests
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, ParameterGrid
from datetime import datetime, timedelta

from torch.utils.data import TensorDataset

def grid_search (model_class, init_args, dataset, param_grid, cv=5,
                 scaler=None, target_indices=None):
    """
    对架构超参数进行网格搜索，使用交叉验证选择最优配置。

    参数:
      - model_class: TimeSeriesTransformer 类
      - init_args: dict, 除了可调超参外的固定初始化参数，例如 {'input_dim':49,'output_dim':4,'seq_length':100,'dropout':0.1}
      - dataset: 完整数据集, TensorDataset(input, label)
      - param_grid: dict, key 为模型初始化参数名（如 'model_dim','num_heads','num_layers'），value 为列表
      - cv: 折数
      - scaler, target_indices: 同前

    返回:
      - best_params: 最佳参数组合（仅包含可调参数）
      - best_score: 对应的平均 MSE
    """
    best_score = float ('inf')
    best_params = None
    for params in ParameterGrid (param_grid):
        print (f"Testing architecture params: {params}")
        # 合并固定参数与可调参数，重新实例化模型
        model_kwargs = {**init_args, **params}
        model = model_class (**model_kwargs)
        # 使用默认训练配置进行 CV
        cv_results = model.cross_validate (
            dataset,
            k=cv,
            scaler=scaler,
            target_indices=target_indices
        )
        # 计算平均 MSE
        mean_mse = np.mean ([np.mean (res['mse']) for res in cv_results])
        mean_r2 = np.mean ([np.mean (res['r2']) for res in cv_results])
        print (f" Avg CV MSE: {mean_mse:.6f}, Avg R2: {mean_r2} \n")
        if mean_mse < best_score:
            best_score = mean_mse
            best_params = params
    return best_params, best_score

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


def sample_dataset (dataset: TensorDataset,
                    fraction: float = 0.1,
                    seed: int = 42) -> TensorDataset:
    """
    从一个 TensorDataset 中随机抽取 fraction 比例的样本，返回新的 TensorDataset。

    参数：
      - dataset: 原始 TensorDataset，内部是 (inputs, targets) 或更多 tensors
      - fraction: 子集样本占原集的比例，0 < fraction <= 1
      - seed: 随机种子，保证可复现

    返回：
      - new_ds: 新的 TensorDataset，其中每个 tensor 都只保留抽样的那些索引
    """
    total = len (dataset)
    subset_size = int (total * fraction)
    if subset_size < 1:
        raise ValueError (f"fraction={fraction} 太小，subset_size={subset_size}")

    # 生成随机索引
    rng = np.random.RandomState (seed)
    idx = rng.choice (total, size=subset_size, replace=False)
    idx_tensor = torch.as_tensor (idx, dtype=torch.long)

    # dataset.tensors 是一个 tuple，可能是 (X, y) 或更多
    sampled_tensors = tuple (tensor[idx_tensor] for tensor in dataset.tensors)
    return TensorDataset (*sampled_tensors)


def safeLoadCSV (df):
    # 检查data前几个数据是否有nan，如果有，则舍弃前n行。
    if df.isna ().any ().any ():  # 检查是否有 NaN
        first_valid_index = df.dropna ().index[0]  # 找到第一个非 NaN 数据的索引
        df = df.loc[first_valid_index:]  # 丢弃 NaN 之前的数据
        df = df.dropna ().reset_index (drop=True)  # 重新索引，确保连续性

    return df

def safe_inverse_transform (preds, scaler, target_indices):
    """
    仅对目标列进行逆缩放

    参数：
        - preds: 预测值数组(scaled)
        - scaler: 用于逆缩放的 scaler
        - target_indices: 目标列索引
    """
    preds_inv = preds.copy ()
    for i, col_idx in enumerate (target_indices):
        preds_inv[:, i] = preds[:, i] * scaler.data_range_[col_idx] + scaler.data_min_[col_idx]
    return preds_inv


def fetch_stock_data_finnhub_paginated(
    symbol,
    start_date,
    end_date,
    interval='1',
    token='YOUR_API_KEY',
    chunk_days=7,
    verbose=False
):
    """
    分段拉取分钟/小时级别数据（1min~60min），自动拼接为完整 DataFrame。

    参数：
      - symbol: 股票代码，如 'AAPL'
      - start_date, end_date: 'YYYY-MM-DD' 字符串
      - interval: '1', '5', '15', '30', '60'（单位：分钟）
      - token: Finnhub API token
      - chunk_days: 每次请求的时间范围（天数），建议 5~15 天
      - verbose: 是否输出每个分段请求的信息

    返回：
      - pandas.DataFrame（时间顺序排列）
    """

    resolution_map = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60'}
    if interval not in resolution_map:
        raise ValueError(f"Unsupported interval: {interval}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    url = 'https://finnhub.io/api/v1/stock/candle'

    all_chunks = []

    current_dt = start_dt
    while current_dt < end_dt:
        next_dt = min(current_dt + timedelta(days=chunk_days), end_dt)

        from_ts = int(time.mktime(current_dt.timetuple()))
        to_ts = int(time.mktime(next_dt.timetuple()))

        params = {
            'symbol': symbol,
            'resolution': resolution_map[interval],
            'from': from_ts,
            'to': to_ts,
            'token': token
        }

        r = requests.get(url, params=params)
        data = r.json()

        if verbose:
            print(f"Fetching {current_dt.date()} to {next_dt.date()}... Status: {data.get('s')}")

        if data.get('s') != 'ok':
            current_dt = next_dt
            continue

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        all_chunks.append(df)

        current_dt = next_dt
        time.sleep(0.3)  # 避免触发速率限制

    if not all_chunks:
        raise Exception("No data retrieved. Try adjusting time window or check token/symbol.")

    full_df = pd.concat(all_chunks).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return full_df
