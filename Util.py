import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_length, target_cols=None, scaler=None, scale=True):
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
    feature_columns = data.columns.tolist()[1:]  # 排除第一列时间戳
    df_copy = data.copy()
    df_copy[df_copy.columns[1:]] = df_copy[df_copy.columns[1:]].astype(np.float32)

    # ======== 仅在 scale=True 时执行缩放 ========
    if scale:
        print("数据被缩放")
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df_copy.iloc[:, 1:])  # 只 fit 特征列
        df_copy.iloc[:, 1:] = scaler.transform(df_copy.iloc[:, 1:]).astype(np.float32)
    else:
        print("数据不缩放")
        scaler = None  # 不缩放时，不返回 scaler
    # ==================================================

    data_array = df_copy[feature_columns].values

    if target_cols is None:
        target_indices = list(range(len(feature_columns)))
    elif isinstance(target_cols, str):
        target_indices = [df_copy.columns.get_loc(target_cols) - 1]  # -1 因为去掉了时间列
    elif isinstance(target_cols, list):
        target_indices = [df_copy.columns.get_loc(col) - 1 for col in target_cols]
    else:
        raise ValueError("target_cols 参数必须为 None, str 或 list")


    X, y = [], []
    for i in range(len(data_array) - seq_length):
        X.append(data_array[i: i + seq_length])
        y.append(data_array[i + seq_length][target_indices])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

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
    colors = plt.cm.tab10 (np.linspace (0, 1, len (accuracy_curve_dict)))

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
