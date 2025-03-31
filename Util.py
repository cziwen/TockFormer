import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_sequences (data, seq_length, target_cols=None):
    """
    将 DataFrame 数据转换为模型的序列样本。时间会默认排除在外

    参数：
      - data: pd.DataFrame，包含特征列（数据假设已按时间顺序排列）
      - seq_length: 每个序列的长度
      - target_cols: 指定用于作为目标的列名，默认为 None，表示使用所有列
    返回：
      - X_Tensor (shape: (samples, seq_length, F))，F为特征总数
      - y_Tensor (shape: (samples, T))，T为目标列的数量（若 target_cols 为 None，则 T=F）
    """

    # 获取所有列作为特征
    feature_columns = data.columns.tolist()[1:] # 默认排除第一列，因为是 时间戳

    # scale 数据
    scaler = MinMaxScaler (feature_range=(0, 1))
    df_scaled = data.copy ()
    # 除了第一列 时间戳 之外，其他列直接转换为 float32
    df_scaled[df_scaled.columns[1:]] = df_scaled[df_scaled.columns[1:]].astype (np.float32)
    df_scaled.iloc[:, 1:] = scaler.fit_transform (df_scaled.iloc[:, 1:]).astype (np.float32) # norm


    data_array = df_scaled[feature_columns].values

    # 根据 target_cols 参数，确定目标列的索引
    if target_cols is None:
        target_indices = list (range (len (feature_columns)))
    elif isinstance (target_cols, str):
        target_indices = [df_scaled.columns.get_loc (target_cols)]
    elif isinstance (target_cols, list):
        target_indices = [df_scaled.columns.get_loc (col) for col in target_cols]
    else:
        raise ValueError ("target_cols 参数必须为 None, str 或 list")

    X, y = [], []
    # 使用滑动窗口生成序列和对应目标
    for i in range (len (data_array) - seq_length):
        X.append (data_array[i: i + seq_length])
        y.append (data_array[i + seq_length][target_indices])

    X_tensor = torch.tensor (np.array (X), dtype=torch.float32)
    y_tensor = torch.tensor (np.array (y), dtype=torch.float32)
    return X_tensor, y_tensor


def plot_metric(values, y_label='Value', title='Training Metric', color='blue', show=True):
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
    epochs = range(1, len(values) + 1)  # 构造 x 轴，从 1 到 len(values)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, values, marker='o', linestyle='-', color=color)
    plt.xlabel('Num Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    if show:
        plt.show()



def plot_multiple_curves (accuracy_curve_dict, x_label='period', y_label='Value', title='Training Metric', color='blue', show=True):
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
        plt.plot (range (1, len (acc_curve) + 1), acc_curve, marker='o', linestyle='-', label=label,
                  color=colors[i % 10])

    plt.xlabel (x_label)
    plt.ylabel (y_label)
    plt.title (title)
    plt.legend ()
    plt.grid (True)
    plt.show ()
