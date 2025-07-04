import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, ParameterGrid
from datetime import datetime, timedelta

# 平行计算
from concurrent.futures import ProcessPoolExecutor, as_completed


from torch.utils.data import TensorDataset


def split_time_series (df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs (train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    n = len (df)
    train_end = int (n * train_ratio)
    val_end = train_end + int (n * val_ratio)

    df_train = df.iloc[:train_end].reset_index (drop=True)
    df_val = df.iloc[train_end:val_end].reset_index (drop=True)
    df_test = df.iloc[val_end:].reset_index (drop=True)

    return df_train, df_val, df_test


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




def _compute_metric_for_series (
        name: str,
        ic_global: float,
        series_rank: pd.Series,
        rank_ret: pd.Series,
        window: int | None
) -> dict:
    """
    Compute global IC and (if window>1) sliding-window IC/IR for one series.
    """
    rec = {'factor': name, 'ic_global': float (ic_global)}
    if window and window > 1:
        # rolling Pearson on ranks == Spearman
        ic_sw = series_rank.rolling (window).corr (rank_ret)
        mu = ic_sw.mean (skipna=True)
        sigma = ic_sw.std (ddof=0, skipna=True)
        rec.update ({
            'ic_sw_mean': float (mu),
            'ic_sw_std': float (sigma),
            'ir_sw': float (mu / sigma) if sigma else np.nan,
        })
    return rec


def evaluate_time_series_ic (
        df: pd.DataFrame,
        factors: list[str],
        target_col: str,
        forward: int = 1,
        date_col: str = "timestamp",
        window: int | None = None,
        n_jobs: int | None = None
) -> pd.DataFrame:
    """
    多进程并行的单标的时间序列因子 IC & IR（带 Composite）计算

    Args:
      df: 原始数据，需含 [date_col, target_col] + factors。
      factors: 因子名列表。
      target_col: 价格列（如 'close'）。
      forward: 向后看几个 timestamp 的收益。
      date_col: 时间戳列名。
      window: 若指定 (>1)，就做长度为 window 的滑动 Spearman 计算。
      n_jobs: 并行进程数，None 表示使用所有可用 CPU 核心。

    Returns:
      pd.DataFrame: index 为 factor（含 composite），列含
        ic_global, ic_sw_mean, ic_sw_std, ir_sw（若 window>1）。
    """

    # 过滤 factors
    # Drop non-numeric factors like timestamp
    factors = [f for f in factors if pd.api.types.is_numeric_dtype (df[f])]

    # 1) 准备数据
    df = df.copy ()
    df[date_col] = pd.to_datetime (df[date_col])  # ← 添加这行确保时间正确排序
    df = df.sort_values (date_col)
    df['forward_ret'] = df[target_col].shift (-forward) / df[target_col] - 1
    df = df.dropna (subset=factors + ['forward_ret'])

    # 2) Rank-transform factors + forward_ret
    cols = factors + ['forward_ret']
    df_rank = df[cols].rank (method='average')
    rank_ret = df_rank['forward_ret']

    # 3) 全局 IC for each factor
    ic_global: dict = df_rank[factors].corrwith (rank_ret).to_dict ()

    # 4) Composite 因子
    means = df[factors].mean ()
    stds = df[factors].std (ddof=0)
    df['composite'] = ((df[factors] - means) / stds).sum (axis=1)
    comp_rank = df['composite'].rank (method='average')
    ic_global['composite'] = comp_rank.corr (rank_ret)

    # 5) Prepare rank series mapping
    series_ranks = {f: df_rank[f] for f in factors}
    series_ranks['composite'] = comp_rank

    # 6) 并行计算每个因子的指标
    records = []
    with ProcessPoolExecutor (max_workers=n_jobs) as executor:
        futures = {
            executor.submit (
                _compute_metric_for_series,
                name,
                ic_global[name],
                series_ranks[name],
                rank_ret,
                window
            ): name
            for name in series_ranks
        }
        for fut in as_completed (futures):
            records.append (fut.result ())

    # 7) 构造结果 DataFrame
    result = pd.DataFrame (records).set_index ('factor')
    # 保证列顺序
    cols_out = ['ic_global']
    if window and window > 1:
        cols_out += ['ic_sw_mean', 'ic_sw_std', 'ir_sw']
    return result[cols_out]

def clean_and_interpolate(df: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """
    1. 丢弃 DataFrame 开头连续出现任何 NaN 的行（“前几个含有 null 的行”）。
    2. 对剩余部分，每列如果缺失值占比 ≤ tol，就用线性插值填充；如果 > tol，则整列删除。

    Args:
        df: 待清洗的 DataFrame，索引可保留原始时间戳或其它索引。
        tol: 单列允许填充的最大缺失比例（默认为 0.05 即 5%）。

    Returns:
        经过清洗和插值处理后的 DataFrame。
        - 已删除开头连续含 NaN 的行
        - 对每列缺失值比例 ≤ tol 的，已做线性插值
        - 对每列缺失值比例 > tol 的，已删除该列
    """
    # 1. 复制一份，避免修改原始 DataFrame
    df_clean = df.copy()

    # 2. 找到开头连续含有 NaN 的行数
    #    mask[i] = True 表示 df_clean.iloc[i] 这一整行里有至少一个 NaN
    mask = df_clean.isnull().any(axis=1).tolist()
    #    找到第一个 mask[i] == False 的位置 pos，若全是 True，则 pos 为 None
    pos = next((i for i, has_nan in enumerate(mask) if not has_nan), None)

    #    如果所有行都含 NaN，就直接返回空的 DataFrame
    if pos is None:
        return pd.DataFrame(columns=df_clean.columns)

    #    丢弃开头 pos 行
    df_clean = df_clean.iloc[pos:].copy()

    # 3. 对剩余部分按列检查缺失比例并决定插值或删除
    n_rows = len(df_clean)
    for col in list(df_clean.columns):
        null_count = df_clean[col].isnull().sum()
        if null_count == 0:
            # 该列无缺失，跳过
            continue

        frac = null_count / n_rows
        if frac <= tol:
            # 缺失比例在允许范围内，做线性插值
            # 保留原来索引，method='linear'
            df_clean[col] = df_clean[col].interpolate(method='linear')
            # 如果边缘仍有 NaN（如首尾），可以再向前后填补或直接保留 NaN
            # 通常线性插值后，首尾的 NaN 会保留；如果想再用前向/后向填充，可按需添加：
            # df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        else:
            # 缺失比例过高，删除整列
            df_clean.drop(columns=[col], inplace=True)

    return df_clean
