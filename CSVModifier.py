import os
import pandas as pd
import numpy as np


def calculate_rsi (prices, period=14):
    delta = np.diff (prices)
    gain = np.maximum (delta, 0)
    loss = -np.minimum (delta, 0)

    avg_gain = np.zeros_like (prices)
    avg_loss = np.zeros_like (prices)
    avg_gain[period] = np.mean (gain[:period])
    avg_loss[period] = np.mean (loss[:period])

    for i in range (period + 1, len (prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def calculate_ema (prices, period):
    alpha = 2 / (period + 1)
    ema = np.zeros_like (prices, dtype=np.float64)
    ema[0] = prices[0]
    for i in range (1, len (prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_macd (prices, short=7, long=14, signal=5):
    short_ema = calculate_ema (prices, short)
    long_ema = calculate_ema (prices, long)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema (macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_stochastic (df, period=14, smooth_k=3, signal=5):
    high = df['high'].rolling (period).max ()
    low = df['low'].rolling (period).min ()
    k_raw = (df['close'] - low) / (high - low) * 100
    k_smooth = k_raw.rolling (smooth_k).mean ()
    d = k_smooth.rolling (signal).mean ()
    return k_smooth, d


def calculate_roc (prices, period=14):
    """
    计算价格的变化率（Rate of Change），单位为百分比
    """
    prices_series = pd.Series (prices)
    roc = (prices_series - prices_series.shift (period)) / prices_series.shift (period) * 100
    return roc.values


def calculate_volume_roc (volume, period=14):
    """
    计算成交量的变化率
    """
    volume_series = pd.Series (volume)
    roc = (volume_series - volume_series.shift (period)) / volume_series.shift (period) * 100
    return roc.values


def calculate_obv (df):
    """
    计算能量潮指标（On Balance Volume, OBV）
    OBV 的计算方法：
      - 如果当日收盘价 > 前一日收盘价，则 OBV 加上当日成交量
      - 如果当日收盘价 < 前一日收盘价，则 OBV 减去当日成交量
      - 如果相等，则 OBV 不变
    """
    obv = [0]
    for i in range (1, len (df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append (obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append (obv[-1] - df['volume'].iloc[i])
        else:
            obv.append (obv[-1])
    return np.array (obv)


def add_factors_to_csv (csv_path, output_dir="factoredData"):
    '''
    添加趋势、动量、成交量三类因子到 CSV 文件中

    趋势指标：
      - 对 open/high/low/close 分别计算简单移动均线 (SMA5, SMA10, SMA20)
      - 以及指数移动均线 (EMA5, EMA10, EMA20)

    动量指标：
      - 计算 RSI、MACD（及其 signal 与 histogram）、Stochastic 指标，以及价格 ROC

    成交量指标：
      - 对 volume 计算 SMA5, SMA10
      - 计算成交量 ROC
      - 计算 OBV（需要 close 与 volume）

    :param csv_path: 原始 CSV 文件路径，要求包含至少 open, high, low, close, volume 列
    :param output_dir: 输出目录，默认为 "factoredData"
    :return: 将新增因子后的 CSV 保存到 output_dir
    '''
    df = pd.read_csv (csv_path)

    # 趋势指标 - 对价格计算移动均线（SMA 和 EMA）
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # SMA
            # df[f'SMA5_{col}'] = df[col].rolling(window=5).mean()
            # df[f'SMA10_{col}'] = df[col].rolling(window=10).mean()
            # df[f'SMA20_{col}'] = df[col].rolling(window=20).mean()
            # EMA（采用 pandas 内置 ewm 方法）
            df[f'EMA5_{col}'] = df[col].ewm (span=5, adjust=False).mean ()
            df[f'EMA10_{col}'] = df[col].ewm (span=10, adjust=False).mean ()
            df[f'EMA20_{col}'] = df[col].ewm (span=20, adjust=False).mean ()

    # 动量指标 - 对价格计算 RSI, MACD, Stochastic, ROC
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[f'RSI_{col}'] = calculate_rsi (df[col].values, period=14)
            macd, signal_line, hist = calculate_macd (df[col].values, short=5, long=13, signal=9)
            df[f'MACD_value_{col}'] = macd
            df[f'MACD_signal_{col}'] = signal_line
            df[f'MACD_histogram_{col}'] = hist
            df[f'ROC_{col}'] = calculate_roc (df[col].values, period=14)

    if all (col in df.columns for col in ['high', 'low', 'close']):
        k, d = calculate_stochastic (df)
        df['Stoch_K'] = k
        df['Stoch_D'] = d

    # 成交量指标
    if 'volume' in df.columns:
        df['Volume_SMA5'] = df['volume'].rolling (window=5).mean ()
        df['Volume_SMA10'] = df['volume'].rolling (window=10).mean ()
        df['Volume_ROC'] = calculate_volume_roc (df['volume'].values, period=14)
        if 'close' in df.columns:
            df['OBV'] = calculate_obv (df)

    # 输出新的 CSV 文件
    filename = os.path.basename (csv_path).replace (".csv", "_factored.csv")
    os.makedirs (output_dir, exist_ok=True)
    output_path = os.path.join (output_dir, filename)
    df.to_csv (output_path, index=False)
    print (f"✔️ Factors added and saved to {output_path}")


def clean_outliers (df, columns=None, z_thresh=5):
    """
    清洗 DataFrame 中价格类或数值类字段的异常值：
    - 若未指定 columns，则自动选择所有数值列
    - 使用前后邻居均值替代异常值（如果两个邻居都存在且非异常）
    - 填充完成后，再统一 drop 含 NaN 的整行
    """
    if columns is None:
        columns = df.select_dtypes (include=[np.number]).columns.tolist ()

    total_replaced = 0

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].copy ()
        z_scores = (series - series.mean ()) / series.std ()
        outliers = np.abs (z_scores) > z_thresh
        outlier_indices = np.where (outliers)[0]

        print (f"检测到 {len (outlier_indices)} 个异常值在 {col} 列")

        for i in outlier_indices:
            left = None
            right = None

            # 向前找邻居
            for j in range (i - 1, -1, -1):
                if not outliers[j] and not pd.isna (series[j]):
                    left = series[j]
                    break

            # 向后找邻居
            for j in range (i + 1, len (series)):
                if not outliers[j] and not pd.isna (series[j]):
                    right = series[j]
                    break

            if left is not None and right is not None:
                series[i] = (left + right) / 2
                total_replaced += 1
            else:
                series[i] = np.nan  # 标记为 NaN，之后统一 drop

        df[col] = series

    # 统一删除含 NaN 的整行
    original_len = len (df)
    df.dropna (inplace=True)
    dropped_rows = original_len - len (df)

    print (f"✅ 替换了 {total_replaced} 个异常值")
    print (f"❌ 删除了 {dropped_rows} 行（因为存在 NaN）")

    return df


def aggregate_high_freq_to_low (df, freq='1h'):
    """
    使用标准差等波动性指标替代原始价格信息，保留趋势性因子
    """
    df['timestamp'] = pd.to_datetime (df['timestamp'])
    df.set_index ('timestamp', inplace=True)

    # 使用标准差来表示 open, high, low, close, volume 的波动性
    df_agg = df.resample (freq).agg ({
        'open': 'std',
        'high': 'std',
        'low': 'std',
        'close': 'std',
        'volume': 'std'
    })

    df_agg.rename (columns={
        'open': f'open_volatility',
        'high': 'high_volatility',
        'low': 'low_volatility',
        'close': 'close_volatility',
        'volume': 'volume_volatility'
    }, inplace=True)

    df_agg.reset_index (inplace=True)
    return df_agg


def process_high_freq_to_low (csv_path, output_dir="aggregatedData", freq='1h'):
    """
    读取高频 CSV，清洗异常值，聚合为低频数据，去除缺失行，并保存。
    输出文件名示例：SPY_5minute_train_agg_1h.csv
    """
    df = pd.read_csv (csv_path)

    print ("\n清洗 open high low close 异常值:")
    df = clean_outliers (df, columns=['open', 'high', 'low', 'close'], z_thresh=5)

    # 聚合为低频
    df_low = aggregate_high_freq_to_low (df, freq=freq)

    print ("\n清洗 open high low close 异常值, Drop 空 行:")
    df_low = clean_outliers (df_low, columns=['open', 'high', 'low', 'close'], z_thresh=10)

    # 保存文件
    filename = os.path.basename (csv_path).replace (".csv", f"_agg_{freq}.csv")
    os.makedirs (output_dir, exist_ok=True)
    output_path = os.path.join (output_dir, filename)
    df_low.to_csv (output_path, index=False)
    print (f"✅ Aggregated {freq} data saved to {output_path} (Total: {len (df_low)} rows)")


def join_csv_on_timestamp (csv1, csv2, output_file, how='inner'):
    df1 = pd.read_csv (csv1, parse_dates=['timestamp'])
    df2 = pd.read_csv (csv2, parse_dates=['timestamp'])

    df_merged = pd.merge (df1, df2, on='timestamp', how=how)

    clean_outliers (df_merged, columns=['open', 'high', 'low', 'close'], z_thresh=10)

    df_merged.to_csv (output_file, index=False)
    print (f"✅ Join complete. Saved to {output_file} ({len (df_merged)} rows)")


def add_direction_labels_to_csv (input_csv, output_csv, feature_list):
    """
    读取一个 CSV 文件，并在其后面添加新的数据特征，
    每个新特征表示对应列与前一行数据比较的涨跌情况：
      - 上涨为 1
      - 下跌为 -1
      - 持平为 0

    参数：
      - input_csv: 输入 CSV 文件路径
      - output_csv: 输出 CSV 文件路径（保存添加新特征后的数据）
      - feature_list: 需要生成涨跌标签的列名列表，如 ['open', 'high', 'low', 'close']

    返回：
      - None
    """
    # 读取 CSV 数据
    df = pd.read_csv (input_csv)

    # 遍历每个指定的特征，计算与前一行的差值并生成涨跌标签
    for feature in feature_list:
        # diff() 计算当前行与前一行的差值
        diff = df[feature].diff ()
        # 根据差值判断涨跌：
        # 大于 0 为 1， 小于 0 为 -1， 等于 0 为 0
        # 首行因为无前一行，diff 得到 NaN，统一填充为 0
        df[f'{feature}_label'] = diff.apply (lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).fillna (0)

    # 将更新后的 DataFrame 写入新的 CSV 文件
    df.to_csv (output_csv, index=False)
    print (f"已生成新的 CSV 文件：{output_csv}")


# 把高频转换低频数据
process_high_freq_to_low ("rawdata/SPY_10minute_test.csv", freq='30min')
process_high_freq_to_low ("rawdata/SPY_10minute_validate.csv", freq='30min')
process_high_freq_to_low ("rawdata/SPY_10minute_train.csv", freq='30min')

# 添加因子
add_factors_to_csv ("rawData/SPY_30minute_test.csv")
add_factors_to_csv ("rawData/SPY_30minute_train.csv")
add_factors_to_csv ("rawData/SPY_30minute_validate.csv")

# 合并
join_csv_on_timestamp ("factoredData/SPY_30minute_test_factored.csv",
                       "aggregatedData/SPY_10minute_test_agg_30min.csv",
                       output_file="readyData/SPY_30minute_test.csv",
                       how='left')

join_csv_on_timestamp ("factoredData/SPY_30minute_validate_factored.csv",
                       "aggregatedData/SPY_10minute_validate_agg_30min.csv",
                       output_file="readyData/SPY_30minute_validate.csv",
                       how='left')

join_csv_on_timestamp ("factoredData/SPY_30minute_train_factored.csv",
                       "aggregatedData/SPY_10minute_train_agg_30min.csv", output_file="readyData/SPY_30minute_train.csv",
                       how='left')


# df = pd.read_csv("readyData/SPY_1hour_test.csv")
# print(df.shape)
