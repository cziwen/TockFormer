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
    high = df['High'].rolling (period).max ()
    low = df['Low'].rolling (period).min ()
    k_raw = (df['Close'] - low) / (high - low) * 100
    k_smooth = k_raw.rolling (smooth_k).mean ()
    d = k_smooth.rolling (signal).mean ()
    return k_smooth, d


def add_factors_to_csv (csv_path, output_dir="factoredData"):
    '''
    添加根据 open high low close 来添加 各种因子（rsi，macd，stoch_kd）

    :param csv_path:
    :param output_dir:
    :return: 保存到路径新的csv
    '''
    df = pd.read_csv (csv_path)

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[f'RSI_{col}'] = calculate_rsi (df[col].values, period=14)
            macd, signal_line, hist = calculate_macd (df[col].values)
            df[f'MACD_value_{col}'] = macd
            df[f'MACD_signal_{col}'] = signal_line
            df[f'MACD_histogram_{col}'] = hist

    if all (col in df.columns for col in ['High', 'Low', 'Close']):
        k, d = calculate_stochastic (df)
        df['Stoch_K'] = k
        df['Stoch_D'] = d

    # === 输出路径 ===
    filename = os.path.basename (csv_path).replace (".csv", "_factored.csv")
    os.makedirs (output_dir, exist_ok=True)
    output_path = os.path.join (output_dir, filename)
    df.to_csv (output_path, index=False)
    print (f"✔️ Factors added and saved to {output_path}")


def clean_price_outliers(df, columns=None, z_thresh=5):
    """
    清洗 DataFrame 中价格类或数值类字段的异常值：
    - 若未指定 columns，则自动选择所有数值列
    - 使用前后邻居均值替代异常值（如果两个邻居都存在且非异常）
    - 填充完成后，再统一 drop 含 NaN 的整行
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    total_replaced = 0

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].copy()
        z_scores = (series - series.mean()) / series.std()
        outliers = np.abs(z_scores) > z_thresh
        outlier_indices = np.where(outliers)[0]

        print(f"检测到 {len(outlier_indices)} 个异常值在 {col} 列")

        for i in outlier_indices:
            left = None
            right = None

            # 向前找邻居
            for j in range(i - 1, -1, -1):
                if not outliers[j] and not pd.isna(series[j]):
                    left = series[j]
                    break

            # 向后找邻居
            for j in range(i + 1, len(series)):
                if not outliers[j] and not pd.isna(series[j]):
                    right = series[j]
                    break

            if left is not None and right is not None:
                series[i] = (left + right) / 2
                total_replaced += 1
            else:
                series[i] = np.nan  # 标记为 NaN，之后统一 drop

        df[col] = series

    # 统一删除含 NaN 的整行
    original_len = len(df)
    df.dropna(inplace=True)
    dropped_rows = original_len - len(df)

    print(f"✅ 替换了 {total_replaced} 个异常值")
    print(f"❌ 删除了 {dropped_rows} 行（因为存在 NaN）")

    return df


def aggregate_high_freq_to_low (df, freq='1h'):
    """
    将高频数据（带时间戳）聚合为低频数据，用于大时间粒度建模的因子构造
    """
    df['timestamp'] = pd.to_datetime (df['timestamp'])
    df.set_index ('timestamp', inplace=True)

    df_agg = df.resample (freq).agg ({
        'open': ['first', 'mean', 'std'],
        'high': ['max', 'mean'],
        'low': ['min', 'mean'],
        'close': ['last', 'mean', 'std'],
        'volume': ['sum', 'mean']
    })

    df_agg.columns = ['_'.join (col).strip () for col in df_agg.columns.values]
    df_agg.reset_index (inplace=True)

    return df_agg


def process_high_freq_to_low (csv_path, output_dir="aggregatedData", freq='1h'):
    """
    读取高频 CSV，清洗异常值，聚合为低频数据，去除缺失行，并保存。
    输出文件名示例：SPY_5minute_train_agg_1h.csv
    """
    df = pd.read_csv (csv_path)

    print ("\n清洗 open high low close 异常值:")
    df = clean_price_outliers (df, columns= ['open', 'high', 'low', 'close'], z_thresh=5)

    # 聚合为低频
    df_low = aggregate_high_freq_to_low (df, freq=freq)

    print ("\n清洗 全部feature 异常值:")
    df_low = clean_price_outliers (df_low, z_thresh=10)

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
    df_merged.to_csv (output_file, index=False)
    print (f"✅ Join complete. Saved to {output_file} ({len (df_merged)} rows)")


process_high_freq_to_low ("rawdata/SPY_30minute_train.csv")
# join_csv_on_timestamp ("rawdata/SPY_1hour_train.csv", "aggregatedData/SPY_1minute_train_agg_1H.csv", "leftJoined.csv",
#                        how='inner')
