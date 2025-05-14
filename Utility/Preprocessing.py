import pandas as pd
import numpy as np


def aggregate_tick_to_minute (df: pd.DataFrame) -> pd.DataFrame:

    df = normalize_timestamp(df, column='timestamp')

    # 打印原始行数
    original_len = len (df)
    df = df.dropna (subset=['timestamp'])  # 防止时间列转换失败导致 NaT
    dropped_len = original_len - len (df)
    print (f"[Info] Dropped {dropped_len} rows due to invalid timestamps.")

    df = df.set_index ('timestamp')

    # 转换数值类型
    df['price'] = df['price'].astype (float)
    df['volume'] = df['volume'].astype (float)
    df['dollar_value'] = df['price'] * df['volume']

    # 分钟级分组
    grouped = df.resample ('1min')

    # VWAP 计算
    vwap_series = grouped.apply (
        lambda g: (g['price'] * g['volume']).sum () / g['volume'].sum ()
        if g['volume'].sum () != 0 else np.nan
    )
    vwap_series.name = 'vwap'

    # 聚合成 OHLCV + dollar volume
    df_minute = grouped.agg ({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
        'dollar_value': 'sum'
    })
    df_minute.columns = ['open', 'high', 'low', 'close', 'volume', 'dollar_volume']
    df_minute['vwap'] = vwap_series.reindex (df_minute.index)

    # 微观结构特征
    microstructure_df = grouped.apply (extract_microstructure_features)
    df_minute = pd.concat ([df_minute, microstructure_df], axis=1)

    return df_minute.reset_index ()


def extract_microstructure_features (g: pd.DataFrame, z: float = 2.5) -> pd.Series:
    price = g['price'].astype (float).values
    volume = g['volume'].astype (float).values
    diffs = np.diff (price)

    # 基本微观结构特征
    tick_count = len (price)
    trade_size_mean = np.mean (volume) if tick_count > 0 else 0
    trade_size_std = np.std (volume) if tick_count > 0 else 0
    zero_return_count = np.sum (diffs == 0)
    price_direction_ratio = np.mean (diffs > 0) if len (diffs) > 0 else 0.5

    # 大单判定逻辑（z-score）
    if trade_size_std > 1e-6:  # 避免除以0
        z_scores = (volume - trade_size_mean) / trade_size_std
        large_trades = z_scores > z
        large_trade_count = np.sum (large_trades)
        large_trade_ratio = large_trade_count / tick_count
        large_trade_volume_ratio = np.sum (volume[large_trades]) / np.sum (volume)
    else:
        large_trade_count = 0
        large_trade_ratio = 0
        large_trade_volume_ratio = 0

    return pd.Series ({
        'tick_count': tick_count,
        'trade_size_mean': trade_size_mean,
        'trade_size_std': trade_size_std,
        'zero_return_count': zero_return_count,
        'price_direction_ratio': price_direction_ratio,
        'large_trade_count': large_trade_count,
        'large_trade_ratio': large_trade_ratio,
        'large_trade_volume_ratio': large_trade_volume_ratio
    })

def normalize_timestamp(df: pd.DataFrame, column: str = 'timestamp') -> pd.DataFrame:
    """
    标准化 DataFrame 中的时间戳列，自动处理 Unix 毫秒、字符串、datetime 类型，并转换为 America/New_York 时区。
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    ts = df[column]

    if pd.api.types.is_numeric_dtype(ts):
        # Unix 毫秒
        df[column] = (
            pd.to_datetime(ts, unit='ms', utc=True)
              .dt.tz_convert('America/New_York')
        )

    elif pd.api.types.is_datetime64_any_dtype(ts):
        # 已经是 datetime64
        # 1) 如果是无时区的，先 localize 到 UTC
        if ts.dt.tz is None:
            df[column] = ts.dt.tz_localize('UTC')
        else:
            df[column] = ts  # 保持原样以便下一步转换
        # 2) 统一转换到纽约时区（对已在纽约时区的 Series 也无副作用）
        df[column] = df[column].dt.tz_convert('America/New_York')

    else:
        # 字符串或其他
        parsed = pd.to_datetime(ts, errors='coerce')
        if parsed.dt.tz is None:
            df[column] = parsed.dt.tz_localize('UTC')
        else:
            df[column] = parsed
        df[column] = df[column].dt.tz_convert('America/New_York')

    return df