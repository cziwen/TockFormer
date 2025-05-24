import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class Preprocessor:
    @staticmethod
    def normalize_timestamp (df: pd.DataFrame, column: str = 'timestamp') -> pd.DataFrame:
        df = df.copy ()
        if column not in df.columns:
            raise ValueError (f"Column '{column}' not found in DataFrame.")

        ts = df[column]
        # 1) 第一遍：数值当 ms，字符串自动推断
        if pd.api.types.is_numeric_dtype (ts):
            parsed = pd.to_datetime (ts, unit='ms', errors='coerce', utc=True)
        else:
            parsed = pd.to_datetime (ts, errors='coerce', utc=True)

        # 2) 对第一遍中还是 NaT 的，再用带 %z 的格式专门解析一次
        mask_nat = parsed.isna ()
        if mask_nat.any ():
            # 只对那些还没解析上的原始字符串做第二次尝试
            parsed_fallback = pd.to_datetime (
                ts[mask_nat],
                format='%Y-%m-%d %H:%M:%S%z',
                errors='coerce',
                utc=True
            )
            parsed.loc[mask_nat] = parsed_fallback

        # 3) 报告一下最终还有多少没解析上
        final_nat = parsed.isna ()
        if final_nat.any ():
            bad = ts[final_nat].unique ()[:10]
            tqdm.write (f"[Warning] 最终还有 {final_nat.sum ()} 条解析失败，示例：{bad}")

        # 4) 统一转换时区并 floor 到分钟
        df[column] = (
            parsed
            .dt.tz_convert ('America/New_York')
            .dt.floor ('min')
        )
        return df

    @staticmethod
    def normalize_timestamp_parallel (
            df: pd.DataFrame,
            column: str = 'timestamp',
            n_jobs: int = 4,
            max_chunk_size: int = 10000
    ) -> pd.DataFrame:
        """
        多进程版：把 DataFrame 切块后并发标准化时间戳，再合并。

        参数：
          - df:           原始 DataFrame
          - column:       时间戳列名
          - n_jobs:       并发进程数
          - max_chunk_size: 每块最大行数，超过会再分
        """
        # 1) 切分成大致相同大小的子块
        N = len (df)
        if max_chunk_size:
            n_jobs = min (n_jobs, math.ceil (N / max_chunk_size))
        # 用 numpy.array_split 保持顺序
        chunks = np.array_split (df, n_jobs)

        results = []
        with ProcessPoolExecutor (max_workers=n_jobs) as exe:
            # futures 列表
            futures = {exe.submit (Preprocessor.normalize_timestamp, chunk, column): i
                       for i, chunk in enumerate (chunks)}
            for future in tqdm (as_completed (futures), total=len (futures), desc="Normalize ts"):
                # 按完成顺序收集
                results.append (future.result ())

        # 按原来顺序合并
        # 这里简单按照 chunks 的顺序拼回去
        results_sorted = [None] * len (results)
        for fut, idx in futures.items ():
            # 这里 futures mapping 存了 idx，但 as_completed 丢了 key
            # 所以更稳妥的方法是：在 submit 时把 (idx,chunk) 一同传回：
            pass

        # ——为了简单——我们直接 concat 并 sort_index
        df_out = pd.concat (results, axis=0)
        return df_out.sort_index ()

    @staticmethod
    def aggregate_tick_to_minute (df: pd.DataFrame, z: float = 2.0) -> pd.DataFrame:
        """
        将逐笔 Tick 数据聚合到分钟级，输出 OHLCV、VWAP 以及微观结构特征。
        保留 tqdm 进度条，分为 5 步：
          1) 时间戳标准化
          2) 丢弃无效行 & 类型转换
          3) 计算 dollar_value
          4) 预计算微观结构中间量
          5) 按分钟重采样并聚合 + 衍生比例
        """
        pbar = tqdm (total=5, desc='Aggregate tick to minute')

        # 1. 标准化 timestamp
        df = Preprocessor.normalize_timestamp_parallel (df, 'timestamp', n_jobs=10, max_chunk_size=100000)
        pbar.update (1)

        # 2. 丢弃转换失败的行，并设为索引
        original_len = len (df)
        df = df.dropna (subset=['timestamp']).set_index ('timestamp')
        dropped = original_len - len (df)
        tqdm.write (f"[Info] Dropped {dropped} rows due to invalid timestamps.")

        # 3. 转换类型 & 预计算 dollar_value
        df['price'] = df['price'].astype (float)
        df['volume'] = df['volume'].astype (float)
        df['dollar_value'] = df['price'] * df['volume']
        pbar.update (1)

        # 4. 预计算微观结构中间量
        #   - price_diff, is_zero, is_up
        #   - 每分钟的 volume 均值/标准差，用于 z-score
        #   - large_trade 及其 volume
        df['minute'] = df.index.floor ('1min')
        df['price_diff'] = df.groupby ('minute')['price'].diff ().fillna (0.0)
        df['is_zero'] = (df['price_diff'] == 0).astype (int)
        df['is_up'] = (df['price_diff'] > 0).astype (int)

        vol_mean = df.groupby ('minute')['volume'].transform ('mean')
        vol_std = df.groupby ('minute')['volume'].transform ('std').replace (0, np.nan)
        df['z_score'] = (df['volume'] - vol_mean) / vol_std
        df['large_trade'] = (df['z_score'] > z).astype (int)
        df['large_trade_vol'] = df['volume'] * df['large_trade']
        pbar.update (1)

        # 5. 按分钟重采样并命名聚合
        df_min = df.resample ('1min').agg (
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('volume', 'sum'),
            dollar_volume=('dollar_value', 'sum'),
            tick_count=('price', 'count'),
            trade_size_mean=('volume', 'mean'),
            trade_size_std=('volume', 'std'),
            zero_return_count=('is_zero', 'sum'),
            price_direction_ratio=('is_up', 'mean'),
            large_trade_count=('large_trade', 'sum'),
            large_trade_volume=('large_trade_vol', 'sum'),
        )
        pbar.update (1)

        # 6. 衍生比率指标
        df_min['vwap'] = df_min['dollar_volume'] / df_min['volume']
        df_min['large_trade_ratio'] = df_min['large_trade_count'] / df_min['tick_count']
        df_min['large_trade_volume_ratio'] = df_min['large_trade_volume'] / df_min['volume']
        pbar.update (1)

        return df_min.reset_index ()
