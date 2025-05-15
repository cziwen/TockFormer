import pandas as pd
import ta
from typing import Callable, Dict, List, Any
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


# Registry to hold factor methods and their metadata
FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

def factor(*tags: str):
    """
    Decorator to register a method as a factor generator.
    Supports multiple tags, e.g., @factor('bounded', 'technical').
    """
    def decorator(fn: Callable):
        FACTOR_REGISTRY[fn.__name__] = {
            'func': fn,
            'tags': set(tags)
        }
        return fn
    return decorator

def compute_temporal_cross_feature(task, df):
    prefix, f1, f2 = task
    s1, s2 = df[f1], df[f2]
    if s1.nunique() <= 1 or s2.nunique() <= 1:
        return {}
    return {
        f'{prefix}_cross_{f1}_mul_{f2}': s1 * s2,
        f'{prefix}_cross_{f1}_minus_{f2}': s1 - s2,
        f'{prefix}_cross_{f1}_div_{f2}': s1 / s2.replace(0, pd.NA)
    }

class FactorFactory:
    def __init__(self, df: pd.DataFrame):
        # parameter configurations
        self.rsi_windows = [6, 10, 14, 20, 30] # 10
        self.sma_windows = [5, 10, 20, 30, 50, 100, 200] # 21
        self.ema_spans   = [5, 10, 12, 20, 30, 50, 100, 200] # 28
        self.macd_pairs  = [(12,26), (5,20), (8,17), (10,30)] # 6
        self.bb_windows  = [10, 20, 30] # 3
        self.bb_dev      = [1.5, 2.0, 2.5] # 3
        self.price_cols  = ['open','high','low','close']
        self.df_global = df.copy().sort_values('timestamp').set_index('timestamp')

    def generate_factors(self, include_only_bounded_factors: bool = True) -> pd.DataFrame:
        """
        Generate all registered factors:
        1. Run all with 'individual' tag first (optionally restricted to 'bounded')
        2. Then run all with 'advanced' tag
        Update self.df_global with computed columns.
        """

        feats = {}

        # Phase 1: run 'individual' factors
        for name, meta in FACTOR_REGISTRY.items():
            tags = meta['tags']
            if 'individual' not in tags:
                continue
            if include_only_bounded_factors and 'bounded' not in tags:
                continue
            fn = meta['func']
            out = fn(self)
            if isinstance(out, pd.Series):
                feats[name] = out
            else:
                feats.update(out)

        self.df_global = pd.concat([self.df_global, pd.DataFrame(feats, index=self.df_global.index)], axis=1)

        # Phase 2: run 'advanced' factors
        feats = {}
        for name, meta in FACTOR_REGISTRY.items():
            tags = meta['tags']
            if 'advanced' not in tags:
                continue
            fn = meta['func']
            out = fn(self)
            if isinstance(out, pd.Series):
                feats[name] = out
            else:
                feats.update(out)

        self.df_global = pd.concat([self.df_global, pd.DataFrame(feats, index=self.df_global.index)], axis=1)

        return self.df_global.reset_index()

    # ====== individual factor methods ======
    @factor('bounded', 'individual')
    def rsi_factors(self) -> Dict[str, pd.Series]:
        print("Computing RSI factors...")
        feats = {}
        for w in self.rsi_windows:
            for c in self.price_cols:
                key = f'rsi_{w}_{c}'
                feats[key] = ta.momentum.RSIIndicator(self.df_global[c], window=w).rsi()
        return feats

    @factor('unbounded', 'individual')
    def sma_factors(self) -> Dict[str, pd.Series]:
        print("Computing SMA factors...")
        feats = {}
        for w in self.sma_windows:
            for c in self.price_cols:
                key = f'sma_{w}_{c}'
                feats[key] = self.df_global[c].rolling(w).mean()
        return feats

    @factor('unbounded', 'individual')
    def ema_factors(self) -> Dict[str, pd.Series]:
        print("Computing EMA factors...")
        feats = {}
        for s in self.ema_spans:
            for c in self.price_cols:
                key = f'ema_{s}_{c}'
                feats[key] = self.df_global[c].ewm(span=s).mean()
        return feats

    @factor('unbounded', 'individual')
    def macd_factors(self) -> Dict[str, pd.Series]:
        print("Computing MACD factors...")
        feats = {}
        for fast, slow in self.macd_pairs:
            for c in self.price_cols:
                key = f'macd_diff_{fast}_{slow}_{c}'
                macd = ta.trend.MACD(self.df_global[c], window_fast=fast, window_slow=slow)
                feats[key] = macd.macd_diff()
        return feats

    @factor('bounded', 'individual')
    def bb_pband_factors(self) -> Dict[str, pd.Series]:
        print("Computing BB factors...")
        feats = {}
        for w in self.bb_windows:
            for dev in self.bb_dev:
                key = f'bb_pband_{w}_{dev}'
                bb = ta.volatility.BollingerBands(self.df_global['close'], window=w, window_dev=dev)
                feats[key] = bb.bollinger_pband()
        return feats

    @factor('unbounded', 'individual')
    def bb_price_factors(self) -> Dict[str, pd.Series]:
        print("Computing BB price factors...")
        feats = {}
        for w in self.bb_windows:
            for dev in self.bb_dev:
                mavg = ta.volatility.BollingerBands(self.df_global['close'], window=w, window_dev=dev).bollinger_mavg()
                feats[f'bb_mavg_{w}_{dev}'] = mavg
                feats[f'bb_hband_{w}_{dev}'] = ta.volatility.BollingerBands(self.df_global['close'], window=w, window_dev=dev).bollinger_hband()
                feats[f'bb_lband_{w}_{dev}'] = ta.volatility.BollingerBands(self.df_global['close'], window=w, window_dev=dev).bollinger_lband()
        return feats


    # ====== advanced factor methods =====

    @factor('bounded', 'advanced')
    def temporal_cross_feature(self) -> Dict[str, pd.Series]:
        """
        For each factor type (rsi, sma, ema, macd), compute cross features (mul, minus, div)
        between all different windows with the same price field.
        """

        print("[CrossFeature] Building cross features by factor, price, and window...")

        feats = {}
        tasks = []

        def collect_factor_variants(factor_name: str, window_list: List, price_list: List[str]):
            for price in price_list:
                cols = []
                print(f"\n>>> Collecting for {factor_name} + {price}")
                for w in window_list:
                    col_name = f'{factor_name}_{w}_{price}'
                    print(f"Looking for: {col_name} → {'✓' if col_name in self.df_global.columns else '✗'}")
                    if col_name in self.df_global.columns:
                        cols.append(col_name)
                # Build pairwise tasks
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        tasks.append((factor_name, cols[i], cols[j]))

        # rsi, sma, ema follow (factor_name, window, price)
        collect_factor_variants('rsi', self.rsi_windows, self.price_cols)
        collect_factor_variants('sma', self.sma_windows, self.price_cols)
        collect_factor_variants('bb_pband', self.ema_spans, self.bb_dev)

        if not tasks:
            print("[CrossFeature] Warning: No valid factor combinations found.")
            return {}

        print(f"[CrossFeature] Launching {len(tasks)} tasks using {mp.cpu_count()} cores...")

        chunk_size = max(1, len(tasks) // (mp.cpu_count() * 4))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            for res in tqdm(pool.imap_unordered(partial(compute_temporal_cross_feature, df=self.df_global), tasks,
                                                chunksize=chunk_size), total=len(tasks),
                            desc="[CrossFeature] Progress"):
                feats.update(res)

        print("[CrossFeature] Done.")
        return feats


    # @factor('unbounded')
    # def momentum_atr_ratio(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
    #     """
    #     Momentum / ATR across all momentum periods and ATR windows.
    #     """
    #     feats = {}
    #     for mom_period in self.sma_windows:  # reuse sma_windows for momentum periods
    #         mom = ta.momentum.ROCIndicator(df['close'], window=mom_period).roc()
    #         for atr_w in self.bb_windows:  # reuse bb_windows for atr window
    #             atr = ta.volatility.AverageTrueRange(
    #                 df['high'], df['low'], df['close'], window=atr_w
    #             ).average_true_range()
    #             feats[f'mom{mom_period}_div_atr{atr_w}'] = mom / atr.replace(0, pd.NA)
    #     return feats
    #
    # @factor('bounded')
    # def obv_volume_ratio(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
    #     """
    #     OBV(OnBalanceVolume) / Volume across different EMA spans for volume smoothing.
    #     """
    #     feats = {}
    #     obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    #     for span in self.ema_spans:
    #         vol_ema = df['volume'].ewm(span=span, adjust=False).mean()
    #         feats[f'obv_div_vol_ema{span}'] = obv / vol_ema.replace(0, pd.NA)
    #     return feats
    #
    # @factor('unbounded')
    # def rsi_macd_cross(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
    #     """
    #     RSI × MACD(diff) for all combinations of RSI windows and MACD pairs.
    #     """
    #     feats = {}
    #     for rsi_w in self.rsi_windows:
    #         rsi = ta.momentum.RSIIndicator(df['close'], window=rsi_w).rsi()
    #         for fast, slow in self.macd_pairs:
    #             macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow).macd_diff()
    #             feats[f'cross_feature_rsi{rsi_w}_mul_macd_diff_{fast}_{slow}'] = rsi * macd
    #     return feats
    #
    # @factor(tag='unbounded')
    # def long_term_deviation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
    #     """
    #     Compute standardized deviation of close price from EMA for multiple long-term spans.
    #     Formula: (close / EMA - 1) * 100
    #     """
    #     feats = {}
    #     for span in self.sma_windows:  # 使用 sma_windows 作为 EMA 的周期范围
    #         ema = df['close'].ewm(span=span, adjust=False).mean()
    #         dev = ((df['close'] / ema.replace(0, pd.NA)) - 1.0) * 100
    #         feats[f'close_ema{span}_dev_pct'] = dev
    #     return feats


    # ... similarly add ATR, ADX, CCI, ROC, MOM, WR, volume factors
    # each method decorated with @factor(tag='bounded' or 'unbounded')

    def list_factors(self, tag: str = None) -> List[str]:
        """
        List registered factor method names, optionally filtered by tag.
        """
        if tag:
            return [name for name, meta in FACTOR_REGISTRY.items() if meta['tag'] == tag]
        return list(FACTOR_REGISTRY.keys())

    def call_factor(self, name: str, df: pd.DataFrame) -> pd.Series:
        """
        Call a single factor by name and return its series.
        """
        meta = FACTOR_REGISTRY[name]
        return meta['func'](self, df.set_index('timestamp')).rename(name)



# Usage:
# ff = FactorFactory()
# all_df = ff.generate_factors(df) # 求所有的因子
# bounded_list = ff.list_factors(tag='bounded') # 查看所有 有范围的 因子
# single_series = ff.call_factor('rsi_factors', df) # 计算单一因子方程
