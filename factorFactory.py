import pandas as pd
import ta
from typing import Callable, Dict, List, Any

# Registry to hold factor methods and their metadata
FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

def factor(tag: str):
    """
    Decorator to register a method as a factor generator.
    tag: 'bounded' or 'unbounded'
    """
    def decorator(fn: Callable):
        FACTOR_REGISTRY[fn.__name__] = {
            'func': fn,
            'tag': tag
        }
        return fn
    return decorator

class FactorFactory:
    def __init__(self):
        # parameter configurations
        self.rsi_windows = [6, 10, 14, 20, 30]
        self.sma_windows = [5, 10, 20, 30, 50, 100, 200]
        self.ema_spans   = [5, 10, 12, 20, 30, 50, 100, 200]
        self.macd_pairs  = [(12,26), (5,20), (8,17), (10,30)]
        self.bb_windows  = [10, 20, 30]
        self.bb_dev      = [1.5, 2.0, 2.5]
        self.price_cols  = ['open','high','low','close']

    def generate_factors(
        self,
        df: pd.DataFrame,
        include_bounded_factors: bool = True
    ) -> pd.DataFrame:
        """
        Generate all registered factors.
        """
        df_in = df.copy().sort_values('timestamp').set_index('timestamp')
        feats = {}
        # iterate registry
        for name, meta in FACTOR_REGISTRY.items():
            fn = meta['func']
            tag = meta['tag']
            # skip price-based when include_price_factors False and tag == 'price'
            if not include_bounded_factors and getattr(fn, 'price_factor', False):
                continue
            out = fn(self, df_in)
            if isinstance(out, pd.Series):
                feats[name] = out
            else:
                feats.update(out)
        return pd.concat([df_in, pd.DataFrame(feats, index=df_in.index)], axis=1).reset_index()

    # ====== individual factor methods ======

    @factor(tag='bounded')
    def rsi_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.rsi_windows:
            for c in self.price_cols:
                key = f'rsi_{w}_{c}'
                feats[key] = ta.momentum.RSIIndicator(df[c], window=w).rsi()
        return feats

    @factor(tag='unbounded')
    def sma_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.sma_windows:
            for c in self.price_cols:
                key = f'sma_{w}_{c}'
                feats[key] = df[c].rolling(w).mean()
        return feats
    sma_factors.price_factor = True

    @factor(tag='unbounded')
    def ema_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats = {}
        for s in self.ema_spans:
            for c in self.price_cols:
                key = f'ema_{s}_{c}'
                feats[key] = df[c].ewm(span=s).mean()
        return feats
    ema_factors.price_factor = True

    @factor(tag='unbounded')
    def macd_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats = {}
        for fast, slow in self.macd_pairs:
            for c in self.price_cols:
                key = f'macd_diff_{fast}_{slow}_{c}'
                macd = ta.trend.MACD(df[c], window_fast=fast, window_slow=slow)
                feats[key] = macd.macd_diff()
        return feats

    @factor(tag='bounded')
    def bb_pband_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.bb_windows:
            for dev in self.bb_dev:
                key = f'bb_pband_{w}_{dev}'
                bb = ta.volatility.BollingerBands(df['close'], window=w, window_dev=dev)
                feats[key] = bb.bollinger_pband()
        return feats

    @factor(tag='unbounded')
    def bb_price_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats = {}
        for w in self.bb_windows:
            for dev in self.bb_dev:
                mavg = ta.volatility.BollingerBands(df['close'], window=w, window_dev=dev).bollinger_mavg()
                feats[f'bb_mavg_{w}_{dev}'] = mavg
                feats[f'bb_hband_{w}_{dev}'] = ta.volatility.BollingerBands(df['close'], window=w, window_dev=dev).bollinger_hband()
                feats[f'bb_lband_{w}_{dev}'] = ta.volatility.BollingerBands(df['close'], window=w, window_dev=dev).bollinger_lband()
        return feats
    bb_price_factors.price_factor = True

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
