import pandas as pd
import ta
from Utility.registry import factor
from typing import Callable, Dict, List, Any, Optional


# —— 在这里只写因子函数 ——
@factor ('bounded')
def rsi_factors (
        self,
        windows: List[int] = [6, 10, 14]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in windows:
        for c in self.base_cols:
            if c not in self.df_global.columns:
                continue
            key = f"rsi_{w}_{c}"
            out[key] = ta.momentum.RSIIndicator (self.df_global[c], window=w).rsi ()
    return out


@factor ('unbounded')
def sma_factors (
        self,
        windows: List[int] = [5, 10, 20]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in windows:
        for c in self.base_cols:
            if c not in self.df_global.columns:
                continue
            out[f"sma_{w}_{c}"] = self.df_global[c].rolling (w).mean ()
    return out


@factor ('unbounded')
def ema_factors (
        self,
        spans: List[int] = [5, 10, 20]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for span in spans:
        for c in self.base_cols:
            if c not in self.df_global.columns:
                continue
            out[f"ema_{span}_{c}"] = self.df_global[c].ewm (span=span).mean ()
    return out


@factor ('unbounded')
def macd_factors (
        self,
        pairs: List[tuple] = [(12, 26), (5, 20)]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for fast, slow in pairs:
        for c in self.base_cols:
            if c not in self.df_global.columns:
                continue
            macd = ta.trend.MACD (self.df_global[c], window_fast=fast, window_slow=slow)
            out[f"macd_diff_{fast}_{slow}_{c}"] = macd.macd_diff ()
    return out


@factor ('bounded')
def bb_pband_factors (
        self,
        windows: List[int] = [10, 20],
        devs: List[float] = [1.5, 2.0]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if 'close' in self.df_global.columns and 'close' in self.base_cols:
        for w in windows:
            for dev in devs:
                bb = ta.volatility.BollingerBands (
                    self.df_global['close'], window=w, window_dev=dev
                )
                out[f"bb_pband_{w}_{dev}"] = bb.bollinger_pband ()
    return out

# …继续使用 @factor 添加更多因子…
