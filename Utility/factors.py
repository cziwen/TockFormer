import pandas as pd
import ta
from Utility.registry import factor
from typing import Callable, Dict, List, Any, Optional

# —— 在这里只写因子函数 ——


@factor ('bounded', 'individual')
def rsi_factors (self) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in self.rsi_windows:
        for c in self.target_cols:
            if c not in self.df_global.columns:
                continue
            key = f"rsi_{w}_{c}"
            out[key] = ta.momentum.RSIIndicator (self.df_global[c], window=w).rsi ()
    return out


@factor ('unbounded', 'individual')
def sma_factors (self) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in self.sma_windows:
        for c in self.target_cols:
            if c not in self.df_global.columns:
                continue
            out[f"sma_{w}_{c}"] = self.df_global[c].rolling (w).mean ()
    return out


@factor ('unbounded', 'individual')
def ema_factors (self) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for span in self.ema_spans:
        for c in self.target_cols:
            if c not in self.df_global.columns:
                continue
            out[f"ema_{span}_{c}"] = self.df_global[c].ewm (span=span).mean ()
    return out


@factor ('unbounded', 'individual')
def macd_factors (self) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for fast, slow in self.macd_pairs:
        for c in self.target_cols:
            if c not in self.df_global.columns:
                continue
            macd = ta.trend.MACD (
                self.df_global[c], window_fast=fast, window_slow=slow
            )
            out[f"macd_diff_{fast}_{slow}_{c}"] = macd.macd_diff ()
    return out


@factor ('bounded', 'advanced')
def bb_pband_factors (self) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if 'close' in self.df_global.columns and 'close' in self.target_cols:
        for w in self.bb_windows:
            for dev in self.bb_dev:
                bb = ta.volatility.BollingerBands (
                    self.df_global['close'], window=w, window_dev=dev
                )
                out[f"bb_pband_{w}_{dev}"] = bb.bollinger_pband ()
    return out

# …继续使用 @factor 添加更多因子…
