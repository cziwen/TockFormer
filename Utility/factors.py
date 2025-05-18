import pandas as pd
import ta
from Utility.registry import factor
from typing import Dict, List

# —— 在这里只写因子函数 ——
@factor('bounded')
def rsi(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: List[int] = [6, 10, 14]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in windows:
        for c in cols:
            if c not in df.columns:
                continue
            key = f"rsi_{w}_{c}"
            out[key] = ta.momentum.RSIIndicator(df[c], window=w).rsi()
    return out


@factor('unbounded')
def sma(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: List[int] = [5, 10, 20]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in windows:
        for c in cols:
            if c not in df.columns:
                continue
            key = f"sma_{w}_{c}"
            out[key] = df[c].rolling(w).mean()
    return out


@factor('unbounded')
def ema(
        self,
        df: pd.DataFrame,
        cols: List[str],
        spans: List[int] = [5, 10, 20]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for span in spans:
        for c in cols:
            if c not in df.columns:
                continue
            key = f"ema_{span}_{c}"
            out[key] = df[c].ewm(span=span).mean()
    return out


@factor('unbounded')
def macd(
        self,
        df: pd.DataFrame,
        cols: List[str],
        pairs: List[tuple] = [(12, 26), (5, 20)]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for fast, slow in pairs:
        for c in cols:
            if c not in df.columns:
                continue
            macd_ind = ta.trend.MACD(df[c], window_fast=fast, window_slow=slow)
            key = f"macd_diff_{fast}_{slow}_{c}"
            out[key] = macd_ind.macd_diff()
    return out


@factor('bounded')
def bb_pband(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: List[int] = [10, 20],
        devs: List[float] = [1.5, 2.0]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for w in windows:
        for dev in devs:
            for c in cols:
                if c not in df.columns:
                    continue
                bb = ta.volatility.BollingerBands(df[c], window=w, window_dev=dev)
                key = f"bb_pband_{w}_{dev}_{c}"
                out[key] = bb.bollinger_pband()
    return out
