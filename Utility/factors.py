import pandas as pd
import ta
import math
from Utility.registry import factor, FACTOR_REGISTRY
from typing import Dict, List


# ──────────── operator functions ────────────
def mul_func (x, y):    return x * y


def minus_func (x, y):  return x - y


def div_func (x, y):    return x.div (y.replace (0, pd.NA))


def sin_func (x):       return x.map (math.sin)


def cos_func (x):       return x.map (math.cos)


CROSS_OPS = [
    {'name': 'mul', 'arity': 2, 'func': mul_func, 'preserves_bounded': True},
    {'name': 'minus', 'arity': 2, 'func': minus_func, 'preserves_bounded': True},
    {'name': 'div', 'arity': 2, 'func': div_func, 'preserves_bounded': False},
    {'name': 'sin', 'arity': 1, 'func': sin_func, 'preserves_bounded': True},
    {'name': 'cos', 'arity': 1, 'func': cos_func, 'preserves_bounded': True},
]


# ──────────── Feature functions ────────────
@factor (category='bounded', param_keys=['window'])
def rsi (
        df: pd.DataFrame,
        cols: List[str],
        window: List[int] = [6, 10, 14]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    tpl = FACTOR_REGISTRY['rsi']['key_template']
    for w in window:
        for c in cols:
            if c not in df.columns: continue
            key = tpl ([w], c)
            out[key] = ta.momentum.RSIIndicator (df[c], window=w).rsi ()
    return out


@factor (category='unbounded', param_keys=['window'])
def sma (
        df: pd.DataFrame,
        cols: List[str],
        window: List[int] = [5, 10, 20]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    tpl = FACTOR_REGISTRY['sma']['key_template']
    for w in window:
        for c in cols:
            if c not in df.columns: continue
            key = tpl ([w], c)
            out[key] = df[c].rolling (w).mean ()
    return out


@factor (category='unbounded', param_keys=['spans'])
def ema (
        df: pd.DataFrame,
        cols: List[str],
        spans: List[int] = [5, 10, 20]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    tpl = FACTOR_REGISTRY['ema']['key_template']
    for span in spans:
        for c in cols:
            if c not in df.columns: continue
            key = tpl ([span], c)
            out[key] = df[c].ewm (span=span).mean ()
    return out


@factor (
    category='unbounded',
    param_keys=['fast', 'slow'],
    key_template=lambda params, col: f"macd_{params[0]}_{params[1]}_({col})"
)
def macd (
        df: pd.DataFrame,
        cols: List[str],
        fast: List[int] = [12, 5],
        slow: List[int] = [26, 20]
) -> Dict[str, pd.Series]:
    # 确保 fast 和 slow 长度一致
    if len (fast) != len (slow):
        raise ValueError (f"参数 fast 和 slow 必须等长，got fast={fast}, slow={slow}")

    out: Dict[str, pd.Series] = {}
    tpl = FACTOR_REGISTRY['macd']['key_template']

    for c in cols:
        if c not in df.columns:
            continue

        # 对应位置配对：(12,26), (5,20)
        for f_val, s_val in zip (fast, slow):
            # 生成 key，比如 macd_12_26_(close)
            key = tpl ((f_val, s_val), c)
            # 计算对应参数的 MACD
            macd_ind = ta.trend.MACD (
                df[c],
                window_fast=f_val,
                window_slow=s_val
            )
            out[key] = macd_ind.macd_diff ()

    return out


@factor (category='bounded', param_keys=['window', 'std'])
def bbpband (
        df: pd.DataFrame,
        cols: List[str],
        window: List[int] = [10, 20],
        std: List[float] = [1.5, 2.0]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    tpl = FACTOR_REGISTRY['bbpband']['key_template']
    for w in window:
        for dev in std:
            for c in cols:
                if c not in df.columns: continue
                key = tpl ([w, dev], c)
                bb = ta.volatility.BollingerBands (df[c], window=w, window_dev=dev)
                out[key] = bb.bollinger_pband ()
    return out
