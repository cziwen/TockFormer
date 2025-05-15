import pandas as pd
import numpy as np
import ta
import math
from typing import Callable, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# === 1. 因子注册 Decorator & Registry ===
FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

def factor(*tags: str):
    """
    注册因子方法；tags 示例：
      @factor('bounded','individual')
      @factor('unbounded','advanced')
    """
    def decorator(fn: Callable):
        FACTOR_REGISTRY[fn.__name__] = {
            'func': fn,
            'tags': set(tags)
        }
        return fn
    return decorator

# === 2. 定义可用的交叉/一元操作及其“安全元数据” ===
CROSS_OPS: List[Dict[str, Any]] = [
    {
        'name': 'mul',
        'arity': 2,
        'func': lambda x, y: x * y,
        'preserves_bounded': True
    },
    {
        'name': 'minus',
        'arity': 2,
        'func': lambda x, y: x - y,
        'preserves_bounded': True
    },
    {
        'name': 'div',
        'arity': 2,
        'func': lambda x, y: x.div(y.replace(0, pd.NA)),
        'preserves_bounded': False
    },
    {
        'name': 'sin',
        'arity': 1,
        'func': lambda x: x.map(math.sin),
        'preserves_bounded': True
    },
    {
        'name': 'cos',
        'arity': 1,
        'func': lambda x: x.map(math.cos),
        'preserves_bounded': True
    },
]

# === 3. FactorFactory 类 ===
class FactorFactory:
    def __init__(self, df: pd.DataFrame, target_cols: List[str]):
        """
        df: 原始包含 timestamp 列的数据
        target_cols: 要生成因子的列名列表；若某列不存在则跳过
        """
        self.df_global = (
            df.copy()
              .sort_values('timestamp')
              .set_index('timestamp')
        )
        self.target_cols = target_cols

        # —— 因子参数 ——
        self.rsi_windows = [6, 10, 14]
        self.sma_windows = [5, 10, 20]
        self.ema_spans   = [5, 10, 20]
        self.macd_pairs  = [(12,26), (5,20)]
        self.bb_windows  = [10, 20]
        self.bb_dev      = [1.5, 2.0]

    def generate_factors(
        self,
        layers: int = 2,
        include_only_bounded_factors: bool = True
    ) -> pd.DataFrame:
        """
        layers: 总层数（第 1 层基础因子；第 2+ 层做交叉）
        include_only_bounded_factors: 第 1 层是否只跑标记为 bounded 的 individual 因子
        """
        # ——— 1. 第 1 层：跑 individual + advanced 因子 ———
        base_feats: Dict[str, pd.Series] = {}
        # 存 metadata：每列是否 bounded
        meta: Dict[str, Dict[str, bool]] = {}

        # 收集 individual 方法
        ind_methods = [
            info for name, info in FACTOR_REGISTRY.items()
            if 'individual' in info['tags']
           and (not include_only_bounded_factors or 'bounded' in info['tags'])
        ]
        # 收集 advanced 方法
        adv_methods = [
            info for name, info in FACTOR_REGISTRY.items()
            if 'advanced' in info['tags']
        ]

        # 执行 individual
        for info in tqdm(ind_methods, desc="1️⃣ Individual factors"):
            out = info['func'](self)
            items = ([out] if isinstance(out, pd.Series)
                     else [pd.Series(v, index=self.df_global.index, name=k)
                           for k, v in out.items()])
            for s in items:
                col = s.name
                base_feats[col] = s
                meta[col] = {'is_bounded': ('bounded' in info['tags'])}

        # 执行 advanced
        for info in tqdm(adv_methods, desc="2️⃣ Advanced factors"):
            out = info['func'](self)
            items = ([out] if isinstance(out, pd.Series)
                     else [pd.Series(v, index=self.df_global.index, name=k)
                           for k, v in out.items()])
            for s in items:
                col = s.name
                base_feats[col] = s
                meta[col] = {'is_bounded': ('bounded' in info['tags'])}

        # 构造第 1 层 DataFrame
        feat_df = pd.DataFrame(base_feats)

        # ——— 2. 多层交叉 ———
        prev_feats = base_feats.copy()
        for layer in range(2, layers + 1):
            new_feats: Dict[str, pd.Series] = {}
            future_to_col: Dict[Any, str] = {}

            with ThreadPoolExecutor() as exe:
                # 二元交叉：上一层 × 第 1 层
                for name1, s1 in prev_feats.items():
                    for name2, s2 in base_feats.items():
                        for op in CROSS_OPS:
                            if op['arity'] != 2 or not op['preserves_bounded']:
                                continue
                            # 仅 bounded × bounded
                            if not (meta[name1]['is_bounded'] and meta[name2]['is_bounded']):
                                continue
                            col = f"{name1}_{op['name']}_{name2}"
                            fut = exe.submit(op['func'], s1, s2)
                            future_to_col[fut] = col

                # 一元交叉：sin/cos
                for name1, s1 in prev_feats.items():
                    for op in CROSS_OPS:
                        if op['arity'] != 1 or not op['preserves_bounded']:
                            continue
                        col = f"{op['name']}_{name1}"
                        fut = exe.submit(op['func'], s1)
                        future_to_col[fut] = col

                # 收集结果并用 tqdm 展示进度
                for fut in tqdm(as_completed(future_to_col),
                                total=len(future_to_col),
                                desc=f"🔄 Layer {layer} cross"):
                    col = future_to_col[fut]
                    s = fut.result()
                    if s.isna().all():
                        continue
                    s.name = col
                    new_feats[col] = s
                    # 新特征一律标为 bounded
                    meta[col] = {'is_bounded': True}

            # 下一层以 new_feats 为基础
            prev_feats = new_feats
            if new_feats:
                feat_df = pd.concat([feat_df, pd.DataFrame(new_feats)], axis=1)

        # 最终拼回 timestamp 列
        feat_df.insert(0, 'timestamp', feat_df.index)
        return feat_df.reset_index(drop=True)

    # ========== 示例因子方法 ==========

    @factor('bounded', 'individual')
    def rsi_factors(self) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for w in self.rsi_windows:
            for c in self.target_cols:
                if c not in self.df_global.columns:
                    continue
                key = f"rsi_{w}_{c}"
                out[key] = ta.momentum.RSIIndicator(
                    self.df_global[c], window=w
                ).rsi()
        return out

    @factor('unbounded', 'individual')
    def sma_factors(self) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for w in self.sma_windows:
            for c in self.target_cols:
                if c not in self.df_global.columns:
                    continue
                out[f"sma_{w}_{c}"] = self.df_global[c].rolling(w).mean()
        return out

    @factor('unbounded', 'individual')
    def ema_factors(self) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for span in self.ema_spans:
            for c in self.target_cols:
                if c not in self.df_global.columns:
                    continue
                out[f"ema_{span}_{c}"] = self.df_global[c].ewm(span=span).mean()
        return out

    @factor('unbounded', 'individual')
    def macd_factors(self) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for fast, slow in self.macd_pairs:
            for c in self.target_cols:
                if c not in self.df_global.columns:
                    continue
                macd = ta.trend.MACD(
                    self.df_global[c],
                    window_fast=fast,
                    window_slow=slow
                )
                out[f"macd_diff_{fast}_{slow}_{c}"] = macd.macd_diff()
        return out

    @factor('bounded', 'advanced')
    def bb_pband_factors(self) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        # 只针对 'close' 列
        if 'close' in self.df_global.columns and 'close' in self.target_cols:
            for w in self.bb_windows:
                for dev in self.bb_dev:
                    bb = ta.volatility.BollingerBands(
                        self.df_global['close'], window=w, window_dev=dev
                    )
                    out[f"bb_pband_{w}_{dev}"] = bb.bollinger_pband()
        return out

    # …你可以继续用 @factor 模板添加更多因子方法…