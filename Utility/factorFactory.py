import pandas as pd
import numpy as np
import math
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from mpi4py import MPI

from Utility.registry import FACTOR_REGISTRY
from Utility.factors import *

# ──────────── operator functions ────────────
def mul_func(x, y):    return x * y
def minus_func(x, y):  return x - y
def div_func(x, y):    return x.div(y.replace(0, pd.NA))
def sin_func(x):       return x.map(math.sin)
def cos_func(x):       return x.map(math.cos)

CROSS_OPS = [
    {'name':'mul',   'arity':2, 'func':mul_func,   'preserves_bounded':True},
    {'name':'minus', 'arity':2, 'func':minus_func, 'preserves_bounded':True},
    {'name':'div',   'arity':2, 'func':div_func,   'preserves_bounded':False},
    {'name':'sin',   'arity':1, 'func':sin_func,   'preserves_bounded':True},
    {'name':'cos',   'arity':1, 'func':cos_func,   'preserves_bounded':True},
]

def _cross_apply(args):
    op, a, b, data_dict = args
    name = f"{a}_{op['name']}_{b}"
    return name, op['func'](data_dict[a], data_dict[b])


class FactorFactory:
    def __init__(
        self,
        df: pd.DataFrame,
        base_cols: Optional[List[str]] = None,
        target_col: str = 'close',
        forward_period: int = 1,
        window: int = 20,
        scaler: str = 'minmax',
        top_k: Optional[int] = None,
        decimals: int = 4
    ):
        # 原始数据按 timestamp 索引
        self.df_global = (
            df.copy()
              .sort_values('timestamp')
              .set_index('timestamp')
        )
        self.base_cols = (
            base_cols if base_cols is not None
            else [c for c in self.df_global.select_dtypes(include=[np.number]).columns]
        )
        self.target_col = target_col
        self._eval_kwargs = dict(
            forward_period=forward_period,
            window=window,
            scaler=scaler,
            top_k=top_k
        )
        self.decimals = decimals
        self.df_features = pd.DataFrame()
        self.summary     = pd.DataFrame()

    def generate_factors(
        self,
        mode: str = 'single',
        n_job: Optional[int] = None,
        bounded_only: bool = False
    ) -> pd.DataFrame:
        # 1) 生成基础因子
        base_feats: Dict[str, pd.Series] = {}
        for info in FACTOR_REGISTRY.values():
            tags = info.get('tags', [])
            if bounded_only and 'bounded' not in tags:
                continue
            out = info['func'](self)
            series_list = (
                [out] if isinstance(out, pd.Series)
                else [pd.Series(v, index=self.df_global.index, name=k)
                      for k, v in out.items()]
            )
            for s in series_list:
                base_feats[s.name] = s

        # 2) 构造 DataFrame
        feat_df = pd.DataFrame(base_feats, index=self.df_global.index)
        feat_df.index.name = 'timestamp'

        # 3) 交叉运算
        cross_df = self.cross_op(feat_df, mode, n_job, bounded_only)

        # 4) 合并并去常数/NaN
        merged = pd.concat([feat_df, cross_df], axis=1)
        drop_cols = merged.nunique()[merged.nunique() <= 1].index.tolist()
        df_feat = merged.drop(columns=drop_cols).dropna()

        # 5) reset_index 并评估
        self.df_features = df_feat.reset_index()
        # 保留小数
        self.df_features = self.df_features.round(self.decimals)

        self.evaluate_factors(**self._eval_kwargs)
        return self.df_features

    def cross_op(
        self,
        input_df: pd.DataFrame,
        mode: str = 'single',
        n_job: Optional[int] = None,
        bounded_only: bool = False
    ) -> pd.DataFrame:
        data = input_df.copy()
        cols = list(data.columns)
        feats: Dict[str, pd.Series] = {}

        # 一元操作
        for a in cols:
            s1 = data[a]
            for op in CROSS_OPS:
                if op['arity']==1 and (not bounded_only or op['preserves_bounded']):
                    feats[f"{op['name']}_{a}"] = op['func'](s1)

        # 二元任务
        dd = data.to_dict('series')
        tasks = [
            (op, a, b, dd)
            for op in CROSS_OPS
            if op['arity']==2 and (not bounded_only or op['preserves_bounded'])
            for a in cols for b in cols
        ]

        # 并行或串行执行
        if mode=='mpi':
            comm = MPI.COMM_WORLD
            size, rank = comm.Get_size(), comm.Get_rank()
            local = [t for i,t in enumerate(tasks) if i%size==rank]
            local_res = dict(_cross_apply(t) for t in local)
            all_res = comm.gather(local_res, root=0)
            if rank==0:
                merged = {}
                for d in all_res: merged.update(d)
                feats.update(merged)
            comm.Barrier()

        elif mode=='process':
            with ProcessPoolExecutor(max_workers=n_job) as exe:
                for name, series in tqdm(exe.map(_cross_apply, tasks),
                                          total=len(tasks), desc="🔄 cross_op (proc)"):
                    feats[name] = series

        elif mode=='thread':
            with ThreadPoolExecutor(max_workers=n_job) as exe:
                for name, series in tqdm(exe.map(_cross_apply, tasks),
                                          total=len(tasks), desc="🔄 cross_op (thread)"):
                    feats[name] = series

        else:  # single
            for t in tqdm(tasks, total=len(tasks), desc="🔄 cross_op (single)"):
                name, series = _cross_apply(t)
                feats[name] = series

        # 构造 DataFrame 并对齐
        df_new = pd.DataFrame(feats).reindex(data.index)
        df_new.index.name = 'timestamp'
        return df_new

    def evaluate_factors(
        self,
        forward_period: int,
        window: int,
        scaler: str = 'minmax',
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        # 对齐索引
        df = self.df_features.set_index('timestamp')
        price = self.df_global[self.target_col]
        returns = price.shift(-forward_period) / price - 1
        df_eval = df.join(returns.rename('forward_return')).dropna()

        # 如果数据行少于窗口，则自动缩小窗口
        win_eff = min(window, max(1, len(df_eval)-1))

        # 1) 收集 Spearman IC 和 IR
        stats: Dict[str, Dict[str, float]] = {}
        for col in df_eval.columns.drop('forward_return'):
            sp   = spearmanr(df_eval[col], df_eval['forward_return']).correlation

            roll = df_eval[col].rolling(win_eff).corr(df_eval['forward_return'])
            arr  = roll.values
            valid = arr[~np.isnan(arr)]

            if valid.size == 0:
                ir = np.nan
            else:
                pm = float(valid.mean())
                ps = float(valid.std(ddof=0))
                ir = pm/ps if ps != 0 else np.nan

            stats[col] = {'spearman_ic': sp, 'pearson_ir': ir}

        # 2) 计算 PCA 载荷并更新
        feats    = list(stats.keys())
        X_sub    = (df[feats] - df[feats].mean()) / df[feats].std()
        pca      = PCA(n_components=1); pca.fit(X_sub)
        loadings = np.abs(pca.components_[0])
        for f, ld in zip(feats, loadings):
            stats[f]['pca_coeff'] = float(ld)

        # 3) 一次性构建 summary
        summary = pd.DataFrame.from_dict(stats, orient='index')

        # 4) 归一化 & 合成
        for m in ['spearman_ic','pearson_ir','pca_coeff']:
            arr = summary[m].values
            valid = arr[~np.isnan(arr)]

            if valid.size == 0:
                summary[f'{m}_norm'] = np.nan
            elif scaler=='minmax':
                mi = float(np.nanmin(valid))
                ma = float(np.nanmax(valid))
                summary[f'{m}_norm'] = 0.0 if ma==mi else (summary[m]-mi)/(ma-mi)
            else:
                mu = float(np.nanmean(valid))
                sd = float(np.nanstd(valid, ddof=0))
                summary[f'{m}_norm'] = 0.0 if sd==0 else (summary[m]-mu)/sd

        summary['combined_score'] = (
            summary['spearman_ic_norm'] +
            summary['pearson_ir_norm'] +
            summary['pca_coeff_norm']
        )

        summary = summary.sort_values('combined_score', ascending=False)
        if top_k:
            summary = summary.head(top_k)

        # 保留小数
        summary = summary.round(self.decimals)

        self.summary = summary
        return summary

    def get_summary(self, sort_by='combined_score', ascending=False) -> pd.DataFrame:
        return self.summary.sort_values(sort_by, ascending=ascending)

    def next(
        self,
        steps: int = 1,
        k: int = 1,
        mode: str = 'single',
        n_job: Optional[int] = None,
        bounded_only: bool = False
    ) -> pd.DataFrame:
        for _ in tqdm(range(steps), desc="🔄 next steps"):
            features = list(self.summary.index)
            mat = self.df_features.set_index('timestamp')[features].values
            corr = np.abs(np.corrcoef(mat.T))
            idx_map = {f: i for i, f in enumerate(features)}

            kept = [features.pop(0)]
            while len(kept) < k and features:
                max_corrs = [
                    max(corr[idx_map[f], idx_map[kf]] for kf in kept)
                    for f in features
                ]
                pos = int(np.argmin(max_corrs))
                kept.append(features.pop(pos))

            sub_df = self.df_features.set_index('timestamp')[kept]
            newf   = self.cross_op(sub_df, mode, n_job, bounded_only)
            merged = pd.concat([sub_df, newf], axis=1)
            drop_cols = merged.nunique()[merged.nunique() <= 1].index.tolist()
            merged = merged.drop(columns=drop_cols).dropna()

            self.df_features = merged.reset_index()
            self.df_features = self.df_features.round(self.decimals)
            self.evaluate_factors(**self._eval_kwargs)

        return self.df_features

    def visualize_structure_2d(self, *args, **kwargs) -> None:
        pass