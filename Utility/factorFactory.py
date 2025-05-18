import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from mpi4py import MPI

from Utility.registry import FACTOR_REGISTRY
from Utility.factors import *


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


def _cross_apply (args):
    op, a, b, data_dict = args
    name = f"{a}_{op['name']}_{b}"
    return name, op['func'] (data_dict[a], data_dict[b])


class FactorFactory:
    def __init__ (
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
            df.copy ()
            .sort_values ('timestamp')
            .set_index ('timestamp')
        )
        self.base_cols = (
            base_cols if base_cols is not None
            else [c for c in self.df_global.select_dtypes (include=[np.number]).columns]
        )
        self.target_col = target_col
        self._eval_kwargs = dict (
            forward_period=forward_period,
            window=window,
            scaler=scaler,
            top_k=top_k
        )
        self.decimals = decimals
        self.df_features = pd.DataFrame ()
        self.summary = pd.DataFrame ()

    def apply_registry (
            self,
            df: pd.DataFrame,
            cols: List[str],
            bounded_only: bool = False
    ) -> pd.DataFrame:
        """
        对传入的 df（必须有 timestamp 索引）按列依次执行 FACTOR_REGISTRY
        中注册的所有 func。因子函数自行命名 out[key]，这里直接汇总。
        """
        # 备份原属性
        orig_df, orig_base, orig_target = self.df_global, self.base_cols, self.target_col

        # 临时替换
        self.df_global = df.copy ()
        self.base_cols = cols.copy ()
        self.target_col = self.target_col if self.target_col in df.columns else None

        new_feats: Dict[str, pd.Series] = {}
        for prefix, info in FACTOR_REGISTRY.items ():
            tags = info.get ('tags', [])
            if bounded_only and 'bounded' not in tags:
                continue

            out = info['func'] (self, df, cols)
            for key, series in out.items ():
                # trust the key from factors.py
                new_feats[key] = pd.Series (series, index=df.index, name=key)

        # 恢复原属性
        self.df_global, self.base_cols, self.target_col = orig_df, orig_base, orig_target
        return pd.DataFrame (new_feats, index=df.index)

    def generate_factors (
            self,
            mode: str = 'single',
            n_job: Optional[int] = None,
            bounded_only: bool = False
    ) -> pd.DataFrame:
        # 1) 一阶因子
        feat_df = self.apply_registry (
            df=self.df_global,
            cols=self.base_cols,
            bounded_only=bounded_only
        )
        feat_df.index.name = 'timestamp'

        # 2) 交叉特征
        cross_df = self.cross_op (feat_df, mode, n_job, bounded_only)

        # 3) 合并 & 清洗
        merged = pd.concat ([feat_df, cross_df], axis=1)
        drop_cols = merged.nunique ()[merged.nunique () <= 1].index.tolist ()
        df_feat = merged.drop (columns=drop_cols).dropna ()

        # 4) 存储 & 评估
        self.df_features = df_feat.reset_index ().round (self.decimals)
        self.evaluate_factors (**self._eval_kwargs)
        return self.df_features

    def cross_op (
            self,
            input_df: pd.DataFrame,
            mode: str = 'single',
            n_job: Optional[int] = None,
            bounded_only: bool = False
    ) -> pd.DataFrame:
        data = input_df.copy ()
        feats: Dict[str, pd.Series] = {}

        # —— 一阶因子 on 子集 ——
        reg_df = self.apply_registry (
            df=data,
            cols=list (data.columns),
            bounded_only=bounded_only
        )
        feats.update (reg_df.to_dict ('series'))

        # —— 一元算子 ——
        for col in data.columns:
            for op in CROSS_OPS:
                if op['arity'] == 1 and (not bounded_only or op['preserves_bounded']):
                    feats[f"{op['name']}_{col}"] = op['func'] (data[col])

        # —— 二元算子构造任务列表 ——
        dd = data.to_dict ('series')
        tasks = [
            (op, a, b, dd)
            for op in CROSS_OPS
            if op['arity'] == 2 and (not bounded_only or op['preserves_bounded'])
            for a in data.columns for b in data.columns
        ]

        # 并行或串行执行
        if mode == 'mpi':
            comm = MPI.COMM_WORLD
            size, rank = comm.Get_size (), comm.Get_rank ()
            local = [t for i, t in enumerate (tasks) if i % size == rank]
            local_res = dict (_cross_apply (t) for t in local)
            all_res = comm.gather (local_res, root=0)
            if rank == 0:
                merged = {}
                for d in all_res: merged.update (d)
                feats.update (merged)
            comm.Barrier ()

        elif mode == 'process':
            with ProcessPoolExecutor (max_workers=n_job) as exe:
                for name, series in tqdm (exe.map (_cross_apply, tasks),
                                          total=len (tasks), desc="🔄 cross_op (proc)"):
                    feats[name] = series

        elif mode == 'thread':
            with ThreadPoolExecutor (max_workers=n_job) as exe:
                for name, series in tqdm (exe.map (_cross_apply, tasks),
                                          total=len (tasks), desc="🔄 cross_op (thread)"):
                    feats[name] = series

        else:  # single
            for t in tqdm (tasks, total=len (tasks), desc="🔄 cross_op (single)"):
                name, series = _cross_apply (t)
                feats[name] = series

        df_new = pd.DataFrame (feats).reindex (data.index)
        df_new.index.name = 'timestamp'
        return df_new

    def evaluate_factors (
            self,
            forward_period: int,
            window: int,
            scaler: str = 'minmax',
            top_k: Optional[int] = None
    ) -> pd.DataFrame:
        df = self.df_features.set_index ('timestamp')
        price = self.df_global[self.target_col]
        returns = price.shift (-forward_period) / price - 1
        df_eval = df.join (returns.rename ('forward_return')).dropna ()

        win_eff = min (window, max (1, len (df_eval) - 1))

        stats: Dict[str, Dict[str, float]] = {}
        for col in df_eval.columns.drop ('forward_return'):
            raw_sp = spearmanr (df_eval[col], df_eval['forward_return']).correlation
            sp = float (np.atleast_1d (raw_sp).ravel ()[0])

            roll = df_eval[col].rolling (win_eff).corr (df_eval['forward_return'])
            valid = roll.dropna ().values
            ir = float (valid.mean () / valid.std (ddof=0)) if valid.size else np.nan

            stats[col] = {'spearman_ic': sp, 'pearson_ir': ir}

        X_sub = (df[list (stats)] - df[list (stats)].mean ()) / df[list (stats)].std ()
        pca = PCA (n_components=1);
        pca.fit (X_sub)
        for f, ld in zip (stats.keys (), np.abs (pca.components_[0])):
            stats[f]['pca_coeff'] = float (ld)

        summary = pd.DataFrame.from_dict (stats, orient='index')
        for m in ['spearman_ic', 'pearson_ir', 'pca_coeff']:
            arr = summary[m].to_numpy (dtype=float)
            if scaler == 'minmax':
                mi, ma = arr.min (), arr.max ()
                summary[f'{m}_norm'] = (summary[m] - mi) / (ma - mi) if ma != mi else 0.0
            else:
                mu, sd = arr.mean (), arr.std (ddof=0)
                summary[f'{m}_norm'] = (summary[m] - mu) / sd if sd != 0 else 0.0

        summary['combined_score'] = (
                summary['spearman_ic_norm']
                + summary['pearson_ir_norm']
                + summary['pca_coeff_norm']
        )
        summary = summary.sort_values ('combined_score', ascending=False)
        if top_k: # 只保留 top k 个因子
            summary = summary.head (top_k)
            cols_to_keep = ['timestamp'] + summary.index.tolist ()
            self.df_features = self.df_features[cols_to_keep]
        self.summary = summary.round (self.decimals)
        return self.summary

    def get_summary (self, sort_by='combined_score', ascending=False) -> pd.DataFrame:
        return self.summary.sort_values (sort_by, ascending=ascending)

    def next (
            self,
            steps: int = 1,
            k: int = 1,
            mode: str = 'single',
            n_job: Optional[int] = None,
            bounded_only: bool = False
    ) -> pd.DataFrame:
        for _ in tqdm (range (steps), desc="🔄 next steps"):
            features = list (self.summary.index)
            mat = self.df_features.set_index ('timestamp')[features].values
            corr = np.abs (np.corrcoef (mat.T))
            idx_map = {f: i for i, f in enumerate (features)}

            kept = [features.pop (0)]
            while len (kept) < k and features:
                max_corrs = [
                    max (corr[idx_map[f], idx_map[kf]] for kf in kept)
                    for f in features
                ]
                kept.append (features.pop (int (np.argmin (max_corrs))))

            sub_df = self.df_features.set_index ('timestamp')[kept]
            newf = self.cross_op (sub_df, mode, n_job, bounded_only)
            merged = pd.concat ([sub_df, newf], axis=1)
            drop_cols = merged.nunique ()[merged.nunique () <= 1].index.tolist ()
            merged = merged.drop (columns=drop_cols).dropna ()

            self.df_features = merged.reset_index ().round (self.decimals)
            self.evaluate_factors (**self._eval_kwargs)

        return self.df_features

    def visualize_structure_2d (
            self,
            seq_len: int,
            perplexity: float = 30.0,
            n_neighbors: int = 10,
            random_state: int = 42
    ) -> None:
        """
        一次性绘制 4 种二维降维散点图，按 self.target_col 的前向涨跌着色。
        进度条覆盖滑窗展开和各算法降维两个阶段。

        参数：
          - seq_len:      滑窗长度；
          - perplexity:   t-SNE 的 perplexity；
          - n_neighbors:  LLE 的邻居数量；
          - random_state: 随机种子。
        """

        # ——— 1. 提取已生成的特征矩阵 & 时间索引 ———
        df_feat = self.df_features.copy ()
        timestamps = df_feat['timestamp']
        X = df_feat.drop (columns=['timestamp']).values  # (T, d)
        T, d = X.shape

        # ——— 2. 计算前向收益 & 对齐标签 ———
        fp = self._eval_kwargs['forward_period']
        forward_ret = (
                self.df_global[self.target_col].shift (-fp)
                / self.df_global[self.target_col]
                - 1
        )
        aligned_forward = forward_ret.reindex (timestamps).reset_index (drop=True)

        # ——— 3. 滑窗展开 & 标准化（带进度条） ———
        N = T - seq_len + 1
        V = np.empty ((N, seq_len * d), dtype=float)
        for i in tqdm (range (N), desc="🔄 窗口展平"):
            V[i, :] = X[i:i + seq_len, :].flatten ()
        V = (V - V.mean (axis=0)) / (V.std (axis=0, ddof=0) + 1e-8)

        # ——— 4. 生成涨跌标签 ———
        labels = (aligned_forward.iloc[seq_len - 1: seq_len - 1 + N] >= 0).astype (int).to_numpy ()

        # ——— 5. 四种降维方法（带进度条） ———
        reducers = [
            ('PCA-2', PCA (n_components=2, random_state=random_state)),
            ('LLE', LocallyLinearEmbedding (
                n_neighbors=n_neighbors,
                n_components=2,
                random_state=random_state)),
            ('t-SNE', TSNE (
                n_components=2,
                perplexity=perplexity,
                random_state=random_state)),
            # 这里直接存一个 float，表示「保留 90%」的阈值
            ('PCA(90%)→t-SNE', 0.90)
        ]
        results = {}
        for name, algo in tqdm (reducers, desc="🔄 降维算法"):
            if name == 'PCA(90%)→t-SNE':
                # 先 PCA 保留 90%，再 t-SNE
                V_pca90 = PCA (n_components=algo, random_state=random_state).fit_transform (V)
                Z = TSNE (n_components=2, perplexity=perplexity, random_state=random_state).fit_transform (V_pca90)
            else:
                Z = algo.fit_transform (V)
            results[name] = Z

        # ——— 6. 绘图（使用 constrained_layout） ———
        fig, axes = plt.subplots (
            2, 2,
            figsize=(12, 10),
            constrained_layout=True
        )
        cm = plt.cm.coolwarm

        for ax, (title, Z) in zip (axes.flatten (), results.items ()):
            sc = ax.scatter (
                Z[:, 0], Z[:, 1],
                c=labels,
                cmap=cm,
                s=10,
                alpha=0.7
            )
            ax.set_title (title, fontsize=12)
            ax.set_xlabel ('Component 1')
            ax.set_ylabel ('Component 2')

        cbar = fig.colorbar (sc, ax=axes.ravel ().tolist (), ticks=[0, 1])
        cbar.ax.set_yticklabels (['Down', 'Up'])

        fig.suptitle (f'2D Structure Visualization (seq_len={seq_len})', fontsize=14)
        plt.show ()
