import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import umap
import warnings

from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from sklearn.impute import SimpleImputer
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
    MeanShift,
    estimate_bandwidth
)
from sklearn.mixture import GaussianMixture

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


# ================ Pickle Function ================

def _cross_apply (args):
    op, a, b, data_dict = args
    name = f"{a}_{op['name']}_{b}"
    return name, op['func'] (data_dict[a], data_dict[b])


def _evaluate_clustering_task (algo_name, Est, params, X, metrics):
    """
    顶级函数：评估单个算法+参数组合。
    返回 dict 或 None。
    """
    # 限制簇数不超过样本数-1
    if 'n_clusters' in params and params['n_clusters'] > X.shape[0] - 1:
        return None
    try:
        with warnings.catch_warnings ():
            warnings.simplefilter ('error', UserWarning)
            warnings.simplefilter ('error', ConvergenceWarning)
            if Est is MeanShift and params.get ('bandwidth') == 'estimate':
                bw = estimate_bandwidth (X)
                model = Est (bandwidth=bw)
            else:
                model = Est (**params)
            labels = model.fit_predict (X) if hasattr (model, 'fit_predict') else model.fit (X).predict (X)
        if len (set (labels)) < 2:
            return None
        rec = {'algo': algo_name}
        rec.update (params)
        if 'silhouette' in metrics:
            rec['silhouette'] = silhouette_score (X, labels)
        if 'calinski_harabasz' in metrics:
            rec['calinski_harabasz'] = calinski_harabasz_score (X, labels)
        if 'davies_bouldin' in metrics:
            rec['davies_bouldin'] = davies_bouldin_score (X, labels)
        rec['_labels'] = labels
        return rec
    except (UserWarning, ConvergenceWarning):
        return None


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
        self.cluster_report = pd.DataFrame ()

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
        imputer = SimpleImputer(strategy='mean')
        X_sub = imputer.fit_transform (X_sub)
        pca = PCA (n_components=1)
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
        if top_k:  # 只保留 top k 个因子
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
            random_state: int = 42,
            pca_evp: Optional[float] = 0.90,
            umap_components: Optional[int] = 2,
            n_jobs: int = None
    ) -> None:
        """
        一次性绘制 6 种二维降维散点图，按 self.target_col 的前向涨跌着色。
        支持多线程并行计算降维结果，进度条覆盖滑窗展开和并行降维阶段。

        参数：
          - seq_len: …
          - perplexity: …
          - n_neighbors: …
          - random_state: …
          - pca_evp:    PCA 保留方差比例（0< pca_evp <=1），或 None 跳过该方案；
          - umap_components: UMAP→t-SNE 首步输出维度（整型），或 None 跳过该方案；
          - n_jobs:     并行线程数…
        """

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
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
        labels = (aligned_forward.iloc[seq_len - 1: seq_len - 1 + N] >= 0)
        labels = labels.astype (int).to_numpy ()

        # ——— 5. 定义降维任务列表 ———
        reducers = [
            ('PCA-2', PCA (n_components=2, random_state=random_state)),
            ('LLE', LocallyLinearEmbedding (n_neighbors=n_neighbors, n_components=2, random_state=random_state)),
            ('t-SNE', TSNE (n_components=2, perplexity=perplexity, random_state=random_state)),
            (f'PCA({pca_evp * 100:.2f}%)→t-SNE', pca_evp),
            ('UMAP-2', umap.UMAP (n_neighbors=n_neighbors, n_components=2)),
            (f'UMAP({umap_components})→t-SNE', umap_components),
        ]
        results = {}
        # 并行计算降维
        n_workers = n_jobs or os.cpu_count ()

        def _compute (item):
            name, algo = item
            if name == f'PCA({pca_evp * 100:.2f}%)→t-SNE':
                V_pca90 = PCA (n_components=algo, random_state=random_state).fit_transform (V)
                Z = TSNE (n_components=2, perplexity=perplexity, random_state=random_state).fit_transform (V_pca90)
            elif name == f'UMAP({umap_components})→t-SNE':
                V_umap_components = umap.UMAP (
                    n_neighbors=n_neighbors,
                    n_components=algo,
                ).fit_transform (V)
                Z = TSNE (n_components=2, perplexity=perplexity, random_state=random_state).fit_transform (
                    V_umap_components)
            else:
                Z = algo.fit_transform (V)
            return name, Z

        with ThreadPoolExecutor (max_workers=n_workers) as executor:
            futures = {executor.submit (_compute, item): item for item in reducers}
            for fut in tqdm (as_completed (futures), total=len (futures), desc="🔄 并行降维"):
                name, Z = fut.result ()
                results[name] = Z

        # ——— 6. 绘图（2x3 网格，使用 constrained_layout） ———
        fig, axes = plt.subplots (2, 3, figsize=(18, 10), constrained_layout=True)
        cm = plt.cm.coolwarm

        for ax, (title, Z) in zip (axes.flatten (), results.items ()):
            sc = ax.scatter (Z[:, 0], Z[:, 1], c=labels, cmap=cm, s=10, alpha=0.7)
            ax.set_title (title, fontsize=12)
            ax.set_xlabel ('Component 1')
            ax.set_ylabel ('Component 2')

        cbar = fig.colorbar (sc, ax=axes.ravel ().tolist (), ticks=[0, 1])
        cbar.ax.set_yticklabels (['Down', 'Up'])

        fig.suptitle (f'2D Structure Visualization (seq_len={seq_len})', fontsize=14)
        plt.show ()

    def evaluate_clusterings (
            self,
            algos: Optional[list] = None,
            param_grids: Optional[Dict[str, Dict[str, list]]] = None,
            metrics: Optional[list] = None,
            dim_reduction: str = 'none',
            reduction_params: Optional[Dict[str, Any]] = None,
            seq_len: int = 1,
            n_jobs: int = 1,
            backend: str = 'thread',
    ) -> pd.DataFrame:
        """
        对多种聚类算法和参数组合进行评估，并将最佳聚类标签存入 self.df_features['cluster']。

        参数:
          - algos: Optional[Dict[str, Estimator]]，算法映射，键为算法名称字符串，
            值为聚类器类；支持的算法名称包括：
            'KMeans', 'AgglomerativeClustering', 'SpectralClustering',
            'GaussianMixture', 'DBSCAN', 'MeanShift'。
            None 表示使用所有默认算法。

          - param_grids: Optional[Dict[str, Dict[str, list]]]，超参数网格，
            键与 algos 中算法名称对应，值为 {参数名: 候选值列表}。
            None 表示使用默认网格：
              • KMeans: {'n_clusters': [2..10], 'random_state': [0]}
              • AgglomerativeClustering: {'n_clusters': [2..10], 'linkage': ['ward','average','complete']}
              • SpectralClustering: {'n_clusters': [2..10], 'affinity': ['rbf','nearest_neighbors'], 'random_state': [0]}
              • GaussianMixture: {'n_components': [2..10], 'covariance_type': ['full'], 'random_state': [0]}
              • DBSCAN: {'eps': [0.1,0.2,0.5], 'min_samples': [5,10,20]}
              • MeanShift: {'bandwidth': ['estimate']} (自动估算带宽)

          - metrics: Optional[List[str]]，评估指标列表，可选：
            'silhouette', 'calinski_harabasz', 'davies_bouldin'。
            None 时使用三者全部。

          - dim_reduction: str，降维方式，
            'none'（默认，不降维）、
            'pca'（PCA 保留方差比例）、
            'umap'（UMAP 降维）。

          - reduction_params: Optional[Dict[str, Any]]，降维参数；
            当 dim_reduction='pca' 时，支持：
              {'variance_ratio': float (0<var<=1)}；
            当 dim_reduction='umap' 时，支持：
              {'n_components': int, 'n_neighbors': int, 'min_dist': float, ...}。
            None 使用算法默认参数。

          - seq_len: int，序列窗口长度，
            大于 1 时将原始 (T, d) 数据按窗口展开为
            (T-seq_len+1, seq_len*d) 样本，标签对应窗口末尾 timestamp。

          - n_jobs: int，并行 worker 数量，>0。
            backend='thread' 时启动线程池，
            backend='process' 时启动进程池。

          - backend: str，并行类型，'thread' 或 'process'。
            'thread' 适合释放 GIL 的 sklearn 算法，
            'process' 适合 CPU 密集型、纯 Python 任务。
        返回:
          - pd.DataFrame：每行对应一次算法+参数组合的评估，
            列包括 'algo', 各超参数, 'silhouette', 'calinski_harabasz', 'davies_bouldin'。
        """

        # 默认算法与参数网格
        default_algos = {
            'KMeans': KMeans,
            'AgglomerativeClustering': AgglomerativeClustering,
            'SpectralClustering': SpectralClustering,
            'GaussianMixture': GaussianMixture,
            'DBSCAN': DBSCAN,
            'MeanShift': MeanShift,
        }
        default_param_grids = {
            'KMeans': {'n_clusters': list (range (2, 11)), 'random_state': [0]},
            'AgglomerativeClustering': {'n_clusters': list (range (2, 11)), 'linkage': ['ward', 'average', 'complete']},
            'SpectralClustering': {'n_clusters': list (range (2, 11)), 'affinity': ['rbf', 'nearest_neighbors'],
                                   'random_state': [0]},
            'GaussianMixture': {'n_components': list (range (2, 11)), 'covariance_type': ['full'], 'random_state': [0]},
            'DBSCAN': {'eps': [0.1, 0.2, 0.5], 'min_samples': [5, 10, 20]},
            'MeanShift': {'bandwidth': ['estimate']}
        }
        # 解析 algos
        if algos is None:
            to_run = default_algos
        elif isinstance (algos, list):
            to_run = {name: default_algos[name] for name in algos if name in default_algos}
        else:
            raise ValueError ("`algos` must be None or list of algorithm names")
        # 解析 param_grids
        if param_grids is None:
            grids = default_param_grids
        else:
            grids = {name: param_grids.get (name, default_param_grids[name]) for name in to_run}
        metrics = metrics or ['silhouette', 'calinski_harabasz', 'davies_bouldin']

        # 构造 X 和时间索引
        df = self.df_features.drop (columns=['cluster'], errors='ignore').copy ()
        # 仅保留因子特征列，排除 timestamp 和 cluster（防止 NaN 引入）
        feature_cols = [c for c in df.columns if c not in ('timestamp', 'cluster')]
        X0 = df[feature_cols].values
        times = df['timestamp'].tolist ()
        if seq_len > 1:
            T, d = X0.shape
            N = T - seq_len + 1
            X = np.vstack ([X0[i:i + seq_len].flatten () for i in range (N)])
            time_index = [times[i + seq_len - 1] for i in range (N)]
        else:
            X = X0.copy ()
            time_index = times

        # 标准化
        scaler = (MinMaxScaler () if self._eval_kwargs['scaler'] == 'minmax'
                  else RobustScaler () if self._eval_kwargs['scaler'] == 'robust'
        else StandardScaler ())

        X = scaler.fit_transform (X)

        # 降维
        reduction_params = reduction_params or {}
        if dim_reduction == 'pca':
            vr = reduction_params.get ('variance_ratio', 1.0)
            pca = PCA (n_components=vr, svd_solver='full', random_state=0)
            X = pca.fit_transform (X)
        elif dim_reduction == 'umap':
            n_c = reduction_params.get ('n_components', 2)
            umap_kwargs = {k: v for k, v in reduction_params.items () if k != 'n_components'}
            reducer = umap.UMAP (n_components=n_c, **umap_kwargs)
            X = reducer.fit_transform (X)

        # 并行评估
        Executor = ThreadPoolExecutor if backend == 'thread' else ProcessPoolExecutor
        tasks = [(name, Est, params, X, metrics)
                 for name, Est in to_run.items ()
                 for params in ParameterGrid (grids.get (name, {}))]
        records = []
        with Executor (max_workers=n_jobs) as executor:
            futures = {executor.submit (_evaluate_clustering_task, *task): task for task in tasks}
            iterator = as_completed (futures)
            iterator = tqdm (iterator, total=len (futures), desc='Clustering eval')
            for fut in iterator:
                rec = fut.result ()
                if rec:
                    records.append (rec)

        df_eval = pd.DataFrame.from_records (records)
        self.cluster_eval_ = df_eval.drop (columns=['_labels'], errors='ignore')

        # 综合绘图
        plt.figure ()
        best_sil = df_eval.groupby ('algo')['silhouette'].max ()
        best_sil.plot (kind='bar')
        plt.ylabel ('Max Silhouette')
        plt.title (f'Algorithm Comparison (dim_red={dim_reduction})')
        plt.show ()

        # silhouette plot
        best_idxs = df_eval.groupby ('algo')['silhouette'].idxmax ().items ()
        items = list (best_idxs)

        # 每 3 个算法做一组
        for i in range (0, len (items), 3):
            group = items[i:i + 3]
            n = len (group)
            # 每行 n 列子图
            fig, axes = plt.subplots (1, n, figsize=(5 * n, 4), squeeze=False)
            for ax, (algo, idx) in zip (axes[0], group):
                labels = df_eval.loc[idx, '_labels']
                # 安全获取簇数
                raw_nc = df_eval.loc[idx, 'n_clusters'] if 'n_clusters' in df_eval.columns else np.nan
                if pd.notna (raw_nc):
                    n_clusters = int (raw_nc)
                else:
                    unique_labels = np.unique (labels)
                    n_clusters = int ((unique_labels != -1).sum ())
                if n_clusters < 2:
                    ax.text (0.5, 0.5, f"{algo}\n只有{n_clusters}簇，跳过",
                             ha='center', va='center')
                    ax.axis ('off')
                    continue

                sil_vals = silhouette_samples (X, labels)
                y_lower = 10
                for j in range (n_clusters):
                    ith = np.sort (sil_vals[labels == j])
                    size_j = ith.shape[0]
                    y_upper = y_lower + size_j
                    ax.fill_betweenx (
                        np.arange (y_lower, y_upper),
                        0, ith, alpha=0.7
                    )
                    y_lower = y_upper + 10
                ax.set_title (f"{algo} (clusters={n_clusters})")
                ax.set_xlabel ("Silhouette")
                ax.set_ylabel ("Cluster label")
                ax.set_yticks ([])
                ax.axvline (df_eval.loc[idx, 'silhouette'], color='red', linestyle='--')

            plt.tight_layout ()
            plt.show ()

            self.cluster_report = df_eval

        return df_eval
