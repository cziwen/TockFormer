import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import warnings
import os
import gc

from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
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

from Utility.factors import CROSS_OPS
from Utility.registry import FACTOR_REGISTRY
from Utility.IOcache import dump_dataframe, load_dataframe, wash, delete_parquet_files

from Utility.Util import clean_and_interpolate


# ================ Pickle Function ================

def _evaluate_factor_task_streaming (cache_dir: str, fname: str, win_eff: int, fr: pd.Series):
    """
    Module-level helper to compute Spearman IC and Pearson IR for a single factor column.
    """
    path = os.path.join (cache_dir, fname)
    df = pd.read_parquet (path)
    col = df.columns[0]
    series = df[col]

    # Align and drop NaNs
    series = series.reindex (fr.index)
    sub = pd.DataFrame ({col: series, 'forward_return': fr}).dropna ()

    # Spearman IC
    raw_sp = spearmanr (sub[col], sub['forward_return']).correlation
    sp = float (np.atleast_1d (raw_sp).ravel ()[0])

    # 计算 Rolling IR
    # rolling_corr 里可能会出现许多 NaN
    roll_corr = sub[col].rolling (win_eff).corr (sub['forward_return'])

    # 把 NaN 去掉，只保留有效数值
    valid_roll = roll_corr.dropna ().values

    # 如果没有有效数值，或者有效数值中标准差为 0，则 IR 设为 NaN
    if valid_roll.size == 0:
        ir = np.nan
    else:
        # 过滤后依然有值，计算均值和标准差
        mean_roll = np.nanmean (valid_roll)
        std_roll = np.nanstd (valid_roll, ddof=0)

        if std_roll == 0:
            ir = np.nan
        else:
            ir = float (mean_roll / std_roll)

    return col, sp, ir


# def _compute_norm_eval_factor (metric: str, arr: np.ndarray, scaler: str):
#     """
#     返回 (metric, norm_arr)，对全 NaN 或常数情况做保护。
#     """
#     # 先检测整列是否全是 NaN
#     if np.all (np.isnan (arr)):
#         return metric, np.zeros_like (arr)
#
#     if scaler == 'minmax':
#         # 跳过 NaN，计算 min/max
#         mi = np.nanmin (arr)
#         ma = np.nanmax (arr)
#         # 如果最大最小相等，或都为 NaN（nanmin/nanmax 已跳过），填 0
#         if ma == mi:
#             norm = np.zeros_like (arr)
#         else:
#             norm = (arr - mi) / (ma - mi)
#     else:
#         # 跳过 NaN，计算 mean/std
#         mu = np.nanmean (arr)
#         sd = np.nanstd (arr, ddof=0)
#         # 如果 std 为 0，填 0
#         if sd == 0:
#             norm = np.zeros_like (arr)
#         else:
#             norm = (arr - mu) / sd
#
#     # 最后把任何可能残留的 NaN（理论上不应该有）也填成 0
#     norm = np.where (np.isnan (norm), 0.0, norm)
#     return metric, norm

def _compute_norm_eval_factor (
        scale_metric: str,
        metric: str,
        arr: np.ndarray
):
    """
    对 arr 做归一化/标准化，并处理 NaN/常数列。
    支持三种 scale_metric：
      - 'minmax'       : 普通 MinMaxScaler([min,max] → [0,1])
      - 'abs_minmax'   : 先取绝对值再 MinMaxScaler([|min|,|max|] → [0,1])
      - 其它 (e.g. 'z') : StandardScaler (z-score)

    Returns
    -------
    metric : str
        原 metric 名称
    norm_arr : np.ndarray
        与 arr 等长的归一化结果，NaN/常数列全部映射为 0。
    """
    # 1) NaN => 0
    arr_filled = np.where (np.isnan (arr), 0.0, arr).reshape (-1, 1)

    # 2) 选取要缩放的数据
    if scale_metric == 'minmax':
        arr_to_scale = arr_filled
        scaler = MinMaxScaler ()
    elif scale_metric == 'abs_minmax':
        arr_to_scale = np.abs (arr_filled)
        scaler = MinMaxScaler ()
    else:
        arr_to_scale = arr_filled
        scaler = StandardScaler ()

    # 3) fit-transform，并在异常时回退全 0
    try:
        scaled = scaler.fit_transform (arr_to_scale)
    except Exception:
        scaled = np.zeros_like (arr_to_scale)

    # 4) 返回
    return metric, scaled.ravel ()


# def _compute_one (df_sub: pd.DataFrame, col: str, win_eff: int):
#     raw_sp = spearmanr (df_sub[col], df_sub['forward_return']).correlation
#     sp = float (np.atleast_1d (raw_sp).ravel ()[0])
#
#     roll = df_sub[col].rolling (win_eff).corr (df_sub['forward_return']).dropna ().values
#     # tqdm.write(f"roll.shape = {roll.shape}")
#
#     if roll.size == 0:
#         return col, sp, np.nan
#
#     # 再检查一下：如果全 NaN（dropna 后不可能），或所有值都一样，直接 NaN
#     if np.nanstd (roll, ddof=0) == 0:
#         return col, sp, np.nan
#
#     # 用 nan‐safe 版本
#     mean = np.nanmean (roll)
#     std = np.nanstd (roll, ddof=0)
#     ir = float (mean / std) if std > 0 else np.nan
#     return col, sp, ir


def _cross_apply (args):
    op, a, b, arr_a, arr_b = args
    name = f"{a}_{op['name']}_{b}"
    return name, op['func'] (arr_a, arr_b)


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
            scaler: str = 'abs_minmax',  # minmax
            top_k: Optional[int] = None,
            decimals: int = 6,
            n_jobs: Optional[int] = None,
            use_disk_cache: bool = True,
            cache_dir: str = './tmp',
    ):
        # === 多线程配置 ===
        if n_jobs is None or n_jobs < 1:
            n_jobs = os.cpu_count ()
        self.n_jobs = n_jobs

        # === 磁盘缓存配置 ===
        self.use_disk_cache = use_disk_cache
        self.cache_dir = cache_dir

        # 原始数据按 timestamp 索引
        self.df_global = (
            df.copy ()
            .sort_values ('timestamp')
            .set_index ('timestamp')
        )
        self.df_global_path = f'{cache_dir}/df_global'
        dump_dataframe (self.df_global, self.df_global_path, clear=True)  # 磁盘化

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

        self.df_features_path = f'{cache_dir}/df_features'
        self.df_features = pd.DataFrame ()
        dump_dataframe (self.df_features, self.df_features_path, clear=True)

        self.summary = pd.DataFrame ()

        self.cluster_report = pd.DataFrame ()

        del self.df_global

    def apply_registry (
            self,
            df: pd.DataFrame,
            cols: List[str],
            bounded_only: bool = False,
    ) -> pd.DataFrame:
        """
        对传入的 df（必须有 timestamp 索引），按列并行执行 FACTOR_REGISTRY 中的 func，
        每个线程只负责一个 (prefix, col) 组合。
        不会dump，需要手动dump
        """
        new_feats: Dict[str, pd.Series] = {}
        tasks = []
        for prefix, info in FACTOR_REGISTRY.items ():
            tags = info.get ('category', [])
            if not isinstance (tags, list):
                tags = [tags]
            if bounded_only and 'bounded' not in tags:
                continue
            # 对每个 col 单独提交一个任务
            for col in cols: \
                    tasks.append ((prefix, info['func'], col))

        # 多进程执行
        with ProcessPoolExecutor (max_workers=self.n_jobs) as executor:
            future_to_task = {
                executor.submit (func, df[[col]], [col]): (prefix, col)
                for prefix, func, col in tasks
            }
            for fut in tqdm (as_completed (future_to_task), total=len (future_to_task), desc='Applying Factors'):
                out = fut.result ()
                # 将结果收集到 new_feats
                for key, series in out.items ():
                    new_feats[key] = pd.Series (series, index=df.index, name=key)

        reg_df = pd.DataFrame (new_feats, index=df.index)
        reg_df = clean_and_interpolate (reg_df, tol=0.01)
        return reg_df

    def generate_factors (self, mode: str = 'single', bounded_only: bool = False):
        """
        Generate and evaluate uni- and cross-features, with optional disk caching.
        """

        # 读取原始df
        self.df_global = load_dataframe (self.df_global_path)
        reg_df = self.apply_registry (self.df_global, self.base_cols, bounded_only)
        del self.df_global

        # Compute cross features
        self._get_cross_features (reg_df, mode, bounded_only)

        # clean
        wash (self.df_features_path)

        # Store and evaluate
        self.evaluate_factors (**self._eval_kwargs)

    def _get_cross_features (self, feat_df: pd.DataFrame, mode: str, bounded_only: bool):
        """
        Compute second-order (cross) features, optionally using disk cache.
        """
        self.cross_op (feat_df, mode, bounded_only)

    def cross_op (
            self,
            input_df: pd.DataFrame,
            mode: str = 'single',
            bounded_only: bool = False
    ) -> pd.DataFrame:
        data = input_df.copy ()
        feats: Dict[str, pd.Series] = {}
        file_map: Dict[str, str] = {}

        # —— 一阶因子 ——
        reg_df = self.apply_registry (
            df=data,
            cols=list (data.columns),
            bounded_only=bounded_only,
        )

        if self.use_disk_cache:
            dump_dataframe (reg_df, self.df_features_path, clear=True)  # 磁盘化
            del reg_df
        else:
            feats.update (reg_df.to_dict ('series'))

        # —— 一元算子 ——
        unary_tasks = []
        for col in data.columns:
            for op in CROSS_OPS:
                if op['arity'] == 1 and (not bounded_only or op['preserves_bounded']):
                    name = f"{op['name']}_({col})"
                    func = op['func']
                    arr = data[col].values
                    unary_tasks.append ((name, func, arr))

        def _process_unary (task):
            name, func, arr = task
            series = pd.Series (func (arr), name=name)
            series.index = data.index
            return name, series

        compute_pool = ThreadPoolExecutor (max_workers=self.n_jobs)

        compute_futs = {compute_pool.submit (_process_unary, t): t for t in unary_tasks}

        unary_futures: Dict[str, pd.Series] = {}
        for fut in tqdm (as_completed (compute_futs), total=len (compute_futs), desc='🔄 Unary op'):
            name, series = fut.result ()
            unary_futures[name] = series

        dump_dataframe (pd.DataFrame (unary_futures, index=data.index), self.df_features_path)
        del unary_futures

        # 等待所有线程池任务完成 (取消)
        compute_pool.shutdown (wait=True)
        gc.collect ()

        # —— 二元算子构造任务列表 ——
        dd = data.to_dict ('series')
        tasks = [
            (op, a, b, dd[a].values, dd[b].values)
            for op in CROSS_OPS
            if op['arity'] == 2 and (not bounded_only or op['preserves_bounded'])
            for a in data.columns for b in data.columns
        ]

        if mode == 'thread':
            batch_size = (self.n_jobs or os.cpu_count ()) * 4
            idx = 0
            total = len (tasks)
            pbar = tqdm (total=total, desc="🔄 cross op (thread)")
            pending = set ()
            io_futures = set ()

            # 用于批量写盘的缓存和线程池
            io_pool = ThreadPoolExecutor (max_workers=self.n_jobs)
            with ThreadPoolExecutor (max_workers=self.n_jobs) as executor:
                # 初始提交 batch_size 个计算任务
                while idx < total and len (pending) < batch_size:
                    pending.add (executor.submit (_cross_apply, tasks[idx]))
                    idx += 1

                # 循环，直到所有计算任务完成
                while pending:
                    done, pending = wait (pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        name, series = fut.result ()
                        series.index = data.index
                        if self.use_disk_cache:
                            path = os.path.join (self.df_features_path, f"{name}.parquet")
                            # 提交写盘
                            io_futures.add (io_pool.submit (
                                lambda s=series, p=path, n=name: s.to_frame (n).to_parquet (p)
                            ))
                            file_map[name] = path
                            del series
                        else:
                            feats[name] = series
                        pbar.update (1)
                        # 保持滑动窗口大小
                        if idx < total:
                            pending.add (executor.submit (_cross_apply, tasks[idx]))
                            idx += 1

                        # 如果写盘队列过长，就等最先完成的一批回来
                        if len (io_futures) >= batch_size:
                            done_io, io_futures = wait (io_futures, return_when=FIRST_COMPLETED)

                        # 如果达到一个批次，或计算全完成，回收垃圾
                        if idx % batch_size == 0 or not pending:
                            gc.collect ()
            pbar.close ()
            io_pool.shutdown (wait=True)


        else:  # single
            for t in tqdm (tasks, total=len (tasks), desc="🔄 cross_op (single)"):
                name, series = _cross_apply (t)
                feats[name] = series

        if self.use_disk_cache:
            return pd.DataFrame ()  # 占位符

        df_new = pd.DataFrame (feats).reindex (data.index)
        df_new.index.name = 'timestamp'
        return df_new

    def evaluate_factors (
            self,
            forward_period: int,
            window: int,
            scaler: str = 'minmax',
            top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        evaluate_factors：按需加载每列 Parquet，
        先筛选 top-k 个再做pca。
        """
        # 1) 只加载一次：目标价格列，计算前向收益
        price_df = load_dataframe (
            cache_dir=self.df_global_path,  # 存放 df_global 的目录
            n_jobs=self.n_jobs,
            columns=[self.target_col]
        )
        price = price_df[self.target_col]
        del price_df
        returns = (price.shift (-forward_period) / price - 1).dropna ()
        win_eff = min (window, max (1, len (returns) - 1))

        # 2) 并行计算 Spearman IC & IR
        factor_files = [
            f for f in os.listdir (self.df_features_path)
            if f.endswith ('.parquet')
        ]

        stats: Dict[str, Dict[str, float]] = {}
        with ProcessPoolExecutor (max_workers=self.n_jobs) as exe:
            futures = {exe.submit (_evaluate_factor_task_streaming, self.df_features_path, fn, win_eff, returns): fn for
                       fn in factor_files}
            for fut in tqdm (as_completed (futures), total=len (futures), desc='ic eval'):
                col, sp, ir = fut.result ()
                stats[col] = {'spearman_ic': sp, 'pearson_ir': ir}

        # 4) 构建 summary 表并归一化
        summary = pd.DataFrame.from_dict (stats, orient='index')
        with ProcessPoolExecutor (max_workers=self.n_jobs) as exe:
            futures = {exe.submit (_compute_norm_eval_factor, scaler, m, summary[m].to_numpy (dtype=np.float64)) for m
                       in ['spearman_ic', 'pearson_ir']}
            for fut in tqdm (futures, total=len (futures), desc='normalizing'):
                m, norm_arr = fut.result ()
                summary[f'{m}_norm'] = norm_arr

        # 打分排序
        summary['combined_score'] = (
                summary['spearman_ic_norm']
                + summary['pearson_ir_norm']
        )
        summary = summary.sort_values ('combined_score', ascending=False).round (self.decimals)

        # 5) 保留 top_k
        if top_k:
            retained = summary.head (top_k).index.tolist ()
            # 删除未保留的 Parquet 文件 (dump时自动删除不需要的)
            # to_delete = [col for col in stats.keys () if col not in retained]
            # delete_parquet_files (self.df_features_path, to_delete)
            summary = summary.loc[retained]



        # 5.1) 对齐时间+impute
        #    注意：这里要用 summary.index（因子名列表），而不是 index.names
        top_factors = summary.index.tolist ()
        df_pca = load_dataframe (
            cache_dir=self.df_features_path,
            n_jobs=self.n_jobs,
            columns=top_factors
        )

        df_pca = clean_and_interpolate(df_pca) # 对齐时间
        dump_dataframe(df_pca, self.df_features_path, clear=True) # 存储过滤对齐后的

        # 6) 对剩下的因子做 PCA
        X = df_pca.to_numpy (dtype=float)
        X_scaled = StandardScaler ().fit_transform (X)
        pca = PCA (n_components=1)
        pca.fit (X_scaled)
        coeffs = np.abs (pca.components_[0])
        # 7) 把 PCA 系数注入到 summary
        summary['pca_coeff'] = coeffs

        # 8) 对 PCA 系数做归一化
        metric, norm_arr = _compute_norm_eval_factor (
            scaler,
            'pca_coeff',
            summary['pca_coeff'].to_numpy (dtype=float)
        )
        summary[f'{metric}_norm'] = norm_arr

        # 9) 最终综合打分 & 排序
        summary['combined_score'] = (
                summary['spearman_ic_norm']
                + summary['pearson_ir_norm']
                + summary['pca_coeff_norm']
        )
        summary = summary.sort_values ('combined_score', ascending=False).round (self.decimals)

        # 存回对象并返回
        self.summary = summary
        return summary

    def get_summary (self, sort_by='combined_score', ascending=False) -> pd.DataFrame:
        return self.summary.sort_values (sort_by, ascending=ascending)

    def next (
            self,
            k: int = 1,
            mode: str = 'single',
            bounded_only: bool = False
    ):

        self.df_features = load_dataframe (self.df_features_path)
        features = list (self.summary.index)
        mat = self.df_features[features].values
        corr = np.abs (np.corrcoef (mat.T))
        idx_map = {f: i for i, f in enumerate (features)}

        kept = [features.pop (0)]
        while len (kept) < k and features:  # 在 top-k 里贪心选 k 个因子
            max_corrs = [
                max (corr[idx_map[f], idx_map[kf]] for kf in kept)
                for f in features
            ]
            kept.append (features.pop (int (np.argmin (max_corrs))))

        sub_df = self.df_features[kept] # 保留需要计算的因子
        self.cross_op (sub_df, mode, bounded_only)
        del sub_df;
        del self.df_features
        wash (self.df_features_path)

        # 对齐时间，补齐数据
        df_unaligned = load_dataframe (self.df_features_path)
        df_aligned = clean_and_interpolate(df_unaligned)
        dump_dataframe(df_aligned, self.df_features_path, clear=True)
        del df_unaligned, df_aligned

        self.evaluate_factors (**self._eval_kwargs)

    def visualize_structure_2d (
            self,
            seq_len: int,
            percentage: float = 1,
            perplexity: float = 30.0,
            n_neighbors: int = 10,
            random_state: int = 42,
            pca_evp: Optional[float] = 0.90,
            umap_components: Optional[int] = 2,
    ) -> None:
        """
        一次性绘制 6 种二维降维散点图，按 self.target_col 的前向涨跌着色。
        先把 df_feat 和 target_df 在 timestamp 上对齐，再做滑窗 & 打标签。
        """

        # ——— 1. 读取特征 & 目标，并按 timestamp 对齐索引 ———
        # 1.1 加载 df_feat，index 是时间戳
        df_feat = load_dataframe (self.df_features_path, percentage=percentage)
        df_feat = df_feat.sort_index ()

        # 1.2 加载 target_df，只保留 target 列，同样以 timestamp 为索引并排序
        target_df = load_dataframe (self.df_global_path, n_jobs=self.n_jobs, columns=[self.target_col])
        if 'timestamp' in target_df.columns:
            target_df = target_df.set_index ('timestamp')
        target_df = target_df.sort_index ()

        # 1.3 取索引交集，只保留两个 DataFrame 都存在的时间戳
        common_index = df_feat.index.intersection (target_df.index)
        if len (common_index) < seq_len:
            raise RuntimeError (f"对齐后共同的时间戳只有 {len (common_index)} 行，无法做 seq_len={seq_len} 的滑窗")

        # 1.4 只保留交集部分
        df_feat = df_feat.loc[common_index]
        target_series = target_df.loc[common_index, self.target_col]

        # ——— 2. 计算前向收益 & 生成对齐后的 forward_ret ———
        fp = self._eval_kwargs['forward_period']
        forward_ret = target_series.shift (-fp) / target_series - 1
        # 注意：末尾 fp 个索引会变为 NaN

        # ——— 3. 滑窗展开 & 标准化（带进度条） ———
        #    这里 T 是对齐后 df_feat 的行数
        X = df_feat.values  # 形状 (T, d)
        T, d = X.shape
        N = T - seq_len + 1  # 可滑出的窗口数
        if N <= 0:
            raise RuntimeError (f"T={T} 不足以做 seq_len={seq_len} 的滑窗")

        V = np.empty ((N, seq_len * d), dtype=float)
        for i in tqdm (range (N), desc="🔄 窗口展平"):
            V[i, :] = X[i:i + seq_len, :].flatten ()
        # 对每一列做 (x - mean)/std
        V = (V - V.mean (axis=0)) / (V.std (axis=0, ddof=0) + 1e-8)

        # ——— 4. 生成涨跌标签（对齐后再切片） ———
        # 第 i 个窗口对应的标签索引是： i + (seq_len - 1)
        # 所以要取 forward_ret.iloc[seq_len-1 : seq_len-1 + N]
        raw_slice = forward_ret.iloc[seq_len - 1: seq_len - 1 + N]
        # 如果对齐后 forward_ret 中这段有 NaN，需要谨慎处理：
        # 这里我们先直接 (>=0)，NaN 会被判成 False → 0
        # labels = (raw_slice >= 0).astype (int).to_numpy ()
        # 如果想跳过 NaN 样本，也可以：
        mask_valid = raw_slice.notna()
        V = V[mask_valid.values]
        labels = (raw_slice[mask_valid] >= 0).astype(int).to_numpy()

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

        def _compute (item):
            name, algo = item
            if name == f'PCA({pca_evp * 100:.2f}%)→t-SNE':
                V_pca = PCA (n_components=algo, random_state=random_state).fit_transform (V)
                Z = TSNE (n_components=2, perplexity=perplexity, random_state=random_state).fit_transform (V_pca)
            elif name == f'UMAP({umap_components})→t-SNE':
                V_umap = umap.UMAP (n_neighbors=n_neighbors, n_components=algo).fit_transform (V)
                Z = TSNE (n_components=2, perplexity=perplexity, random_state=random_state).fit_transform (V_umap)
            else:
                Z = algo.fit_transform (V)
            return name, Z

        # ——— 6. 并行计算降维 ———
        with ThreadPoolExecutor (max_workers=self.n_jobs) as executor:
            futures = {executor.submit (_compute, item): item for item in reducers}
            for fut in tqdm (as_completed (futures), total=len (futures), desc="🔄 并行降维"):
                name, Z = fut.result ()
                results[name] = Z

        # ——— 7. 绘图（2x3 网格，使用 constrained_layout） ———
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
            percentage: float = 1,
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
        df = load_dataframe (self.df_features_path, percentage=percentage)
        # 仅保留因子特征列，排除 timestamp 和 cluster（防止 NaN 引入）
        feature_cols = [c for c in df.columns if c not in ('timestamp', 'cluster')]
        X0 = df[feature_cols].values
        # times = df['timestamp'].tolist ()
        times = df.index.tolist()
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
        tasks = [(name, Est, params, X, metrics)
                 for name, Est in to_run.items ()
                 for params in ParameterGrid (grids.get (name, {}))]
        records = []

        with ThreadPoolExecutor (max_workers=self.n_jobs) as executor:
            futures = {executor.submit (_evaluate_clustering_task, *task): task for task in tasks}
            iterator = as_completed (futures)
            iterator = tqdm (iterator, total=len (futures), desc='Clustering eval')
            for fut in iterator:
                rec = fut.result ()
                if rec:
                    records.append (rec)

        df_eval = pd.DataFrame.from_records (records)

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
