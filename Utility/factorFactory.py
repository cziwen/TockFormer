import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
import matplotlib.pyplot as plt

# 导入注册器和因子定义
from Utility.registry import FACTOR_REGISTRY
from Utility.factors import *

# 定义可用的交叉/一元操作及其“安全元数据”
CROSS_OPS: List[Dict[str, Any]] = [
    {'name': 'mul', 'arity': 2, 'func': lambda x, y: x * y, 'preserves_bounded': True},
    {'name': 'minus', 'arity': 2, 'func': lambda x, y: x - y, 'preserves_bounded': True},
    {'name': 'div', 'arity': 2, 'func': lambda x, y: x.div (y.replace (0, pd.NA)), 'preserves_bounded': False},
    {'name': 'sin', 'arity': 1, 'func': lambda x: x.map (math.sin), 'preserves_bounded': True},
    {'name': 'cos', 'arity': 1, 'func': lambda x: x.map (math.cos), 'preserves_bounded': True},
]


def _spearman_row_proc (i: int, X: np.ndarray) -> (int, np.ndarray):
    xi = X[i]
    # 计算第 i 行和其它所有行的 Spearman
    row = np.array ([abs (spearmanr (xi, X[j]).correlation) for j in range (X.shape[0])])
    return i, row


class FactorFactory:
    def __init__ (
            self,
            df: pd.DataFrame,
            target_cols: Optional[List[str]] = None
    ):
        """
        初始化：
        df: 原始包含 timestamp 列的数据
        target_cols: 要生成因子的列名列表；若 None，则取所有数值型列
        """
        self.df_global = df.copy ().sort_values ('timestamp').set_index ('timestamp')
        if target_cols is None:
            self.target_cols = [
                c for c in self.df_global.select_dtypes (include=[np.number]).columns
            ]
        else:
            self.target_cols = target_cols
        self.df_input = self.df_global.reset_index ()
        self.df_features = pd.DataFrame ()
        self.ic_summary: pd.DataFrame = pd.DataFrame ()
        # 因子参数
        self.rsi_windows = [6, 10, 14]
        self.sma_windows = [5, 10, 20]
        self.ema_spans = [5, 10, 20]
        self.macd_pairs = [(12, 26), (5, 20)]
        self.bb_windows = [10, 20]
        self.bb_dev = [1.5, 2.0]

    def generate_factors (
            self,
            layers: int = 2,
            include_only_bounded_factors: bool = True,
            skip_base: bool = False,
            cross_all: bool = False
    ) -> pd.DataFrame:
        base_feats: Dict[str, pd.Series] = {}
        meta: Dict[str, Dict[str, bool]] = {}
        if skip_base:
            for col in self.target_cols:
                if col in self.df_global.columns:
                    s = self.df_global[col]
                    base_feats[col] = s
                    meta[col] = {'is_bounded': True}
        else:
            ind_methods = [info for info in FACTOR_REGISTRY.values ()
                           if 'individual' in info['tags']
                           and (not include_only_bounded_factors or 'bounded' in info['tags'])]
            adv_methods = [info for info in FACTOR_REGISTRY.values () if 'advanced' in info['tags']]
            for info in tqdm (ind_methods, desc="1️⃣ Individual factors"):
                out = info['func'] (self)
                items = ([out] if isinstance (out, pd.Series)
                         else [pd.Series (v, index=self.df_global.index, name=k)
                               for k, v in out.items ()])
                for s in items:
                    base_feats[s.name] = s
                    meta[s.name] = {'is_bounded': ('bounded' in info['tags'])}
            for info in tqdm (adv_methods, desc="2️⃣ Advanced factors"):
                out = info['func'] (self)
                items = ([out] if isinstance (out, pd.Series)
                         else [pd.Series (v, index=self.df_global.index, name=k)
                               for k, v in out.items ()])
                for s in items:
                    base_feats[s.name] = s
                    meta[s.name] = {'is_bounded': ('bounded' in info['tags'])}
        feat_df = pd.DataFrame (base_feats)
        prev_feats = base_feats.copy ()
        for layer in range (2, layers + 1):
            new_feats: Dict[str, pd.Series] = {}
            future_to_col: Dict[Any, str] = {}
            right_feats = feat_df if cross_all else base_feats
            with ThreadPoolExecutor () as exe:
                for n1, s1 in prev_feats.items ():
                    for n2, s2 in right_feats.items ():
                        for op in CROSS_OPS:
                            if op['arity'] != 2 or not op['preserves_bounded']:
                                continue
                            if not (meta[n1]['is_bounded'] and meta[n2]['is_bounded']):
                                continue
                            col = f"{n1}_{op['name']}_{n2}"
                            future_to_col[exe.submit (op['func'], s1, s2)] = col
                for n1, s1 in prev_feats.items ():
                    for op in CROSS_OPS:
                        if op['arity'] != 1 or not op['preserves_bounded']:
                            continue
                        col = f"{op['name']}_{n1}"
                        future_to_col[exe.submit (op['func'], s1)] = col
                for fut in tqdm (as_completed (future_to_col), total=len (future_to_col),
                                 desc=f"🔄 Layer {layer} cross"):
                    col = future_to_col[fut]
                    s = fut.result ()
                    if s.isna ().all ():
                        continue
                    s.name = col
                    new_feats[col] = s
                    meta[col] = {'is_bounded': True}
            prev_feats = new_feats
            if new_feats:
                feat_df = pd.concat ([feat_df, pd.DataFrame (new_feats)], axis=1)
        feat_df.insert (0, 'timestamp', feat_df.index)
        feat_df = feat_df.dropna ()
        constant_cols = [c for c in feat_df.columns if c != 'timestamp' and feat_df[c].nunique () <= 1]
        if constant_cols:
            feat_df = feat_df.drop (columns=constant_cols)
        self.df_features = feat_df.reset_index (drop=True)
        return self.df_features

    def evaluate_factors (
            self,
            forward_period: int,
            window: int,
            top_k: int,
            n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        df_feat = self.df_features.set_index ('timestamp')
        df_input = self.df_input.set_index ('timestamp')
        price = df_input['close'] if 'close' in df_input else df_input.iloc[:, 1]
        forward_return = price.shift (-forward_period) / price - 1
        df = df_feat.join (forward_return.rename ('forward_return')).dropna ()

        def calc_ic (col: str) -> Dict[str, float]:
            s = df[col];
            t = df['forward_return']
            sp = spearmanr (s, t).correlation
            roll = s.rolling (window).corr (t)
            pm = roll.mean ();
            ps = roll.std ();
            pir = pm / ps if ps != 0 else np.nan
            return {'spearman_ic': sp, 'pearson_ic_mean': pm, 'pearson_ic_std': ps, 'pearson_ic_ir': pir}

        ic_dict: Dict[str, Dict[str, float]] = {}
        cols = list (df_feat.columns)
        with ThreadPoolExecutor (max_workers=n_jobs) as exe:
            futures = {exe.submit (calc_ic, c): c for c in cols}
            for fut in tqdm (as_completed (futures), total=len (futures), desc="🔍 Evaluating IC"):
                ic_dict[futures[fut]] = fut.result ()
        summary = pd.DataFrame.from_dict (ic_dict, orient='index')
        summary['combined_score'] = summary['pearson_ic_ir'].abs () + summary['spearman_ic'].abs ()
        summary = summary.sort_values ('combined_score', ascending=False).head (top_k)
        self.ic_summary = summary
        return self.ic_summary

    def filter (
            self,
            k: int,
            corr_combine: str = 'average',
            n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        基于 IC+冗余的贪心算法选出最优的 k 个因子：
          1) 先按 |spearman_ic|+|pearson_ic_ir| 降序生成候选列表
          2) 用合并后的 Pearson/Spearman 相关矩阵，在候选中贪心挑选，
             首先取 IC 最高的那个，后续每次取与已选集合相关度最小的因子
          3) 最后对保留因子做 PCA(99%) 分析
        返回 {'kept': [...], 'dropped': [...], 'pca_n99': int}
        """
        # —— 计算相关度矩阵 ——
        df = self.df_features.drop (columns='timestamp')
        X = df.values.T  # shape (n_features, n_samples)
        n = X.shape[0]

        # Pearson
        corr_p = np.abs (np.corrcoef (X))

        # Spearman 并行
        corr_s = np.zeros ((n, n), dtype=float)
        with ProcessPoolExecutor (max_workers=n_jobs) as exe:
            futures = {exe.submit (_spearman_row_proc, i, X): i for i in range (n)}
            for fut in tqdm (as_completed (futures), total=n, desc="🔍 Spearman rows"):
                i, row = fut.result ()
                corr_s[i, :] = row
        corr_s = (corr_s + corr_s.T) / 2
        np.fill_diagonal (corr_s, 0)

        # 合并相关度
        if corr_combine == 'max':
            corr = np.maximum (corr_p, corr_s)
        else:
            corr = (corr_p + corr_s) / 2.0

        # —— 生成 IC 排序列表 ——
        # 假设 self.ic_summary 已包含 'spearman_ic' 和 'pearson_ic_ir'
        ic_df = self.ic_summary.copy ()
        ic_df['combined_ic'] = ic_df['spearman_ic'].abs () + ic_df['pearson_ic_ir'].abs ()
        # 保证顺序与 corr 矩阵行列对应
        features = list (df.columns)
        ic_df = ic_df.reindex (features)
        candidates = ic_df.sort_values ('combined_ic', ascending=False).index.tolist ()

        # —— 贪心选 k 因子 ——
        kept = []
        # 1) 先取 IC 最高的一个
        kept.append (candidates.pop (0))

        # 2) 依次挑与已选集合相关度最小的
        feature_to_idx = {f: i for i, f in enumerate (features)}
        while len (kept) < k and candidates:
            # 计算每个候选与 kept 集合的最大相关度
            max_corrs = []
            for f in candidates:
                idx = feature_to_idx[f]
                kept_idxs = [feature_to_idx[kf] for kf in kept]
                max_corrs.append (corr[idx, kept_idxs].max ())
            # 选出最大相关度最小的候选
            best_pos = int (np.argmin (max_corrs))
            best_feat = candidates.pop (best_pos)
            kept.append (best_feat)

        dropped = [f for f in features if f not in kept]

        # —— PCA 99% 方差分析 ——
        X_std = (df - df.mean ()) / df.std ()
        evr = np.cumsum (PCA ().fit (X_std).explained_variance_ratio_)
        pca_n99 = int (np.searchsorted (evr, 0.99) + 1)

        # 更新 df_features
        self.df_features = self.df_features[['timestamp'] + kept]

        return {'kept': kept, 'dropped': dropped, 'pca_n99': pca_n99}

    def get_ic_summary (
            self,
            sort_by: str = 'pearson_ic_ir',
            ascending: bool = False,
            by_abs: bool = False
    ) -> pd.DataFrame:
        df = self.ic_summary.copy ()
        if by_abs:
            df['_key'] = df[sort_by].abs ()
            df = df.sort_values ('_key', ascending=ascending).drop (columns=['_key'])
        else:
            df = df.sort_values (sort_by, ascending=ascending)
        return df

    def visualize_structure_2d (
            self,
            target_col: str,
            lle_n_neighbors: int = 10,
            tsne_perplexity: float = 30.0,
            forward: int = 1,
            random_state: int = 42
    ) -> None:
        """
        在四个子图中绘制：
          - PCA（2 维）
          - LLE（2 维）
          - t-SNE（2 维）
          - PCA(99%) + t-SNE
        点的颜色区分 target_col 的下一期正/负收益。
        """
        # 计算 return percentage
        price = self.df_input.set_index ('timestamp')[target_col]
        ret = price.pct_change ().shift (-forward)
        # 合并特征与收益
        merged = self.df_features.merge (
            ret.rename (f"{target_col}_return_pct").reset_index (),
            on='timestamp'
        )
        colors = (merged[f"{target_col}_return_pct"] > 0).astype (int)
        cmap = ListedColormap (['red', 'green'])
        # 标准化特征矩阵
        feats = merged.drop (columns=['timestamp', f"{target_col}_return_pct"])
        X = (feats - feats.mean ()) / feats.std ()
        fig, axes = plt.subplots (2, 2, figsize=(12, 10))
        # 1) PCA 2D
        pca2 = PCA (n_components=2, random_state=random_state)
        X_pca2 = pca2.fit_transform (X)
        axes[0, 0].scatter (X_pca2[:, 0], X_pca2[:, 1], c=colors, cmap=cmap, s=5, alpha=0.7)
        axes[0, 0].set_title ('PCA (2 Components)')
        # 2) LLE 2D
        lle = LocallyLinearEmbedding (n_neighbors=lle_n_neighbors, n_components=2, random_state=random_state)
        X_lle = lle.fit_transform (X)
        axes[0, 1].scatter (X_lle[:, 0], X_lle[:, 1], c=colors, cmap=cmap, s=5, alpha=0.7)
        axes[0, 1].set_title (f'LLE (n_neighbors={lle_n_neighbors})')
        # 3) t-SNE 2D
        tsne = TSNE (n_components=2, perplexity=tsne_perplexity, random_state=random_state)
        X_tsne = tsne.fit_transform (X)
        axes[1, 0].scatter (X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap=cmap, s=5, alpha=0.7)
        axes[1, 0].set_title (f't-SNE (perplexity={tsne_perplexity})')
        # 4) PCA(99%) -> t-SNE
        evr = np.cumsum (PCA ().fit (X).explained_variance_ratio_)
        n99 = int (np.searchsorted (evr, 0.99) + 1)
        pca99 = PCA (n_components=n99, random_state=random_state)
        X_pca99 = pca99.fit_transform (X)
        tsne2 = TSNE (n_components=2, perplexity=tsne_perplexity, random_state=random_state)
        X_pca99_tsne = tsne2.fit_transform (X_pca99)
        axes[1, 1].scatter (X_pca99_tsne[:, 0], X_pca99_tsne[:, 1], c=colors, cmap=cmap, s=5, alpha=0.7)
        axes[1, 1].set_title (f'PCA({n99}) + t-SNE')
        plt.tight_layout ()
        plt.show ()
# —— 继续在 Utility/factors.py 添加更多因子 ——
