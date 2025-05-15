import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import spearmanr

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
        # 全局数据准备
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
        """
        生成因子并存储到 self.df_features:
          - layers: 总层数（第1层基础；第2+层交叉）
          - include_only_bounded_factors: 第1层是否只跑 bounded individual
          - skip_base: 是否跳过第1层基础因子
          - cross_all: 是否对所有累计特征做交叉
        返回：无缺失值的 timestamp + 因子特征表
        """
        base_feats: Dict[str, pd.Series] = {}
        meta: Dict[str, Dict[str, bool]] = {}
        # 第1层基础特征
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
        # 合成 DataFrame
        feat_df = pd.DataFrame (base_feats)
        prev_feats = base_feats.copy ()
        # 多层交叉
        for layer in range (2, layers + 1):
            new_feats: Dict[str, pd.Series] = {}
            future_to_col: Dict[Any, str] = {}
            right_feats = feat_df if cross_all else base_feats
            with ThreadPoolExecutor () as exe:
                # 二元操作
                for n1, s1 in prev_feats.items ():
                    for n2, s2 in right_feats.items ():
                        for op in CROSS_OPS:
                            if op['arity'] != 2 or not op['preserves_bounded']:
                                continue
                            if not (meta[n1]['is_bounded'] and meta[n2]['is_bounded']):
                                continue
                            col = f"{n1}_{op['name']}_{n2}"
                            future_to_col[exe.submit (op['func'], s1, s2)] = col
                # 一元操作
                for n1, s1 in prev_feats.items ():
                    for op in CROSS_OPS:
                        if op['arity'] != 1 or not op['preserves_bounded']:
                            continue
                        col = f"{op['name']}_{n1}"
                        future_to_col[exe.submit (op['func'], s1)] = col
                # 收集结果
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


        # 去除含有 NaN 的行
        feat_df.insert (0, 'timestamp', feat_df.index)
        feat_df = feat_df.dropna ()

        # 删除常数因子并报告
        constant_cols = [
            c for c in feat_df.columns
            if c != 'timestamp' and feat_df[c].nunique () <= 1
        ]
        if constant_cols:
            # print (f"Removed constant factors: {constant_cols}")
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
        """
        并行评估因子有效性，保存前 top_k IC:
        forward_period: 前瞻收益期
        window: 滑动窗口大小
        top_k: 保留最优因子数量
        n_jobs: 最大线程数，None 则使用默认
        返回：筛选后的 ic_summary
        """


        df_feat = self.df_features.set_index ('timestamp')
        df_input = self.df_input.set_index ('timestamp')
        price = df_input['close'] if 'close' in df_input else df_input.iloc[:, 1]
        forward_return = price.shift (-forward_period) / price - 1
        df = df_feat.join (forward_return.rename ('forward_return')).dropna ()

        # 单列 IC 计算函数
        def calc_ic (col: str) -> Dict[str, float]:
            series = df[col]
            target = df['forward_return']
            spearman_val = spearmanr (series, target).correlation
            rolling = series.rolling (window).corr (target)
            pearson_mean = rolling.mean ()
            pearson_std = rolling.std ()
            pearson_ir = pearson_mean / pearson_std if pearson_std != 0 else np.nan
            return {
                'spearman_ic': spearman_val,
                'pearson_ic_mean': pearson_mean,
                'pearson_ic_std': pearson_std,
                'pearson_ic_ir': pearson_ir
            }

        # 并行计算
        ic_dict: Dict[str, Dict[str, float]] = {}
        cols = list (df_feat.columns)
        with ThreadPoolExecutor (max_workers=n_jobs) as exe:
            futures = {exe.submit (calc_ic, c): c for c in cols}
            for fut in tqdm (as_completed (futures), total=len (futures), desc="🔍 Evaluating IC"):
                col = futures[fut]
                ic_dict[col] = fut.result ()
        # 构建 summary 并筛选 top_k
        summary = pd.DataFrame.from_dict (ic_dict, orient='index')
        summary['combined_score'] = summary['pearson_ic_ir'].abs () + summary['spearman_ic'].abs ()
        summary = summary.sort_values ('combined_score', ascending=False).head (top_k)
        self.ic_summary = summary
        return self.ic_summary

    def get_ic_summary (
            self,
            sort_by: str = 'pearson_ic_ir',
            ascending: bool = False,
            by_abs: bool = False
    ) -> pd.DataFrame:
        """
        返回已保存的 ic_summary，按指定列排序。
        sort_by: 排序列名 ('spearman_ic', 'pearson_ic_ir', 'combined_score')
        ascending: 是否升序
        by_abs: 是否按绝对值排序
        """
        df = self.ic_summary.copy ()
        if by_abs:
            df['_key'] = df[sort_by].abs ()
            df = df.sort_values ('_key', ascending=ascending).drop (columns=['_key'])
        else:
            df = df.sort_values (sort_by, ascending=ascending)
        return df

# —— 继续在 factors.py 添加更多因子 ——
