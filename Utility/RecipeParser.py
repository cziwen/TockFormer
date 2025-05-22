# import re
# from typing import List, Dict, Optional
#
# # === 命名规范映射 ===
# # 基础因子对应的参数名称
# FUNC_PARAM_MAP = {
#     "rsi": ["window"],
#     "sma": ["window"],
#     "ema": ["window"],
#     "macddiff": ["fast", "slow"],
#     "bb_pband": ["window", "std"],
# }
#
# # 运算符列表
# BINARY_OPS = ["minus", "mul", "div"]
# UNARY_OPS = ["sin", "cos"]
#
# class RecipeParser:
#     """
#     将因子名称解析为一棵或多棵 Recipe 树，
#     再扁平化为按依赖顺序（子先父后）的配方列表。
#
#     每个 recipe 格式:
#       {
#         "name": str,            # 完整表达式
#         "func": Optional[str],  # 函数名或运算符 (None 代表原始列)
#         "inputs": List[str],    # 依赖的子表达式名称或原始列名
#         "kwargs": Dict,         # 参数字典
#         "subrecipes": List[Dict]# 嵌套的子配方
#       }
#     """
#
#     def __init__(self):
#         pass
#
#     def get_recipes(self, feature_names: List[str]) -> List[Dict]:
#         """
#         对一组因子名执行解析，并返回扁平化、去重且按依赖顺序排列的 Recipe 列表。
#         """
#         all_recs = []
#         for name in feature_names:
#             rec = self._parse_expr(name)
#             rec["name"] = name
#             all_recs.append(rec)
#         return self._flatten_recipes(all_recs)
#
#     def _parse_expr(self, expr: str) -> Dict:
#         expr = expr.strip()
#
#         # 1) 二元运算
#         for op in BINARY_OPS:
#             token = f"_{op}_"
#             idx = self._find_top_level(expr, token)
#             if idx != -1:
#                 left_str = expr[:idx]
#                 right_str = expr[idx + len(token):]
#                 left_rec = self._parse_expr(left_str)
#                 right_rec = self._parse_expr(right_str)
#                 return {
#                     "name": expr,
#                     "func": op,
#                     "inputs": [left_rec["name"], right_rec["name"]],
#                     "kwargs": {},
#                     "subrecipes": [left_rec, right_rec],
#                 }
#
#         # 2) 一元运算
#         for op in UNARY_OPS:
#             prefix = f"{op}_"
#             if self._at_top_level_prefix(expr, prefix):
#                 inner_str = expr[len(prefix):]
#                 inner_rec = self._parse_expr(inner_str)
#                 return {
#                     "name": expr,
#                     "func": op,
#                     "inputs": [inner_rec["name"]],
#                     "kwargs": {},
#                     "subrecipes": [inner_rec],
#                 }
#
#         # 3) 基础函数解析
#         for func, keys in sorted(FUNC_PARAM_MAP.items(), key=lambda x: -len(x[0])):
#             prefix = func + "_"
#             if expr.startswith(prefix):
#                 rest = expr[len(prefix):]
#                 parts = self._top_split(rest)
#                 # 参数数 + 最后应为括号包裹
#                 if len(parts) >= len(keys) + 1:
#                     last = parts[-1]
#                     if last.startswith("(") and last.endswith(")"):
#                         param_vals = parts[:len(keys)]
#                         inner_str = last[1:-1]
#                         # 构造 kwargs
#                         kwargs = {
#                             k: (float(v) if "." in v else int(v))
#                             for k, v in zip(keys, param_vals)
#                         }
#                         inner_rec = self._parse_expr(inner_str)
#                         return {
#                             "name": expr,
#                             "func": func,
#                             "inputs": [inner_rec["name"]],
#                             "kwargs": kwargs,
#                             "subrecipes": [inner_rec],
#                         }
#
#         # 4) 纯括号包裹
#         if expr.startswith("(") and expr.endswith(")"):
#             return self._parse_expr(expr[1:-1])
#
#         # 5) 叶子节点
#         return {
#             "name": expr,
#             "func": None,
#             "inputs": [expr],
#             "kwargs": {},
#             "subrecipes": [],
#         }
#
#     def _find_top_level(self, expr: str, token: str) -> int:
#         """
#         在不进入任何括号的情况下查找 token 第一次出现的位置。
#         """
#         depth = 0
#         L = len(token)
#         for i, ch in enumerate(expr):
#             if ch == '(': depth += 1
#             elif ch == ')': depth -= 1
#             if depth == 0 and expr[i:i+L] == token:
#                 return i
#         return -1
#
#     def _at_top_level_prefix(self, expr: str, prefix: str) -> bool:
#         """
#         判断 expr 是否在顶层以 prefix 开头
#         """
#         if not expr.startswith(prefix):
#             return False
#         # 确保 prefix 第一次出现是在顶层
#         return self._find_top_level(expr, prefix) == 0
#
#     def _top_split(self, expr: str, sep: str = "_") -> List[str]:
#         """
#         顶层按 sep 拆分字符串，忽略括号内部的 sep。
#         """
#         parts = []
#         depth = 0
#         last = 0
#         for i, ch in enumerate(expr):
#             if ch == '(':
#                 depth += 1
#             elif ch == ')':
#                 depth -= 1
#             elif ch == sep and depth == 0:
#                 parts.append(expr[last:i])
#                 last = i + 1
#         parts.append(expr[last:])
#         return parts
#
#     def _flatten_recipes(self, recs: List[Dict]) -> List[Dict]:
#         """
#         将多棵子树按依赖（先子后父）扁平化，并去重。
#         """
#         seen = set()
#         flat = []
#         def dfs(r: Dict):
#             for sub in r.get("subrecipes", []):
#                 dfs(sub)
#             name = r["name"]
#             if name not in seen:
#                 seen.add(name)
#                 flat.append(r)
#         for r in recs:
#             dfs(r)
#         return flat


import re
from typing import List, Dict, Optional
import pandas as pd

from Utility.registry import build_dispatch

BINARY_OPS = ["minus", "mul", "div"]
UNARY_OPS = ["sin", "cos"]


class RecipeParser:
    """
    将因子名称解析为依赖树，再扁平化为按依赖顺序（子先父后）的配方列表。
    每个 recipe 字典:
      {
        "name": str,            # 完整表达式
        "func": Optional[str],  # 函数名或运算符 (None 表示原始列)
        "inputs": List[str],    # 依赖的子表达式名称或原始列名
        "kwargs": Dict,         # 参数字典
        "subrecipes": List[Dict]# 嵌套的子配方
      }
    """

    def __init__ (self):
        from Utility.registry import FACTOR_REGISTRY
        self.FUNC_PARAM_MAP: Dict[str, List[str]] = {
            name: meta.get ('param_keys', [])
            for name, meta in FACTOR_REGISTRY.items ()
        }

    def get_recipes (self, feature_names: List[str]) -> List[Dict]:
        all_recs = []
        for name in feature_names:
            rec = self._parse_expr (name)
            rec["name"] = name
            all_recs.append (rec)
        return self._flatten_recipes (all_recs)

    def _parse_expr (self, expr: str) -> Dict:
        expr = expr.strip ()

        # 1) 二元运算
        for op in BINARY_OPS:
            token = f"_{op}_"
            idx = self._find_top_level (expr, token)
            if idx != -1:
                left = expr[:idx]
                right = expr[idx + len (token):]
                lrec = self._parse_expr (left)
                rrec = self._parse_expr (right)
                return {"name": expr, "func": op,
                        "inputs": [lrec["name"], rrec["name"]],
                        "kwargs": {},
                        "subrecipes": [lrec, rrec]}

        for op in UNARY_OPS:
            # 只匹配 op_(inner) 形式
            token = f"{op}_("
            # 确保 token 在顶层开头，且末尾闭合一个 )
            if self._find_top_level (expr, token) == 0 and expr.endswith (")"):
                # 去掉前缀 op_( 和末尾 )
                inner = expr[len (token):-1]
                irec = self._parse_expr (inner)
                return {
                    "name": expr,
                    "func": op,
                    "inputs": [irec["name"]],
                    "kwargs": {},
                    "subrecipes": [irec],
                }

        # 3) 基础函数解析
        for func, keys in sorted (self.FUNC_PARAM_MAP.items (), key=lambda x: -len (x[0])):
            prefix = func + "_"
            if expr.startswith (prefix) and keys:
                rest = expr[len (prefix):]
                parts = self._top_split (rest)
                # 参数部分 + 最后一个括号包裹的 inner
                if len (parts) >= len (keys) + 1:
                    last = parts[-1]
                    if last.startswith ("(") and last.endswith (")"):
                        param_vals = parts[:len (keys)]
                        inner = last[1:-1]
                        kwargs = {k: (float (v) if "." in v else int (v))
                                  for k, v in zip (keys, param_vals)}
                        irec = self._parse_expr (inner)
                        return {"name": expr, "func": func,
                                "inputs": [irec["name"]],
                                "kwargs": kwargs,
                                "subrecipes": [irec]}

        # 4) 纯括号包裹
        if expr.startswith ("(") and expr.endswith (")"):
            return self._parse_expr (expr[1:-1])

        # 5) 叶子节点 (原始列)
        return {"name": expr, "func": None,
                "inputs": [expr], "kwargs": {},
                "subrecipes": []}

    def _find_top_level (self, expr: str, token: str) -> int:
        depth = 0
        L = len (token)
        for i, ch in enumerate (expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and expr[i:i + L] == token:
                return i
        return -1

    def _top_split (self, expr: str, sep: str = '_') -> List[str]:
        parts = []
        depth = 0
        last = 0
        for i, ch in enumerate (expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append (expr[last:i])
                last = i + 1
        parts.append (expr[last:])
        return parts

    def _flatten_recipes (self, recs: List[Dict]) -> List[Dict]:
        seen = set ()
        flat = []

        def dfs (r):
            for sub in r.get ('subrecipes', []):
                dfs (sub)
            if r['name'] not in seen:
                seen.add (r['name'])
                flat.append (r)

        for r in recs:
            dfs (r)
        return flat

    def apply_recipes (self, df: pd.DataFrame, recipes: List[Dict], columns_to_keep: Optional[List[str]] = None ) -> pd.DataFrame:
        """
        根据 recipes（已扁平化、按依赖顺序），
        在 df_feat（初始拷贝原 df）上依次生成每个因子列并返回。
        """
        from Utility.factors import CROSS_OPS
        from Utility.registry import FACTOR_REGISTRY

        # 1) 初始化：先拷贝原始所有列，方便基础因子直接用原始列
        df_feat = df.copy ()

        # 2) 准备交叉运算符映射
        cross_map = {op['name']: op for op in CROSS_OPS}

        # 3) 按 recipes 顺序，一个循环搞定所有因子
        for rec in recipes:
            name = rec['name']
            func = rec['func']
            inputs = rec['inputs']  # e.g. ['open'] or ['macd_diff_5_20_(open)']
            kwargs = rec['kwargs']  # e.g. {'window': 14}

            if func is None:
                continue

            # --- 基础因子：调用 DataFrame 版函数 ---
            if func in FACTOR_REGISTRY:
                meta = FACTOR_REGISTRY[func]
                df_func = meta['func']
                param_keys = meta['param_keys']

                # 把单值参数包成列表，对应函数签名里的 list 参数
                df_kwargs = {k: [kwargs[k]] for k in param_keys if k in kwargs}

                # df_func(self, df, cols:List[str], **params) → Dict[name,Series]
                out = df_func (df_feat, inputs, **df_kwargs)
                # 拿回那个 key 对应的 Series
                df_feat[name] = out[name]
                continue

            # --- 交叉运算符：sin/mul/div/... ---
            if func in cross_map:
                op_meta = cross_map[func]
                op = op_meta['func']
                arity = op_meta['arity']

                if arity == 1:
                    df_feat[name] = op (df_feat[inputs[0]])
                else:  # arity == 2
                    df_feat[name] = op (df_feat[inputs[0]], df_feat[inputs[1]])
                continue

        if columns_to_keep is None:
            columns_to_keep = df.columns.tolist()
        df_feat = df_feat[columns_to_keep]

        return df_feat
