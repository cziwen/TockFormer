# from typing import Callable, Dict, Any
# FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}
# def factor(*tags: str):
#     def decorator(fn: Callable):
#         FACTOR_REGISTRY[fn.__name__] = {'func': fn, 'tags': set(tags)}
#         return fn
#     return decorator


from typing import Callable, List, Dict, Optional

# ==== 基础因子注册表 ====
# 存储每个因子函数及其元数据
FACTOR_REGISTRY: Dict[str, Dict] = {}


# ==== 交叉运算符注册表 ====


def factor (category: str,
            param_keys: List[str],
            key_template: Optional[Callable[[List, str], str]] = None):
    """
    Decorator: 注册一个基础因子函数。

    参数:
    - category: 因子分类，比如 'bounded' 或 'unbounded'
    - param_keys: key 中参数的名称列表，顺序对应实际参数列表
    - key_template: 可选的自定义 key 生成方法，接受 (params: List, col: str) -> str
      默认模板: func_{param1}_{param2}_..._({col})
    """

    def decorator (fn: Callable):
        name = fn.__name__

        # 默认的 key 生成模板
        def default_template (params: List, col: str) -> str:
            param_str = "_".join (str (p) for p in params)
            return f"{name}_{param_str}_({col})"

        FACTOR_REGISTRY[name] = {
            "func": fn,
            "category": category,
            "param_keys": param_keys,
            "key_template": key_template or default_template
        }
        return fn

    return decorator


def build_dispatch () -> Dict[str, Callable]:
    from Utility.factors import CROSS_OPS
    dispatch: Dict[str, Callable] = {}

    # 基础因子
    for name, meta in FACTOR_REGISTRY.items ():
        dispatch[name] = meta["func"]

    # 交叉运算符
    # CROSS_OPS 是一组 dict，每个 dict 里有 'name' 和 'func'
    for op in CROSS_OPS:
        dispatch[op["name"]] = op["func"]

    return dispatch
