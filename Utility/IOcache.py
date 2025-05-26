import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.impute import SimpleImputer
from tqdm import tqdm

def dump_dataframe(
    df: pd.DataFrame,
    cache_dir: str,
    n_jobs: int = None,
    clear: bool = False
) -> None:
    """
    并行将传入的 DataFrame 按列写入 cache_dir 下的 Parquet 文件，
    每个列保存为单独文件（列名.parquet）。

    Args:
        df: 要持久化的 DataFrame
        cache_dir: 写盘目录路径
        n_jobs: 最大线程数，默认使用 CPU 核心数
        clear: 是否先清空 cache_dir 中已有的 .parquet 文件，默认 False

    重复的列名会直接覆盖已有文件。
    """
    os.makedirs(cache_dir, exist_ok=True)
    workers = n_jobs or os.cpu_count()

    # 如果需要，先清理目录下所有 .parquet 文件
    if clear:
        for fname in os.listdir(cache_dir):
            if fname.endswith('.parquet'):
                try:
                    os.remove(os.path.join(cache_dir, fname))
                except Exception as e:
                    print(f"[IOcache] 清理文件 {fname} 时出错: {e}")

    def _dump(col_name):
        series = df[col_name]
        path = os.path.join(cache_dir, f"{col_name}.parquet")
        # 保存为单列 DataFrame 保留列名，直接覆盖同名文件
        series.to_frame(col_name).to_parquet(path)
        return col_name

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_dump, col): col for col in df.columns}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='IO Dump'):
            col = futures[fut]
            try:
                _ = fut.result()
            except Exception as e:
                print(f"[IOcache] 写入列 {col} 时出错: {e}")


def load_dataframe(
    cache_dir: str,
    n_jobs: int = None,
    columns: list[str] = None
) -> pd.DataFrame:
    """
    并行读取 cache_dir 下指定或所有 .parquet 列文件，
    合并并返回 DataFrame。

    Args:
        cache_dir: 读取目录路径
        n_jobs: 最大线程数，默认使用 CPU 核心数
        columns: 要加载的列名列表，None 表示加载目录下所有列

    Returns:
        pd.DataFrame: 合并后的 DataFrame，列顺序与 columns 一致或按文件名排序
    """
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"目录不存在: {cache_dir}")

    all_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
    # 过滤指定列
    if columns is not None:
        target_files = [f"{col}.parquet" for col in columns if f"{col}.parquet" in all_files]
    else:
        target_files = all_files

    workers = n_jobs or os.cpu_count()
    series_dict: dict[str, pd.Series] = {}

    def _load(fname):
        col_name = os.path.splitext(fname)[0]
        df_col = pd.read_parquet(os.path.join(cache_dir, fname))
        # 单列 DataFrame 转 Series
        return col_name, df_col[col_name]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_load, fn): fn for fn in target_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='IO Load'):
            fname = futures[fut]
            try:
                col_name, series = fut.result()
                series_dict[col_name] = series
            except Exception as e:
                print(f"[IOcache] 读取文件 {fname} 时出错: {e}")

    # 合并所有 Series
    result_df = pd.DataFrame(series_dict)
    return result_df

def _clean(cache_dir, fname, imputer, std_threshold):
    path = os.path.join(cache_dir, fname)
    try:
        # 读取单列 DataFrame，index 是 timestamp
        df = pd.read_parquet(path)
        col = df.columns[0]
        series: pd.Series = df[col]

        # 1. 删除全 NaN 列
        if series.isna().all():
            os.remove(path)
            return fname, 'dropped empty'

        # 2. 填充缺失值
        data = series.values.reshape(-1, 1)
        filled = imputer.fit_transform(data).ravel()
        filled_series = pd.Series(filled, index=series.index, name=col)

        # 3. 标准差判断
        if filled_series.std(ddof=0) <= std_threshold:
            os.remove(path)
            return fname, 'dropped constant'

        # 4. 覆盖写回，并保留 index
        filled_series.to_frame().to_parquet(path)
        return fname, 'imputed and kept'
    except Exception as e:
        return fname, f'error: {e}'

def wash(
    cache_dir: str,
    n_jobs: int = None,
    std_threshold: float = 1e-8,
    impute_strategy: str = 'mean'
) -> None:
    """
    清洗 cache_dir 下所有单列 Parquet 文件（每个文件一列，索引为 timestamp）：
      1. 删除全 NaN 列（删除对应的 Parquet 文件）；
      2. 使用 sklearn SimpleImputer 对缺失值进行填充（mean/median）；
      3. 对填充后的列计算标准差，若 ≤ std_threshold，则删除文件；
      4. 否则覆盖写回，并保留原 timestamp 索引。

    Args:
        cache_dir: Parquet 文件所在目录
        n_jobs: 并行线程数，默认 CPU 核心数
        std_threshold: 标准差阈值，小于等于此值视为近似常数
        impute_strategy: 填充策略，可选 'mean' 或 'median'
    """
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"目录不存在: {cache_dir}")

    files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
    workers = n_jobs or os.cpu_count()

    # 初始化 imputer
    imputer = SimpleImputer(strategy=impute_strategy)

    count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_clean, cache_dir, fn, imputer, std_threshold): fn for fn in files}
        pbar = tqdm (as_completed (futures), total=len (futures), desc='IO Wash')
        for fut in pbar:
            fname = futures[fut]
            result = fut.result()
            # print(f"[IOcache][wash] {result[0]} -> {result[1]}")
            if result[1] == 'dropped empty' or result[1] == 'dropped constant':
                count+=1
            pbar.set_postfix(dropped=count)



def delete_parquet_files(
    cache_dir: str,
    columns: list[str],
    n_jobs: int = None
) -> None:
    """
    并行删除 cache_dir 下指定列的 Parquet 文件。

    Args:
        cache_dir: Parquet 文件所在目录
        columns: 要删除的列名列表（不带 .parquet 后缀）
        n_jobs: 并行线程数，默认使用 CPU 核心数
    """
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"目录不存在: {cache_dir}")

    workers = n_jobs or os.cpu_count()

    def _delete(col_name):
        fname = f"{col_name}.parquet"
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
                return col_name, True
            except Exception as e:
                return col_name, f"error: {e}"
        return col_name, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_delete, col): col for col in columns}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='IO Delete'):
            col, status = fut.result()
            if status is True:
                # print(f"[IOcache] 已删除 {col}.parquet")
                continue
            elif status is False:
                # print(f"[IOcache] 文件 {col}.parquet 不存在")
                continue
            else:
                tqdm.write(f"[IOcache] 删除 {col}.parquet 时出错: {status}")
