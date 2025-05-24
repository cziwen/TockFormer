import time

import finnhub
import pandas as pd
import requests

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from finnhub.exceptions import FinnhubAPIException


# ============================== Candle
def fetch_candle_chunk_sync (symbol: str,
                             resolution: str,
                             from_ts: int,
                             to_ts: int,
                             token: str) -> dict:
    url = 'https://finnhub.io/api/v1/stock/candle'
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': from_ts,
        'to': to_ts,
        'token': token
    }
    r = requests.get (url, params=params, timeout=10)
    r.raise_for_status ()
    return r.json ()


def fetch_candle_data (symbol: str,
                       start_date: str,
                       end_date: str,
                       interval: str = '1',
                       token: str = 'YOUR_API_KEY',
                       chunk_days: int = 7,
                       max_workers: int = 5
                       ) -> pd.DataFrame:
    """
    线程池并发拉取分钟/小时 K 线数据（Notebook 可用）。
    - symbol: 股票代码
    - start_date/end_date: 'YYYY-MM-DD'，end_date 会被包含
    - interval: '1','5','15','30','60'
    - chunk_days: 每个线程负责多少天数据
    - max_workers: 并发线程数
    """
    # 1. 切分时间区间（end_date + 1 天以包含整天）
    start_dt = datetime.strptime (start_date, '%Y-%m-%d')
    end_dt = datetime.strptime (end_date, '%Y-%m-%d') + timedelta (days=1)

    chunks = []
    cur = start_dt
    while cur < end_dt:
        nxt = min (cur + timedelta (days=chunk_days), end_dt)
        chunks.append ((int (cur.timestamp ()), int (nxt.timestamp ())))
        cur = nxt

    # 2. 并发请求
    results = []
    with ThreadPoolExecutor (max_workers=max_workers) as exe:
        futures = {
            exe.submit (fetch_candle_chunk_sync, symbol, interval, f, t, token): (f, t)
            for f, t in chunks
        }
        for fut in tqdm (as_completed (futures),
                         total=len (futures),
                         desc=f"{symbol} {interval}-min K 线块"):
            f, t = futures[fut]
            try:
                data = fut.result ()
                results.append (data)
            except Exception as e:
                print (f"[{symbol}] 区间 {datetime.fromtimestamp (f)}–"
                       f"{datetime.fromtimestamp (t)} 拉取失败：{e}")

    # 3. 汇总 records
    records = []
    for data in results:
        if data.get ('s') != 'ok':
            continue
        ts = data['t']
        for i in range (len (ts)):
            records.append ({
                'timestamp': ts[i],
                'open': data['o'][i],
                'high': data['h'][i],
                'low': data['l'][i],
                'close': data['c'][i],
                'volume': data['v'][i],
            })

    # 4. DataFrame + 时区转换
    df = pd.DataFrame (records)
    if not df.empty:
        df['timestamp'] = (
            pd.to_datetime (df['timestamp'], unit='s', utc=True)
            .dt.tz_convert ('America/New_York')
        )
        df = df.sort_values ('timestamp').reset_index (drop=True)
    return df


# ================================ Tick

def fetch_tick_data_concurrent (
        symbol: str,
        date_str: str,
        api_key: str,
        limit: int = 25000,
        page_workers: int = 5,
        sleep_sec: float = 0.1
) -> list[dict]:
    """
    对单个交易日的 Tick 数据，用线程池并发拉取所有分页。
    打印总数信息并返回所有原始 dict 列表。
    """
    client = finnhub.Client (api_key=api_key)
    while True:  # 尝试等待到api限制恢复为止
        try:
            head = client.stock_tick (symbol=symbol, date=date_str, limit=1, skip=0)
            break
        except FinnhubAPIException as e:
            if e.status_code == 429:
                # tqdm.write (f"[{date_str} Rate limit hit, sleeping 30s...")
                time.sleep (30)
            else:
                tqdm.write (f"Error fetching {date_str}: {e}")
                return []

    total = head.get ('total', 0)
    # tqdm.write (f"Fetching ticks for {date_str} — 当日总 tick 数: {total}")
    if total == 0:
        return []

    skips = list (range (0, total, limit))
    records: list[dict] = []

    def fetch_page (skip: int) -> list[dict]:
        while True:
            try:
                resp = client.stock_tick (
                    symbol=symbol, date=date_str, limit=limit, skip=skip
                )
                count = resp.get ('count', 0)
                if count == 0:
                    return []
                return [
                    {
                        'timestamp': resp['t'][i],
                        'symbol': resp['s'],
                        'price': resp['p'][i],
                        'volume': resp['v'][i],
                        'condition': resp['c'][i],
                    }
                    for i in range (count)
                ]
            except FinnhubAPIException as e:
                if e.status_code == 429:
                    # tqdm.write (f"[{date_str} skip={skip}] Rate limit hit, sleeping 30s...")
                    time.sleep (30)
                else:
                    tqdm.write (f"Error fetching {date_str} skip={skip}: {e}")
                    return []

    # 并发拉取，不显示分页进度条
    with ThreadPoolExecutor (max_workers=page_workers) as exe:
        futures = [exe.submit (fetch_page, sk) for sk in skips]
        for fut in as_completed (futures):
            page_data = fut.result ()
            if page_data:
                records.extend (page_data)
            time.sleep (sleep_sec)

    return records


def fetch_tick_data (
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        api_key: str,
        limit: int = 25000,
        page_workers: int = 5,
        day_workers: int = 3,
        sleep_sec: float = 0.1
) -> pd.DataFrame:
    """
    分日并发拉取 Tick 数据：最外层对多个交易日做并发，每个交易日内部再并发拉取分页。
    **仅在此处显示“Days”总进度条**，内部无进度信息。
    """

    # parse dates
    if isinstance (start_date, str):
        start = datetime.fromisoformat (start_date)
    else:
        start = start_date
    if isinstance (end_date, str):
        end = datetime.fromisoformat (end_date)
    else:
        end = end_date

    # 构造交易日列表（周一到周五）
    dates = []
    dt = start
    while dt <= end:
        if dt.weekday () < 5:
            dates.append (dt.strftime ('%Y-%m-%d'))
        dt += timedelta (days=1)
    if not dates:
        return pd.DataFrame (columns=['timestamp', 'symbol', 'price', 'volume', 'condition'])

    all_records: list[dict] = []

    def fetch_one_day (ds: str) -> list[dict]:
        return fetch_tick_data_concurrent (
            symbol=symbol,
            date_str=ds,
            api_key=api_key,
            limit=limit,
            page_workers=page_workers,
            sleep_sec=sleep_sec
        )

    # 外层按天并发，显示总进度条
    with ThreadPoolExecutor (max_workers=day_workers) as exe:
        futures = {exe.submit (fetch_one_day, ds): ds for ds in dates}
        for fut in tqdm (as_completed (futures),
                         total=len (futures),
                         desc="Days"):
            ds = futures[fut]
            try:
                recs = fut.result ()
            except Exception as e:
                tqdm.write (f"Failed to fetch {ds}: {e}")
                recs = []
            if recs:
                all_records.extend (recs)
            time.sleep (sleep_sec)

    # 汇总到 DataFrame 并做时区转换
    if not all_records:
        return pd.DataFrame (columns=['timestamp', 'symbol', 'price', 'volume', 'condition'])

    df = pd.DataFrame (all_records)
    df['timestamp'] = (
        pd.to_datetime (df['timestamp'], unit='ms', utc=True)
        .dt.tz_convert ('America/New_York')
    )
    return df.sort_values ('timestamp').reset_index (drop=True)


def fetch_tick_data_last_day (
        symbol: str,
        api_key: str,
        limit: int = 25000,
        page_workers: int = 5,
        day_workers: int = 3,
        sleep_sec: float = 0.1
) -> pd.DataFrame:
    """
    分日并发拉取 Tick 数据（前天到今天），自动跳过周末：
      - start = 今天日期 - 1 天
      - end   = 今天日期
    其它参数同原 fetch_tick_data。
    """
    # 计算日期范围
    today = datetime.now ()
    start = today - timedelta (days=1)
    end = today

    # 复用原函数
    return fetch_tick_data (
        symbol=symbol,
        start_date=start,
        end_date=end,
        api_key=api_key,
        limit=limit,
        page_workers=page_workers,
        day_workers=day_workers,
        sleep_sec=sleep_sec
    )


# ===================================== NBBO

def fetch_nbbo_data_concurrent (
        symbol: str,
        date_str: str,
        api_key: str,
        limit: int = 25000,
        page_workers: int = 5,
        sleep_sec: float = 0.1
) -> list[dict]:
    """
    并发分页拉取单个交易日的历史 NBBO 数据。
    """
    client = finnhub.Client (api_key=api_key)

    # —— head 请求重试，直到拿到 total —— #
    while True:
        try:
            head = client.stock_nbbo (symbol=symbol, date=date_str, limit=1, skip=0)
            total = head.get ('total', 0)
            break
        except FinnhubAPIException as e:
            # 限流 sleep 再试
            if e.status_code == 429:
                wait = 30  # seconds
                # tqdm.write(f"[{date_str}] NBBO head error {e.status_code}, retry in {wait}s…")
                time.sleep (wait)
            else:
                tqdm.write (f"Error fetching {date_str}: {e}")
                return []

    if total == 0:
        return []

    skips = list (range (0, total, limit))
    records: list[dict] = []

    def fetch_page (skip: int) -> list[dict]:
        while True:
            try:
                resp = client.stock_nbbo (
                    symbol=symbol, date=date_str, limit=limit, skip=skip
                )
                count = resp.get ('count', 0)
                if count == 0:
                    return []
                return [
                    {
                        'timestamp': resp['t'][i],
                        'bid': resp['b'][i],
                        'bid_size': resp['bv'][i],
                        'ask': resp['a'][i],
                        'ask_size': resp['av'][i],
                        'condition': resp['c'][i],
                    }
                    for i in range (count)
                ]
            except FinnhubAPIException as e:
                if e.status_code == 429:
                    # tqdm.write (f"[{date_str} skip={skip}] Rate limit hit, sleeping 30s...")
                    time.sleep (30)
                else:
                    tqdm.write (f"Error fetching {date_str} skip={skip}: {e}")
                    return []

    # 并发拉取分页
    with ThreadPoolExecutor (max_workers=page_workers) as exe:
        futures = [exe.submit (fetch_page, sk) for sk in skips]
        for fut in as_completed (futures):
            page_data = fut.result ()
            if page_data:
                records.extend (page_data)
            time.sleep (sleep_sec)

    return records


def fetch_nbbo_data (
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        api_key: str,
        limit: int = 25000,
        page_workers: int = 5,
        day_workers: int = 3,
        sleep_sec: float = 0.1
) -> pd.DataFrame:
    """
    分日并发拉取 NBBO 数据（只在最外层显示“Days”进度）。
    """
    # —— 构造交易日列表 —— #
    if isinstance (start_date, str):
        start = datetime.fromisoformat (start_date)
    else:
        start = start_date
    if isinstance (end_date, str):
        end = datetime.fromisoformat (end_date)
    else:
        end = end_date

    dates = []
    dt = start
    while dt <= end:
        if dt.weekday () < 5:
            dates.append (dt.strftime ('%Y-%m-%d'))
        dt += timedelta (days=1)

    if not dates:
        return pd.DataFrame (columns=['timestamp', 'bid', 'bid_size', 'ask', 'ask_size', 'condition'])

    all_records = []

    def fetch_one_day (ds: str):
        return fetch_nbbo_data_concurrent (
            symbol=symbol,
            date_str=ds,
            api_key=api_key,
            limit=limit,
            page_workers=page_workers,
            sleep_sec=sleep_sec
        )

    # —— 外层并发 + 进度条 —— #
    with ThreadPoolExecutor (max_workers=day_workers) as exe:
        futures = {exe.submit (fetch_one_day, ds): ds for ds in dates}
        for fut in tqdm (as_completed (futures),
                         total=len (futures),
                         desc="Days"):
            ds = futures[fut]
            try:
                recs = fut.result ()
            except Exception as e:
                tqdm.write (f"Failed to fetch {ds}: {e}")
                recs = []
            if recs:
                all_records.extend (recs)
            time.sleep (sleep_sec)

    # —— 整合成 DataFrame 并排序 —— #
    df = pd.DataFrame (all_records)
    if not df.empty:
        df['timestamp'] = (
            pd.to_datetime (df['timestamp'], unit='ms', utc=True)
            .dt.tz_convert ('America/New_York')
        )
        df = df.sort_values ('timestamp').reset_index (drop=True)

    return df
