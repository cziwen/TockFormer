import requests
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# A+P+I_KEY = 'oilTTMMexxTBTmjivaMq3R0Y9ZS1BKbK'

TICKER = 'SPY'
MULTIPLIER = 30
TIMESPAN = 'minute'
FROM_DATE = '2025-03-08'
TO_DATE = '2025-04-02'
FOR = 'test'

# 初始 URL 和参数
base_url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{MULTIPLIER}/{TIMESPAN}/{FROM_DATE}/{TO_DATE}"
params = {
    'adjusted': 'true',
    'sort': 'asc',
    'limit': 50000,
    'apiKey': API_KEY
}

all_data = []

print(f"Fetching SPY {MULTIPLIER}-{TIMESPAN} bars...")

while True:
    response = requests.get(base_url, params=params)
    data = response.json()

    if 'results' in data:
        all_data.extend(data['results'])
        print(f"Fetched {len(data['results'])} bars, total: {len(all_data)}")

        if 'next_url' in data:
            # === 修复：添加 API key 到 next_url ===
            parsed = urlparse(data['next_url'])
            query = parse_qs(parsed.query)
            query['apiKey'] = [API_KEY]
            new_query = urlencode(query, doseq=True)
            base_url = urlunparse(parsed._replace(query=new_query))
            params = {}  # 因为 URL 里已经带了所有 query 参数
            time.sleep(1)  # 避免速率限制
        else:
            break
    else:
        print("Error:", data.get('error', 'Unknown error'))
        break

# === 转换为 DataFrame
df = pd.DataFrame(all_data)

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'num_transactions'
    }, inplace=True)

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'num_transactions']]
    path = f"rawdata/SPY_{MULTIPLIER}{TIMESPAN}_{FOR}.csv"
    df.to_csv(f"{path}", index=False)
    print(f"✅ Done! Data saved to '{path}'")
else:
    print("⚠️ No data retrieved.")