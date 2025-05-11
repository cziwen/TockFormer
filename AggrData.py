"""
实时股票交易数据采集与聚合系统
- 通过WebSocket订阅实时交易数据
- 使用REST API回填缺失数据
- 生成标准化的1分钟K线数据
"""

import json
import threading
import websocket
import time
import csv
import os
import pytz
import logging
import signal
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# 从Util导入需要的函数
from Util import fetch_stock_data_finnhub_paginated, safeLoadCSV

# ---------------------------------------- 配置参数 ----------------------------------------

# 要订阅的股票列表
SYMBOLS = ["AAPL", "GOOGL", "TSLA"]

# 固定为1分钟间隔
INTERVAL_KEY = "1min"
INTERVAL_MIN = 1

# API设置
FINNHUB_TOKEN = "cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00"
WS_URL = f"wss://ws.finnhub.io?token={FINNHUB_TOKEN}"

# 输出路径
DATA_DIR = "./data/raw"
LOG_DIR = "./data/logs"

# ---------------------------------------- 主函数 ----------------------------------------

def main():
    """主程序入口"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("股票数据采集系统启动")
    
    # 准备第一个间隔的回填任务
    prepare_first_interval_backfill()
    
    # 启动定时聚合
    start_minute_timer()
    
    # 可选：启动市场交易时间检查
    # start_market_hours_check()
    
    # 启动WebSocket连接
    start_websocket()

# ---------------------------------------- 初始化设置 ----------------------------------------

# 全局变量
ws_app = None
should_exit = False
is_first_interval = True
first_interval_complete = False
backfill_queue = []

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/stock_data.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据存储结构
candle_data = defaultdict(dict)  # 结构: candle_data[symbol][timestamp] = stats
current_data = {}  # 当前周期的交易数据缓存

# 初始化当前数据结构
for symbol in SYMBOLS:
    current_data[symbol] = {
        'trades': [],
        'first_trade_price': None,
        'high': float('-inf'),
        'low': float('inf'),
        'volume': 0,
        'value': 0,  # 用于计算VWAP
        'count': 0,
    }

# ---------------------------------------- 时间处理函数 ----------------------------------------

def get_eastern_time():
    """获取当前美国东部时间"""
    eastern_tz = pytz.timezone('America/New_York')
    return datetime.now(pytz.UTC).astimezone(eastern_tz)

def floor_time_to_minute(dt):
    """将时间向下取整到分钟"""
    return dt.replace(second=0, microsecond=0)

def get_next_minute_time():
    """计算下一个整分钟的时间"""
    now = get_eastern_time()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    return next_minute

# ---------------------------------------- 数据处理函数 ----------------------------------------

def backfill_first_interval():
    """后台任务：获取并填充第一个时间间隔的数据"""
    global backfill_queue, first_interval_complete
    
    # 等待第一个时间间隔结束
    while not first_interval_complete and not should_exit:
        time.sleep(1)
    
    if should_exit:
        return
    
    logger.info("开始回填第一个时间间隔的数据...")
    
    # 处理队列中的每个回填任务
    for task in backfill_queue:
        symbol = task['symbol']
        start_time = task['start_time']
        end_time = task['end_time']
        
        # 格式化为日期字符串
        start_date = start_time.strftime('%Y-%m-%d')
        end_date = (end_time + timedelta(days=1)).strftime('%Y-%m-%d')  # 加一天确保包含当天
        
        try:
            # 使用Util中的函数获取历史数据
            df = fetch_stock_data_finnhub_paginated(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1',  # 1分钟
                token=FINNHUB_TOKEN,
                chunk_days=1,  # 短时间范围，使用1天即可
                verbose=False
            )
            
            # 安全加载，处理可能的NaN值
            df = safeLoadCSV(df)
            
            # 将日期时间转换为字符串进行比较 - 解决类型不匹配问题
            start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 将DataFrame的timestamp列也转为字符串
            df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 使用字符串进行比较
            mask = (df['timestamp_str'] >= start_str) & (df['timestamp_str'] < end_str)
            df_filtered = df[mask]
            
            if not df_filtered.empty:
                # 处理并写入数据
                for _, row in df_filtered.iterrows():
                    timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:00')
                    stat = {
                        'timestamp': timestamp_str,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'vwap': row['close'],  # 简化: 使用收盘价作为VWAP
                        'volume': row['volume'],
                        'num_transactions': 1  # 简化: 设置为1
                    }
                    
                    # 存储数据并写入CSV
                    candle_data[symbol][timestamp_str] = stat
                    write_to_csv(symbol, timestamp_str, stat)
                
                logger.info(f"已成功回填 {symbol} 在 {start_time} 到 {end_time} 之间的 {len(df_filtered)} 条数据")
            else:
                logger.warning(f"过滤后没有 {symbol} 在 {start_time} 到 {end_time} 之间的数据")
                
        except Exception as e:
            logger.error(f"回填 {symbol} 数据时出错: {str(e)}")
    
    logger.info("回填第一个时间间隔的数据完成")
    backfill_queue = []  # 清空队列

def prepare_first_interval_backfill():
    """准备第一个时间间隔的回填任务"""
    global backfill_queue, is_first_interval
    
    # 获取当前时间和下一整分钟时间
    now = get_eastern_time()
    now_floor = floor_time_to_minute(now)
    next_minute = get_next_minute_time()
    
    # 为每个符号创建回填任务
    for symbol in SYMBOLS:
        backfill_queue.append({
            'symbol': symbol,
            'start_time': now_floor,
            'end_time': next_minute
        })
    
    logger.info(f"已准备 {len(backfill_queue)} 个回填任务，将在第一个间隔结束后执行")
    
    # 启动后台回填线程
    threading.Thread(target=backfill_first_interval, daemon=True).start()

def aggregate_minute_data():
    """聚合当前分钟的交易数据"""
    global is_first_interval, first_interval_complete
    
    # 获取当前时间
    now = get_eastern_time()
    minute_timestamp = floor_time_to_minute(now).strftime('%Y-%m-%d %H:%M:00')
    
    # 检查是否为第一个时间间隔
    if is_first_interval:
        is_first_interval = False
        first_interval_complete = True
        logger.info("第一个时间间隔结束，忽略收集到的数据（将通过REST API获取）")
        
        # 重置当前数据，不处理第一个间隔
        for symbol in SYMBOLS:
            current_data[symbol] = {
                'trades': [],
                'first_trade_price': None,
                'high': float('-inf'),
                'low': float('inf'),
                'volume': 0,
                'value': 0,
                'count': 0,
            }
        return
    
    # 处理每个股票的数据
    for symbol, data in current_data.items():
        if not data['trades']:
            continue  # 跳过没有交易的股票
            
        # 计算统计数据
        open_price = data['first_trade_price']
        close_price = data['trades'][-1]['price'] if data['trades'] else None
        high_price = data['high']
        low_price = data['low']
        volume = data['volume']
        vwap = data['value'] / volume if volume > 0 else 0
        num_transactions = data['count']
        
        # 跳过无效数据
        if open_price is None or close_price is None:
            logger.warning(f"跳过无效数据 {symbol}: {minute_timestamp}")
            continue
        
        # 创建聚合数据
        stat = {
            'timestamp': minute_timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'vwap': vwap,
            'volume': volume,
            'num_transactions': num_transactions
        }
        
        # 存储聚合数据并写入CSV
        candle_data[symbol][minute_timestamp] = stat
        write_to_csv(symbol, minute_timestamp, stat)
        
        # 重置当前周期数据
        current_data[symbol] = {
            'trades': [],
            'first_trade_price': None,
            'high': float('-inf'),
            'low': float('inf'),
            'volume': 0,
            'value': 0,
            'count': 0,
        }

def write_to_csv(symbol, timestamp, stats):
    """将聚合数据写入CSV文件"""
    filename = f"{DATA_DIR}/{symbol}_{INTERVAL_KEY}.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'num_transactions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(stats)
    
    logger.info(f"数据已写入 {filename}：{stats['timestamp']}")

# ---------------------------------------- WebSocket处理函数 ----------------------------------------

def on_message(ws, message):
    """收到WebSocket消息时的回调"""
    # 忽略第一个时间间隔的数据
    if is_first_interval:
        return
        
    msg = json.loads(message)
    if msg.get("type") == "trade":
        for trade in msg["data"]:
            if "1" not in trade.get("c", []):
                continue # 没有包含1说明不是正常时段交易
            
            symbol = trade["s"]
            price = trade["p"]
            volume = trade["v"]
            ts_millis = trade["t"]
            
            # 只处理关注的股票
            if symbol not in SYMBOLS:
                continue
                
            # 存储原始交易信息
            trade_info = {
                'price': price,
                'volume': volume,
                'timestamp': ts_millis
            }
            
            # 更新当前周期数据
            if current_data[symbol]['first_trade_price'] is None:
                current_data[symbol]['first_trade_price'] = price
                
            current_data[symbol]['trades'].append(trade_info)
            current_data[symbol]['high'] = max(current_data[symbol]['high'], price)
            current_data[symbol]['low'] = min(current_data[symbol]['low'], price)
            current_data[symbol]['volume'] += volume
            current_data[symbol]['value'] += price * volume
            current_data[symbol]['count'] += 1

            # 打印接收到的交易信息（可选，交易量大时考虑关闭）
            print(f"[{symbol}] Price={price:.2f} Volume={volume}")

def on_error(ws, error):
    print(f"WebSocket错误: {error}")
    logger.error(f"WebSocket错误: {error}")

def on_close(ws, close_status_code, close_msg):
    global should_exit
    print(f"WebSocket关闭: 状态码={close_status_code}, 消息={close_msg}")
    logger.error(f"WebSocket关闭: 状态码={close_status_code}, 消息={close_msg}")
    
    # 只有在未设置退出标志时尝试重连
    if not should_exit:
        print("5秒后尝试重新连接...")
        logger.info("5秒后尝试重新连接...")
        time.sleep(5)
        start_websocket()  # 重新启动WebSocket

def on_open(ws):
    """连接建立后订阅所有股票"""
    def run():
        # 订阅所有股票
        for sym in SYMBOLS:
            sub_msg = json.dumps({"type": "subscribe", "symbol": sym})
            ws.send(sub_msg)
            logger.info(f"已订阅 {sym}")
            time.sleep(0.1)

    threading.Thread(target=run).start()

def start_websocket():
    """启动WebSocket连接"""
    global ws_app
    
    if should_exit:
        return
        
    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # 添加ping_interval参数保持连接活跃
    ws_app.run_forever(ping_interval=30)

# ---------------------------------------- 系统控制函数 ----------------------------------------

def signal_handler(sig, frame):
    """处理CTRL+C信号，优雅退出程序"""
    global should_exit
    print("\n正在退出程序，请稍候...")
    logger.info("收到退出信号，正在关闭...")
    should_exit = True
    
    # 关闭WebSocket连接
    if ws_app:
        logger.info("正在关闭WebSocket连接...")
        ws_app.close()
    
    # 给线程一些时间完成
    time.sleep(2)
    logger.info("程序已终止")
    sys.exit(0)

def start_minute_timer():
    """启动分钟定时器"""
    def run_timer():
        global is_first_interval
        
        # 首先，确定第一个时间间隔的结束时间（下一整分钟）
        next_minute = get_next_minute_time()
        now = get_eastern_time()
        wait_seconds = (next_minute - now).total_seconds()
        
        logger.info(f"等待第一个时间间隔结束，将在 {next_minute.strftime('%Y-%m-%d %H:%M:%S')} 进行，等待 {wait_seconds:.2f} 秒")
        
        # 等待到下一整分钟
        time.sleep(max(0, wait_seconds))
        
        # 从此开始循环聚合每分钟数据
        while not should_exit:
            # 聚合当前分钟的数据
            aggregate_minute_data()
            
            # 计算下一分钟
            next_minute = get_next_minute_time()
            now = get_eastern_time()
            wait_seconds = (next_minute - now).total_seconds()
            
            logger.info(f"等待下一次聚合，将在 {next_minute.strftime('%Y-%m-%d %H:%M:%S')} 进行，等待 {wait_seconds:.2f} 秒")
            
            # 等待到下一整分钟
            time.sleep(max(0, wait_seconds))
    
    # 启动定时器线程
    threading.Thread(target=run_timer, daemon=True).start()

def start_market_hours_check():
    """启动市场交易时间检查线程"""
    def check_market_hours():
        while not should_exit:
            now = get_eastern_time()
            # 检查是否为交易日（周一至周五）
            is_weekday = 0 <= now.weekday() <= 4
            # 检查是否在交易时间（美东时间上午9:30至下午4:00）
            is_trading_hours = (9 < now.hour or (now.hour == 9 and now.minute >= 30)) and now.hour < 16
            
            # 打印市场状态
            if is_weekday and is_trading_hours:
                logger.info("市场开盘中，数据收集活跃")
            else:
                logger.info("市场已关闭，等待开盘")
            
            # 30分钟检查一次
            time.sleep(1800)
    
    # 启动市场检查线程
    threading.Thread(target=check_market_hours, daemon=True).start()

# ---------------------------------------- 主函数 ----------------------------------------
if __name__ == "__main__":
    main()