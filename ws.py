import time
import websocket
import json
import sys

# 替换成你的真实 API key
API_KEY = "cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00"


def on_message(ws, message):
    data = json.loads(message)
    for trade in data.get("data", []):
        sys.stdout.write("\r")  # 回到行首
        sys.stdout.write(
            f"⏱ {time.strftime('%H:%M:%S')} | "
            f"🪙 {trade['s']} | 💰 Price: {trade['p']:.2f} | 📦 Volume: {trade['v']}"
        )
        sys.stdout.flush()


def on_open(ws):
    print("连接成功，开始订阅...")
    # 订阅 Binance 的 BTC/USDT 价格
    ws.send(json.dumps({
        "type": "subscribe",
        "symbol": "BINANCE:BTCUSDT"
    }))

def on_error(ws, error):
    print("发生错误:", error)

def on_close(ws, close_status_code, close_msg):
    print("连接关闭")

if __name__ == "__main__":
    socket = f"wss://ws.finnhub.io?token={API_KEY}"
    ws = websocket.WebSocketApp(socket,
                                 on_message=on_message,
                                 on_open=on_open,
                                 on_error=on_error,
                                 on_close=on_close)
    ws.run_forever()