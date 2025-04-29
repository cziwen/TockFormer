import json
import threading
import websocket
import time

# 直接在这里填写你的 Finnhub API Key
FINNHUB_TOKEN = "cvop3lhr01qihjtq3uvgcvop3lhr01qihjtq3v00"
WS_URL = f"wss://ws.finnhub.io?token={FINNHUB_TOKEN}"

# 要订阅的股票列表，例如 Apple 和 Google
SYMBOLS = ["AAPL", "GOOGL", "TSLA"]

def on_message(ws, message):
    """
    收到消息时回调
    格式示例：
      {"data":[{"p":177.85,"s":"AAPL","t":1621373528000,"v":100}], "type":"trade"}
    """
    msg = json.loads(message)
    if msg.get("type") == "trade":
        for trade in msg["data"]:
            symbol = trade["s"]
            price  = trade["p"]
            volume = trade["v"]
            ts     = trade["t"]
            print(f"[{symbol}] Price={price}  Volume={volume}  Timestamp={ts}")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("### WebSocket closed ###", close_status_code, close_msg)

def on_open(ws):
    """
    连接建立后订阅所有 SYMBOLS
    """
    def run():
        for sym in SYMBOLS:
            sub_msg = json.dumps({"type": "subscribe", "symbol": sym})
            ws.send(sub_msg)
            print(f"Subscribed to {sym}")
            time.sleep(0.1)

    threading.Thread(target=run).start()

if __name__ == "__main__":
    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws_app.run_forever()