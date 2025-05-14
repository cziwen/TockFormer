"""
启动器 (launcher.py)

本脚本用于在指定的开始时间启动聚合脚本，并在指定的结束时间自动终止它。

用法示例：
    python launcher.py \
      --script realtime_aggregator.py \
      --start "2025-05-15 09:30:00" \
      --end   "2025-05-15 16:00:00"
"""

import argparse
import subprocess
import time
import os
import signal
from datetime import datetime

def parse_datetime(dt_str: str) -> datetime:
    """
    解析形如 "YYYY-MM-DD HH:MM:SS" 的字符串为 datetime。
    """
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="按时启动并在结束时停止指定脚本")
    parser.add_argument(
        "--script", required=True,
        help="要启动的 Python 脚本路径（例如 realtime_aggregator.py）"
    )
    parser.add_argument(
        "--start", required=True,
        help="启动时间，格式 YYYY-MM-DD HH:MM:SS"
    )
    parser.add_argument(
        "--end", required=True,
        help="结束时间，格式 YYYY-MM-DD HH:MM:SS"
    )
    args = parser.parse_args()

    # 解析时间
    start_dt = parse_datetime(args.start)
    end_dt   = parse_datetime(args.end)
    if end_dt <= start_dt:
        raise ValueError("结束时间必须晚于开始时间")

    # 等待到启动时间
    now = datetime.now()
    if now < start_dt:
        wait_secs = (start_dt - now).total_seconds()
        print(f"[Launcher] 当前时间 {now}, 等待 {wait_secs:.1f}s 到达启动时间 {start_dt}")
        time.sleep(wait_secs)

    # 启动子进程
    abs_script = os.path.abspath(args.script)
    print(f"[Launcher] 在 {datetime.now()} 启动脚本: {abs_script}")
    proc = subprocess.Popen(["python3", abs_script])

    # 等待到结束时间
    now = datetime.now()
    if now < end_dt:
        wait_secs = (end_dt - now).total_seconds()
        print(f"[Launcher] 脚本将在 {wait_secs:.1f}s 后 ({end_dt}) 被终止")
        time.sleep(wait_secs)

    # 结束子进程
    print(f"[Launcher] 在 {datetime.now()} 终止脚本 (PID {proc.pid})")
    try:
        proc.send_signal(signal.SIGINT)
        # 或者用 proc.terminate() / proc.kill() 视需要而定
    except Exception as e:
        print(f"[Launcher] 终止进程时出错: {e}")

if __name__ == "__main__":
    main()