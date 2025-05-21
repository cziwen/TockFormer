import os
import time
import argparse

def generate_synthetic(n: int) -> list[str]:
    """
    生成合成测试数据：'index,value_index\n'，i 从 0 到 n-1。
    """
    return [f"{i},value_{i}\n" for i in range(n)]

def load_source_lines(path: str) -> list[str]:
    """
    读取真实数据文件，返回所有行（保留换行符）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such source file: {path}")
    with open(path, 'r') as f:
        return f.readlines()

def serial_write(filename: str, lines: list[str]) -> float:
    """
    串行写：一次性写入所有行并 fsync。返回耗时（秒）。
    """
    if os.path.exists(filename):
        os.remove(filename)
    t0 = time.time()
    with open(filename, 'w') as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())
    return time.time() - t0

def serial_read(filename: str) -> float:
    """
    串行读：一次性读入内存。返回耗时（秒）。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")
    t0 = time.time()
    with open(filename, 'r') as f:
        _ = f.read()
    return time.time() - t0

def main():
    parser = argparse.ArgumentParser(
        description="Single‐threaded benchmark: generate (or load), write, and read data"
    )
    parser.add_argument(
        '--lines', type=int, default=100_000,
        help='要生成/写入/读取的总行数（默认 100000）'
    )
    parser.add_argument(
        '--source-file', type=str, default=None,
        help='（可选）真实数据文件路径；指定后用该文件内容循环填充测试行'
    )
    args = parser.parse_args()

    n = args.lines

    # 准备要写入的 data 列表
    if args.source_file:
        src = load_source_lines(args.source_file)
        m = len(src)
        if m == 0:
            raise ValueError("Source file is empty.")
        # 循环取模，超出就从头开始
        data = [ src[i % m] for i in range(n) ]
        print(f"[Serial] Loaded {m} lines from '{args.source_file}', cycling to {n} total lines…")
    else:
        print(f"[Serial] Generating {n} synthetic lines…")
        data = generate_synthetic(n)

    # 写入
    print(f"[Serial] Writing to serial_output.txt…")
    t_write = serial_write('serial_output.txt', data)
    print(f"[Serial] Write time: {t_write:.4f} s")

    # 读取
    print(f"[Serial] Reading from serial_output.txt…")
    t_read = serial_read('serial_output.txt')
    print(f"[Serial] Read time:  {t_read:.4f} s")

if __name__ == '__main__':
    main()