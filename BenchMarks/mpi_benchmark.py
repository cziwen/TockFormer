import os
import time
import argparse
from mpi4py import MPI

# maximum single‐write chunk size to avoid 32‐bit int overflow
_MAX_CHUNK = 2**31 - 1

def generate_data_range(start: int, end: int) -> bytes:
    """
    Generate lines 'index,value_index\n' as UTF-8 bytes for [start, end).
    """
    lines = [f"{i},value_{i}\n" for i in range(start, end)]
    return ''.join(lines).encode('utf-8')

def load_source_lines(path: str) -> list[str]:
    """
    Read a real data file and return its lines (including newline).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such source file: {path}")
    with open(path, 'r') as f:
        return f.readlines()

def mpi_write(filename: str, chunk: bytes, offset: int, comm: MPI.Comm) -> float:
    """
    Collective MPI-IO write of `chunk` at byte `offset`.
    Splits into <= _MAX_CHUNK pieces to avoid overflow.
    Returns elapsed time (secs) on rank 0.
    """
    rank = comm.Get_rank()
    if rank == 0 and os.path.exists(filename):
        os.remove(filename)
    comm.Barrier()

    if rank == 0:
        t0 = time.time()
    comm.Barrier()

    fh = MPI.File.Open(comm, filename, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    remaining = chunk
    curr_off = offset
    while remaining:
        part = remaining[:_MAX_CHUNK]
        fh.Write_at_all(curr_off, part)
        curr_off += len(part)
        remaining = remaining[len(part):]
    fh.Sync()
    fh.Close()

    comm.Barrier()
    if rank == 0:
        return time.time() - t0
    return None

def mpi_read(filename: str, offset: int, length: int, comm: MPI.Comm) -> float:
    """
    Collective MPI-IO read of `length` bytes at byte `offset`.
    Returns elapsed time (secs) on rank 0.
    """
    rank = comm.Get_rank()
    if rank == 0 and not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    comm.Barrier()

    if rank == 0:
        t0 = time.time()
    comm.Barrier()

    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
    buf = bytearray(length)
    fh.Read_at_all(offset, buf)
    fh.Close()

    comm.Barrier()
    if rank == 0:
        return time.time() - t0
    return None

def main():
    parser = argparse.ArgumentParser(
        description="MPI benchmark: generate (or load), write and read data"
    )
    parser.add_argument(
        '--lines', type=int, default=100_000,
        help='Total number of lines to generate/load and I/O (default: 100k)'
    )
    parser.add_argument(
        '--source-file', type=str, default=None,
        help='(optional) Path to a real data file; its lines will be cycled to fill the requested range'
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = args.lines

    # determine per-rank slice
    base = N // size
    rem  = N % size
    counts = [base + (1 if i < rem else 0) for i in range(size)]
    my_count   = counts[rank]
    start_idx  = sum(counts[:rank])
    end_idx    = start_idx + my_count

    # generation or load + chunk assembly
    t0g = time.time()
    if args.source_file:
        src_lines = load_source_lines(args.source_file)
        m = len(src_lines)
        if m == 0:
            raise ValueError("Source file is empty.")
        # cycle through source lines
        lines = [src_lines[i % m] for i in range(start_idx, end_idx)]
        chunk = ''.join(lines).encode('utf-8')
        print(f"[Rank {rank}] Loaded {m} lines from '{args.source_file}', cycling to {my_count} total lines…")
    else:
        print(f"[Rank {rank}] Generating {my_count} synthetic lines…")
        chunk = generate_data_range(start_idx, end_idx)
    t1g = time.time()
    print(f"[Rank {rank}] Generation time: {t1g - t0g:.4f} s")

    # gather lengths and compute offsets
    my_len  = len(chunk)
    all_lens = comm.allgather(my_len)
    offset   = sum(all_lens[:rank])

    # collective write
    t_write = mpi_write('mpi_output.dat', chunk, offset, comm)
    if rank == 0:
        print(f"[MPI] Collective write time ({size} ranks): {t_write:.4f} s")

    # collective read
    t_read = mpi_read('mpi_output.dat', offset, my_len, comm)
    if rank == 0:
        print(f"[MPI] Collective read  time ({size} ranks): {t_read:.4f} s")

if __name__ == '__main__':
    main()