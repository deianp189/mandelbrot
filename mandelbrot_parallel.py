import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import matplotlib.pyplot as plt
from pathlib import Path


# cache=True saves compiled code to __pycache__ so workers don't recompile on startup
@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0:
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


# Must be module-level so multiprocessing can pickle it
def _worker(args):
    return mandelbrot_chunk(*args)


# M1: added n_chunks and pool parameters
def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
                        max_iter=100, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers

    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    if pool is not None:
        # caller manages the Pool so no spawn cost, no warm-up needed here
        return np.vstack(pool.map(_worker, chunks))

    # tiny warm-up task to load jitcache in workers before the real run
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # warm up jit in the main process so cache= True saves before workers start
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.3f}s")

    # quick check inparallel result matches serial
    result_serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    result_parallel = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                                          n_workers=4, n_chunks=16)
    assert np.array_equal(result_serial, result_parallel), "M1 correctness check failed"
    print("M1 correctness check passed")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result_parallel, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
              cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    out = Path(__file__).parent / 'mandelbrot_parallel.png'
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')

    # --- M2: chunk count sweep with LIF (from L05 slides) ---
    n_workers = os.cpu_count() // 2  # adjust to your L04 optimum
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]

    print(f"\nChunk sweep (n_workers={n_workers}, N={N})")
    print(f"{'n_chunks':>10} | {'time (s)':>9} | {'speedup':>8} | {'LIF':>6}")
    print("-" * 44)

    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)  # warm-up: load JIT cache in workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                                    n_workers=n_workers, n_chunks=n_chunks, pool=pool)
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        speedup = t_serial / t_par
        print(f"{n_chunks:>10} | {t_par:>9.3f} | {speedup:>7.2f}x | {lif:>6.2f}")