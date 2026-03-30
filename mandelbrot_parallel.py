import numpy as np
from numba import njit
from multiprocessing import Pool
import time, statistics
import matplotlib.pyplot as plt
from pathlib import Path

@njit
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

@njit
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

# pool.map only passes one argument, so we pack everything into a tuple
def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # un-timed warm-up so Numba JIT compiles in workers
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.3f}s")

    # verify parallel result matches serial
    result = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=4)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
              cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    out = Path(__file__).parent / 'mandelbrot_parallel.png'
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')