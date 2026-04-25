# mandelbrot_parallel.py
"""
Mandelbrot Set — Numba + multiprocessing parallel implementation.
Author: Deian Orlando Petrovics
Course: Numerical Scientific Computing
"""

from __future__ import annotations

import os
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from mandelbrot_naive import generate_mandelbrot
from mandelbrot_numpy import mandelbrot_numpy


# cache=True saves compiled code to __pycache__ so workers don't recompile on startup
@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """Compute the escape iteration count for one complex point.

    Uses the optimised real-arithmetic recurrence to avoid Python complex
    overhead. Checks escape before updating z, so iteration counts may
    differ by one from implementations that update first.

    Parameters
    ----------
    c_real : float
        Real part of the complex coordinate to test.
    c_imag : float
        Imaginary part of the complex coordinate to test.
    max_iter : int
        Maximum iterations; returned if the trajectory does not escape.

    Returns
    -------
    int
        First iteration i where abs(z)^2 > 4, or max_iter.
    """
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
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> np.ndarray:
    """Compute escape counts for a horizontal slice of the complex grid.

    Parameters
    ----------
    row_start : int
        First row index (inclusive).
    row_end : int
        Last row index (exclusive).
    N : int
        Grid side length in pixels.
    x_min, x_max : float
        Real-axis bounds of the complex plane.
    y_min, y_max : float
        Imaginary-axis bounds of the complex plane.
    max_iter : int
        Maximum iterations per pixel.

    Returns
    -------
    np.ndarray
        Integer array of shape (row_end - row_start, N) with escape counts.
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out


def mandelbrot_serial(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute the full Mandelbrot grid in a single Numba-compiled call.

    Parameters
    ----------
    N : int
        Grid side length in pixels (produces an N x N image).
    x_min, x_max : float
        Real-axis bounds of the complex plane.
    y_min, y_max : float
        Imaginary-axis bounds of the complex plane.
    max_iter : int, optional
        Maximum iterations per pixel (default 100).

    Returns
    -------
    np.ndarray
        Integer array of shape (N, N) with escape counts.
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args: tuple) -> np.ndarray:
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    n_workers: int = 4,
    n_chunks: int | None = None,
    pool: Pool | None = None,
) -> np.ndarray:
    """Compute the Mandelbrot grid using multiprocessing over row chunks.

    Parameters
    ----------
    N : int
        Grid side length in pixels (produces an N x N image).
    x_min, x_max : float
        Real-axis bounds of the complex plane.
    y_min, y_max : float
        Imaginary-axis bounds of the complex plane.
    max_iter : int, optional
        Maximum iterations per pixel (default 100).
    n_workers : int, optional
        Number of worker processes (default 4).
    n_chunks : int or None, optional
        Number of row chunks to distribute. Defaults to n_workers.
    pool : Pool or None, optional
        An existing Pool to reuse. If None a new Pool is created and
        a small warm-up job is submitted before the main computation.

    Returns
    -------
    np.ndarray
        Integer array of shape (N, N) with escape counts.
    """
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

    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)
    return np.vstack(parts)


if __name__ == "__main__":
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # warm up jit in the main process so cache=True saves before workers start
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.3f}s")

    result_serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    result_parallel = mandelbrot_parallel(
        N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=4, n_chunks=16
    )
    assert np.array_equal(result_serial, result_parallel), "M1 correctness check failed"
    print("M1 correctness check passed")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        result_parallel,
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="inferno",
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    out = Path(__file__).parent / "mandelbrot_parallel.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    n_workers = os.cpu_count() // 2
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]

    print(f"\nChunk sweep (n_workers={n_workers}, N={N})")
    print(f"{'n_chunks':>10} | {'time (s)':>9} | {'speedup':>8} | {'LIF':>6}")
    print("-" * 44)

    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(
                    N,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    max_iter,
                    n_workers=n_workers,
                    n_chunks=n_chunks,
                    pool=pool,
                )
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        speedup = t_serial / t_par
        print(f"{n_chunks:>10} | {t_par:>9.3f} | {speedup:>7.2f}x | {lif:>6.2f}")

    print(f"\nFull implementation comparison (N={N}, max_iter={max_iter})")
    print(f"{'Implementation':>20} | {'time (s)':>9} | {'speedup':>8}")
    print("-" * 46)

    t0 = time.perf_counter()
    generate_mandelbrot(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, max_iter)
    t_naive = time.perf_counter() - t0
    print(f"{'Naive Python':>20} | {t_naive:>9.3f} | {'1.00x':>8}")

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_numpy(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, max_iter)
        times.append(time.perf_counter() - t0)
    t_numpy = statistics.median(times)
    print(f"{'NumPy':>20} | {t_numpy:>9.3f} | {t_naive / t_numpy:>7.2f}x")
    print(f"{'Numba':>20} | {t_serial:>9.3f} | {t_naive / t_serial:>7.2f}x")

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_parallel(
                N,
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                max_iter,
                n_workers=n_workers,
                n_chunks=2 * n_workers,
                pool=pool,
            )
            times.append(time.perf_counter() - t0)
    t_par = statistics.median(times)
    print(f"{'Parallel (opt.)':>20} | {t_par:>9.3f} | {t_naive / t_par:>7.2f}x")

    print(f"\nWorker sweep (n_chunks = 2 x n_workers, N={N})")
    print(f"{'workers':>8} | {'time (s)':>9} | {'speedup':>8} | {'efficiency':>11}")
    print("-" * 48)

    for nw in range(1, os.cpu_count() + 1):
        n_chunks = 2 * nw
        with Pool(processes=nw) as pool:
            pool.map(_worker, tiny)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(
                    N,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    max_iter,
                    n_workers=nw,
                    n_chunks=n_chunks,
                    pool=pool,
                )
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        efficiency = speedup / nw * 100
        print(f"{nw:>8} | {t_par:>9.3f} | {speedup:>7.2f}x | {efficiency:>10.0f}%")
