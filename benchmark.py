"""
Final benchmark for performance tracker
Author: Deian Orlando Petrovics
Course: NSC
"""

import time
import numpy as np

from mandelbrot_naive import generate_mandelbrot
from mandelbrot_numpy import mandelbrot_numpy
from mandelbrot_numba import mandelbrot_numba


def bench_median(fn, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    def run_naive():
        return generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

    def run_numpy():
        return mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter)

    # Warm-up numba signature (do not time)
    _ = mandelbrot_numba(xmin, xmax, ymin, ymax, 64, 64, max_iter)

    def run_numba():
        return mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter)

    t1 = bench_median(run_naive, runs=3)
    t2 = bench_median(run_numpy, runs=3)
    t3 = bench_median(run_numba, runs=3)

    print(f"Naive (L1)  : {t1:.6f} s")
    print(f"NumPy (L2)  : {t2:.6f} s")
    print(f"Numba (L3)  : {t3:.6f} s")
    print(f"Speedup NumPy vs Naive: {t1/t2:.2f}x")
    print(f"Speedup Numba vs Naive: {t1/t3:.2f}x")


if __name__ == "__main__":
    main()