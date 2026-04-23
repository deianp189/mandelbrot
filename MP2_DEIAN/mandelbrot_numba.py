"""
Mandelbrot L3 – Numba JIT version
Author: Deian Orlando Petrovics
Course: NSC

Goal:
- Compare naive, NumPy, and Numba
- Measure warm-up time explicitly
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# -------------------------------------------------------
# 1. Naive Python version
# -------------------------------------------------------

def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0

            while n < max_iter and abs(z) <= 2:
                z = z*z + c
                n += 1

            result[i, j] = n

    return result


# -------------------------------------------------------
# 2. NumPy version
# -------------------------------------------------------

def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100):

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask]*Z[mask] + C[mask]
        escaped = (np.abs(Z) > 2) & mask
        M[escaped] = i
        mask[escaped] = False

    M[mask] = max_iter
    return M


# -------------------------------------------------------
# 3. Fully compiled Numba version
# -------------------------------------------------------

@njit(cache=True)   # cache=True lets it persist between runs
def mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0

            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z*z + c
                n += 1

            result[i, j] = n

    return result


# -------------------------------------------------------
# 4. Benchmark helper
# -------------------------------------------------------

def bench(fn, *args, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return np.median(times)


# -------------------------------------------------------
# 5. Main experiment
# -------------------------------------------------------

def main():

    #The benchmark code is AI-generated

    args = (-2.0, 1.0, -1.5, 1.5, 1024, 1024)

    print("\n--- Measuring warm-up (Numba first call) ---")
    t0 = time.perf_counter()
    mandelbrot_numba(*args)
    warmup_time = time.perf_counter() - t0
    print(f"Numba first call (includes compilation): {warmup_time:.6f} s")

    print("\n--- Benchmarking steady-state ---")
    t_naive = bench(mandelbrot_naive, *args)
    t_numpy = bench(mandelbrot_numpy, *args)
    t_numba = bench(mandelbrot_numba, *args)

    print(f"\nNaive  : {t_naive:.6f} s")
    print(f"NumPy  : {t_numpy:.6f} s  ({t_naive/t_numpy:.1f}x faster)")
    print(f"Numba  : {t_numba:.6f} s  ({t_naive/t_numba:.1f}x faster)")


if __name__ == "__main__":
    main()