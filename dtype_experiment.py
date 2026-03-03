"""
Milestone 4 - Data type experiment (float32 and float64)
Author: Deian Orlando Petrovics
Course: NSC

Goal:
- Benchmark Numba Mandelbrot with float32 and float64, 4bytes and 8 bytes
- Compare runtime and try to find output differences
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit(cache=True)
def mandelbrot_numba_f32(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(np.float32(xmin), np.float32(xmax), width).astype(np.float32)
    y = np.linspace(np.float32(ymin), np.float32(ymax), height).astype(np.float32)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = np.complex64(x[j]) + np.complex64(1j) * np.complex64(y[i])
            z = np.complex64(0.0) + np.complex64(0.0j)

            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= np.float32(4.0):
                z = z * z + c
                n += 1

            result[i, j] = n

    return result


@njit(cache=True)
def mandelbrot_numba_f64(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(np.float64(xmin), np.float64(xmax), width).astype(np.float64)
    y = np.linspace(np.float64(ymin), np.float64(ymax), height).astype(np.float64)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = np.complex128(x[j]) + np.complex128(1j) * np.complex128(y[i])
            z = np.complex128(0.0) + np.complex128(0.0j)

            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= np.float64(4.0):
                z = z * z + c
                n += 1

            result[i, j] = n

    return result


def bench_median(fn, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


#AI Generated
def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    # Warm-up (compile) - do not time this in the medians
    _ = mandelbrot_numba_f32(xmin, xmax, ymin, ymax, 64, 64, max_iter)
    _ = mandelbrot_numba_f64(xmin, xmax, ymin, ymax, 64, 64, max_iter)

    def run_f32():
        return mandelbrot_numba_f32(xmin, xmax, ymin, ymax, width, height, max_iter)

    def run_f64():
        return mandelbrot_numba_f64(xmin, xmax, ymin, ymax, width, height, max_iter)

    t_f32 = bench_median(run_f32, runs=5)
    t_f64 = bench_median(run_f64, runs=5)

    img_f32 = run_f32()
    img_f64 = run_f64()

    diff = np.abs(img_f32.astype(np.int32) - img_f64.astype(np.int32))

    print(f"float32 median time: {t_f32:.6f} s")
    print(f"float64 median time: {t_f64:.6f} s")
    print(f"speedup (f64/f32): {t_f64/t_f32:.2f}x")
    print(f"diff stats: max={diff.max()}, mean={diff.mean():.4f}, changed_pixels={(diff>0).sum()}")

    # Plot side-by-side and diff
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_f32, cmap="magma", origin="lower")
    plt.title("Numba float32")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_f64, cmap="magma", origin="lower")
    plt.title("Numba float64")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="viridis", origin="lower")
    plt.title("abs(diff)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("dtype_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()