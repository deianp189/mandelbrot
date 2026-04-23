"""
Mandelbrot Set Generator (Lecture 2)
Author: Deian Orlando Petrovics
Course: Numerical Scientific Computing

Goal: get a big speedup using NumPy vectorization (remove pixel loops).
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """
    Vectorized Mandelbrot (keep only the iteration loop).

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Complex plane bounds
    width, height : int
        Image resolution
    max_iter : int
        Max escape iterations

    Returns
    -------
    np.ndarray
        Escape iteration counts
    """

    """Basically when we do it manually we do 1024x2014=1048576 calls, using a grid we do them
    all at once, this will also allow to get rid of the for loop that iterates over them
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)

    # Mask is for points that are still "active", basically points not escaped yet
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]

        escaped_now = (np.abs(Z) > 2) & mask
        M[escaped_now] = i
        mask[escaped_now] = False

    # points that never escaped keep max_iter
    M[mask] = max_iter
    return M


def benchmark_median(func, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    def compute():
        return mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter)

    t_med = benchmark_median(compute, runs=3)
    print(f"NumPy Mandelbrot | {width}x{height} | max_iter={max_iter} | median = {t_med:.6f} s")

    img = compute()  # not timed

    #The plot code is AI generated
    plt.imshow(img, cmap="magma", extent=[xmin, xmax, ymin, ymax], origin="lower")
    plt.colorbar()
    plt.title(f"Mandelbrot (NumPy L2) - {t_med:.3f}s")
    plt.savefig("mandelbrot_l2.png")
    plt.show()


if __name__ == "__main__":
    main()