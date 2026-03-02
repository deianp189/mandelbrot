"""
Mandelbrot Set Generator
Author: Deian Orlando Petrovics
Course: Numerical Scientific Computing

Naive baseline implementation (Lecture 1).
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def compute_escape_iterations(c, max_iter):
    """
    Escape-time algorithm for one complex point.

    Parameters
    ----------
    c : complex
        Point in the complex plane
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    int
        Number of iterations before divergence
    """
    z = 0 + 0j

    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i

    return max_iter


def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """
    Generate Mandelbrot set using a naive double loop.
    """
    image = np.zeros((height, width))

    real_vals = np.linspace(xmin, xmax, width)
    imag_vals = np.linspace(ymin, ymax, height)

    for row in range(height):
        for col in range(width):
            c = complex(real_vals[col], imag_vals[row])
            image[row, col] = compute_escape_iterations(c, max_iter)

    return image


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    def compute():
        return generate_mandelbrot(
            xmin, xmax, ymin, ymax,
            width, height, max_iter
        )

    # median of 3 runs
    times = []
    for _ in range(3):
        start = time.perf_counter()
        compute()
        end = time.perf_counter()
        times.append(end - start)

    runtime = np.median(times)

    print(f"Naive Mandelbrot 1024x1024 | median runtime = {runtime:.6f} s")

    # compute once more for plotting, this part is not timed
    image = compute()

    plt.imshow(image, cmap="hot", extent=[xmin, xmax, ymin, ymax], origin="lower")
    plt.colorbar()
    plt.title(f"Mandelbrot (Naive) - {runtime:.3f}s")
    plt.savefig("mandelbrot_naive.png")
    plt.show()


if __name__ == "__main__":
    main()