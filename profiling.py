"""
Milestone 1 - cProfile
Author: Deian Orlando Petrovics
Course: NSC

Goal:
- Profile naive vs NumPy at a smaller resolution (512x512) to keep it reasonable
"""

import cProfile
import pstats

from mandelbrot_naive import generate_mandelbrot
from mandelbrot_numpy import mandelbrot_numpy


def profile_one(label, statement, out_file):
    print(f"\n--- Profiling {label} ---")
    cProfile.run(statement, out_file)

    stats = pstats.Stats(out_file)
    stats.sort_stats("cumulative")
    stats.print_stats(15)  # top 15 is usually enough


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 512, 512
    max_iter = 100

    # We profile just the compute functions (no plotting)
    profile_one(
        "Naive (L1)",
        f"generate_mandelbrot({xmin}, {xmax}, {ymin}, {ymax}, {width}, {height}, {max_iter})",
        "naive.prof"
    )

    profile_one(
        "NumPy (L2)",
        f"mandelbrot_numpy({xmin}, {xmax}, {ymin}, {ymax}, {width}, {height}, {max_iter})",
        "numpy.prof"
    )


if __name__ == "__main__":
    main()