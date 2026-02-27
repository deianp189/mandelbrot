import time
import numpy as np
import matplotlib.pyplot as plt


def compute_escape_iterations(c: complex, max_iterations: int) -> int:
    z = 0 + 0j #Initialization at 0

    for iteration in range(max_iterations):
        z = z * z + c
        if abs(z) > 2:
            return iteration

    return max_iterations


def generate_mandelbrot(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iterations: int = 100
) -> np.ndarray:
    image = np.zeros((height, width))

    real_values = np.linspace(xmin, xmax, width)
    imaginary_values = np.linspace(ymin, ymax, height)

    for row in range(height):
        for col in range(width):
            point = complex(real_values[col], imaginary_values[row])
            image[row, col] = compute_escape_iterations(point, max_iterations)

    return image


def plot_mandelbrot(image: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float, elapsed_time: float) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(
        image,
        cmap="hot",
        extent=[xmin, xmax, ymin, ymax],
        origin="lower"
    )
    plt.colorbar(label="Escape iterations")
    plt.title(f"Mandelbrot Set (Naive Implementation) — {elapsed_time:.2f}s")
    plt.savefig("mandelbrot_naive.png")
    plt.show()


def main():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    width, height = 1024, 1024
    max_iterations = 100

    start_time = time.time()
    mandelbrot_image = generate_mandelbrot(
        xmin, xmax, ymin, ymax, width, height, max_iterations
    )
    elapsed_time = time.time() - start_time

    print(f"Grid {width}x{height} computed in {elapsed_time:.3f} seconds")

    plot_mandelbrot(mandelbrot_image, xmin, xmax, ymin, ymax, elapsed_time)


if __name__ == "__main__":
    main()