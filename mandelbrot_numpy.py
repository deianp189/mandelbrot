import time
import numpy as np
import matplotlib.pyplot as plt

def generate_mandelbrot_vectorized(
    xmin: float, xmax: float, ymin: float, ymax: float, 
    width: int, height: int, max_iterations: int = 100
) -> np.ndarray:
    """
    Versión optimizada de la Lecture 2 usando vectorización con NumPy.
    Elimina los bucles por píxel para ganar velocidad.
    """
    # 1. Crear la cuadrícula de números complejos (Meshgrid)
    # En lugar de ir píxel a píxel, creamos toda la "hoja" de golpe
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y # Matriz completa de puntos complejos

    # 2. Inicializar variables de cálculo
    Z = np.zeros_like(C)
    counts = np.full(C.shape, max_iterations, dtype=int)
    mask = np.full(C.shape, True, dtype=bool) # Máscara para puntos activos

    # 3. Bucle de iteraciones (Solo 100 vueltas, no 1 millón de píxeles)
    for i in range(max_iterations):
        # Calculamos solo para los puntos que NO han escapado todavía
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        
        # Comprobar quién ha superado el límite de escape |z| > 2
        escaped = np.abs(Z) > 2
        
        # Identificar quién acaba de escapar en esta vuelta exacta
        newly_escaped = escaped & mask
        
        # Anotar la iteración de escape
        counts[newly_escaped] = i
        
        # Actualizar la máscara: los que escaparon ya no se procesan
        mask[escaped] = False

    return counts

def plot_comparison(image: np.ndarray, title: str, elapsed: float, filename: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap="magma", origin="lower")
    plt.colorbar(label="Iteraciones de escape")
    plt.title(f"{title} — Tiempo: {elapsed:.4f}s")
    plt.savefig(filename)
    plt.show()

def main():
    # Configuración de la sesión de estudio
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iterations = 100

    print(f"Hardware detectado: Apple M4 (ARM64)")
    print(f"Calculando Mandelbrot {width}x{height}...")

    # Ejecutar versión vectorizada (Lecture 2)
    start = time.time()
    image_vectorized = generate_mandelbrot_vectorized(
        xmin, xmax, ymin, ymax, width, height, max_iterations
    )
    end = time.time()
    
    elapsed_vectorized = end - start
    print(f"Versión Vectorizada (NumPy): {elapsed_vectorized:.4f} segundos")

    # Visualización
    plot_comparison(image_vectorized, "Mandelbrot Vectorized (L2)", elapsed_vectorized, "mandelbrot_l2.png")

if __name__ == "__main__":
    main()