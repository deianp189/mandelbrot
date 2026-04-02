import numpy as np
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
import time, statistics
from pathlib import Path

# mandelbrot_chunk and mandelbrot_serial are imported from the parallel file
# so we don't duplicate the @njit functions
from mandelbrot_parallel import mandelbrot_chunk, mandelbrot_serial


# --- M1: Dask Mandelbrot with dask.delayed (from L06 slides) ---
def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                   max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        # delayed wraps the entire Numba function as one atomic task per chunk
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    N_WORKERS = 10

    # serial baseline for LIF calculation
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)  # warm up JIT in main process
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.3f}s")

    cluster = LocalCluster(n_workers=N_WORKERS, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

    # warm up Numba JIT in all workers before timing
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # M1: time Dask Mandelbrot with n_chunks=32
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=32)
        times.append(time.perf_counter() - t0)
    t_dask = statistics.median(times)
    print(f"Dask local (n_chunks=32): {t_dask:.3f}s")

    # correctness check against serial
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    assert np.array_equal(ref, result), "M1 correctness check failed"
    print("M1 correctness check passed")

    client.close()
    cluster.close()