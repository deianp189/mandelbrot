import numpy as np
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
import time, statistics
from pathlib import Path
from mandelbrot_parallel import mandelbrot_chunk, mandelbrot_serial



def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                   max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
    
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    N_WORKERS = 10

    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter) 
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

    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

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

    # this code is from the silesd
    # keep LocalCluster open across all measurements

    cluster = LocalCluster(n_workers=N_WORKERS, threads_per_worker=1)
    client = Client(cluster)
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # Dask has higher alpha per task than multiprocessing,
    # so expect the sweet spot at fewer chunks than L05
    chunk_candidates = [4, 8, 10, 16, 20, 32, 40, 64]

    #AI GENERATED CODE
    print(f"\nDask chunk sweep (n_workers={N_WORKERS}, N={N})")
    print(f"{'n_chunks':>10} | {'time (s)':>9} | {'vs 1x':>6} | {'speedup':>8} | {'LIF':>6}")
    print("-" * 52)

    t_baseline = None
    for n_chunks in chunk_candidates:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        if t_baseline is None:
            t_baseline = t_par
        lif = N_WORKERS * t_par / t_serial - 1
        speedup = t_serial / t_par
        vs_1x = t_par / t_baseline
        print(f"{n_chunks:>10} | {t_par:>9.3f} | {vs_1x:>6.2f} | {speedup:>7.2f}x | {lif:>6.2f}")

    client.close()
    cluster.close()