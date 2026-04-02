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

    # serial baseline
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.3f}s")

    # one cluster for all milestones
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

    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    assert np.array_equal(ref, result), "M1 correctness check failed"
    print("M1 correctness check passed")

    chunk_candidates = [4, 8, 10, 16, 20, 32, 40, 64]

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


    #this part is from the slides and i also used ai to improve upon it since it was not working for some reason 
    from mandelbrot_naive import generate_mandelbrot
    from mandelbrot_numpy import mandelbrot_numpy
    from mandelbrot_parallel import mandelbrot_parallel, _worker
    from multiprocessing import Pool

    print(f"\nFull implementation comparison (N={N}, max_iter={max_iter})")
    print(f"{'Implementation':>25} | {'time (s)':>9} | {'speedup':>8}")
    print("-" * 50)

    t0 = time.perf_counter()
    generate_mandelbrot(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, max_iter)
    t_naive = time.perf_counter() - t0
    print(f"{'Naive Python':>25} | {t_naive:>9.3f} | {'1.00x':>8}")

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_numpy(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, max_iter)
        times.append(time.perf_counter() - t0)
    t_numpy = statistics.median(times)
    print(f"{'NumPy':>25} | {t_numpy:>9.3f} | {t_naive/t_numpy:>7.2f}x")

    print(f"{'Numba (@njit)':>25} | {t_serial:>9.3f} | {t_naive/t_serial:>7.2f}x")

    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    with Pool(processes=5) as pool:
        pool.map(_worker, tiny)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                                n_workers=5, n_chunks=10, pool=pool)
            times.append(time.perf_counter() - t0)
    t_mp = statistics.median(times)
    print(f"{'Numba + multiprocessing':>25} | {t_mp:>9.3f} | {t_naive/t_mp:>7.2f}x")

    n_chunks_optimal = 10  # sweet spot from milestone 2
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks_optimal)
        times.append(time.perf_counter() - t0)
    t_dask_opt = statistics.median(times)
    print(f"{'Dask local':>25} | {t_dask_opt:>9.3f} | {t_naive/t_dask_opt:>7.2f}x")
    print(f"{'Dask cluster':>25} | {'(L7)':>9} | {'(L7)':>8}")

    # close once at the very end
    client.close()
    cluster.close()