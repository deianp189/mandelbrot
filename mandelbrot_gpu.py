"""
Mandelbrot GPU — PyOpenCL float32 + float64 implementation.
Author: Deian Orlando Petrovics
Course: Numerical Scientific Computing
MP3 M1 and M2.
"""
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

# --- Step 1: context and command queue ---
ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
print(f"Device: {ctx.devices[0].name}")

# --- Step 2: check fp64 support (M2) ---
dev = ctx.devices[0]
if "cl_khr_fp64" not in dev.extensions:
    print("No native fp64 -- Apple Silicon: emulated, expect large slowdown")

# ---------------------------------------------------------------------------
# Kernel source — float32 (M1)
# ---------------------------------------------------------------------------
KERNEL_F32 = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# ---------------------------------------------------------------------------
# Kernel source — float64 (M2)
# ---------------------------------------------------------------------------
KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# --- Compile kernels ---
prog_f32 = cl.Program(ctx, KERNEL_F32).build()
knl_f32  = cl.Kernel(prog_f32, "mandelbrot")  # reuse to avoid warning

fp64_supported = "cl_khr_fp64" in dev.extensions
prog_f64 = None
knl_f64  = None
if fp64_supported:
    prog_f64 = cl.Program(ctx, KERNEL_F64).build()
    knl_f64  = cl.Kernel(prog_f64, "mandelbrot_f64")
else:
    print("fp64 kernel skipped: not natively supported on this device (Apple M4)")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N        = 1024
MAX_ITER = 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

image     = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)


# ---------------------------------------------------------------------------
# Helper: timed kernel launch
# ---------------------------------------------------------------------------
def timed(fn, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        queue.finish()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------------------------------------------------------------------------
# M1 — warm up then time float32 at N=1024
# ---------------------------------------------------------------------------
knl_f32(queue, (64, 64), None, image_dev,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(64), np.int32(MAX_ITER))
queue.finish()

t_f32 = timed(lambda: knl_f32(
    queue, (N, N), None, image_dev,
    np.float32(X_MIN), np.float32(X_MAX),
    np.float32(Y_MIN), np.float32(Y_MAX),
    np.int32(N), np.int32(MAX_ITER),
))
print(f"GPU f32 {N}x{N}: {t_f32*1e3:.1f} ms")

cl.enqueue_copy(queue, image, image_dev)
queue.finish()

plt.imshow(image, cmap="hot", origin="lower")
plt.axis("off")
plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches="tight")
print("Saved: mandelbrot_gpu.png")

# ---------------------------------------------------------------------------
# M2 — float32 vs float64 at N=1024 and N=2048
# ---------------------------------------------------------------------------
for n in [1024, 2048]:
    image_n     = np.zeros((n, n), dtype=np.int32)
    image_dev_n = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image_n.nbytes)

    t_f32_n = timed(lambda: knl_f32(
        queue, (n, n), None, image_dev_n,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(n), np.int32(MAX_ITER),
    ))
    print(f"N={n}: f32={t_f32_n*1e3:.1f}ms")

    if knl_f64 is not None:
        t_f64_n = timed(lambda: knl_f64(
            queue, (n, n), None, image_dev_n,
            np.float64(X_MIN), np.float64(X_MAX),
            np.float64(Y_MIN), np.float64(Y_MAX),
            np.int32(n), np.int32(MAX_ITER),
        ))
        print(f"N={n}: f64={t_f64_n*1e3:.1f}ms  ratio={t_f64_n/t_f32_n:.1f}x")
    else:
        print(f"N={n}: f64=N/A (emulated fp64 not supported on Apple M4)")