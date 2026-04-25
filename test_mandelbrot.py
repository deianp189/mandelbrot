# test_mandelbrot.py
import numpy as np
import pytest

from mandelbrot_naive import compute_escape_iterations, generate_mandelbrot
from mandelbrot_parallel import mandelbrot_pixel, mandelbrot_serial, mandelbrot_parallel
from mandelbrot_numpy import mandelbrot_numpy

# Bounds used across all grid-level tests
X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5

# ---------------------------------------------------------------------------
# Known cases for the naive pixel function (update z THEN check escape)
#
# c = 0+0j  : z stays at 0 forever             -> max_iter
# c = 5+0j  : i=0: z=5, |5|>2                  -> return 0
# c = -3+0j : i=0: z=-3, |-3|>2                -> return 0
# ---------------------------------------------------------------------------
NAIVE_CASES = [
    (0 + 0j,  100, 100),
    (5 + 0j,  100,   0),
    (-3 + 0j, 100,   0),
]

# ---------------------------------------------------------------------------
# Known cases for the numba pixel function (check escape THEN update z)
#
# c = 0+0j   : z stays at 0 forever             -> max_iter
# c = 5+0j   : i=0: |0|²>4? No, z_real=5
#              i=1: |5|²=25>4                   -> return 1
# c = -2.5+0j: i=0: |0|²>4? No, z_real=-2.5
#              i=1: |-2.5|²=6.25>4              -> return 1
# ---------------------------------------------------------------------------
NUMBA_CASES = [
    (0 + 0j,    100, 100),
    (5 + 0j,    100,   1),
    (-2.5 + 0j, 100,   1),
]


# ---------------------------------------------------------------------------
# Test 1: naive pixel — parametrized over known analytically provable cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c, max_iter, expected", NAIVE_CASES)
def test_naive_pixel(c: complex, max_iter: int, expected: int) -> None:
    assert compute_escape_iterations(c, max_iter) == expected


# ---------------------------------------------------------------------------
# Test 2: numba pixel — parametrized over known analytically provable cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c, max_iter, expected", NUMBA_CASES)
def test_numba_pixel(c: complex, max_iter: int, expected: int) -> None:
    assert mandelbrot_pixel(c.real, c.imag, max_iter) == expected


# ---------------------------------------------------------------------------
# Test 3: numpy grid must match naive on a small grid
# Both use update-then-check, so results are identical pixel by pixel.
# ---------------------------------------------------------------------------
def test_numpy_matches_naive_on_small_grid() -> None:
    N, MAX_ITER = 32, 50
    ref = generate_mandelbrot(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, MAX_ITER)
    got = mandelbrot_numpy(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, MAX_ITER)
    np.testing.assert_array_equal(got, ref.astype(np.int32))


# ---------------------------------------------------------------------------
# Test 4: parallel must match serial on a small grid
# Both call mandelbrot_pixel (check-then-update), so results are identical.
# ---------------------------------------------------------------------------
def test_parallel_matches_serial_on_small_grid() -> None:
    N, MAX_ITER = 32, 50
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    got = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER,
                              n_workers=2, n_chunks=4)
    np.testing.assert_array_equal(got, ref)