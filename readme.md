# mandelbrot-nsc

Mandelbrot set computation, progressively optimised.
for the *Numerical Scientific Computing* course at AAU.

## Setup

```bash
mamba env create -f environment.yml
mamba activate nsc2026
```

Tested on macOS (Apple Silicon, M4) with Python 3.11.14. The GPU kernel
requires a working OpenCL driver Apple ships one by default. Note that
the M-series GPUs do not expose `cl_khr_fp64`, so the float64 GPU kernel
is detected at runtime and skipped on those devices.

## Running the implementations

Each of the implementation files is runnable on its own and prints a timing
plus saves a PNG:

```bash
python mandelbrot_naive.py
python mandelbrot_numpy.py
python mandelbrot_numba.py
python mandelbrot_parallel.py
python mandelbrot_dask.py
python mandelbrot_gpu.py
```

## Tests and linting

```bash
pytest -v test_mandelbrot.py
pytest --cov=. --cov-report=term-missing test_mandelbrot.py
ruff check mandelbrot_parallel.py
```

The suite covers the four CPU implementations (naive, NumPy, Numba serial,
Numba + multiprocessing); the GPU implementation is not in the suite.

## Performance notebook

```bash
jupyter lab performance_notebook.ipynb
```

## Repository layout

```
mandelbrot_naive.py       
mandelbrot_numpy.py        
mandelbrot_numba.py        
mandelbrot_parallel.py     
mandelbrot_dask.py         
mandelbrot_gpu.py          
mandelbrot_precision.py    
dtype_experiment.py        
benchmark.py               # naive/numpy/numba combined benchmark
profiling.py               # cProfile + line_profiler driver
test_mandelbrot.py         # pytest suite (parametrized + cross-impl)
performance_notebook.ipynb 
environment.yml            # mamba env spec
git_log.txt                # commit history (per submission requirements)
```

## Author

Deian Orlando Petrovics — Master's in Computer Engineering (AI Vision
and Sound), Aalborg University.