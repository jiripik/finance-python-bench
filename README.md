<div align="center">

# 🐍 Finance Python Benchmark Suite

**Data-driven Docker image selection for hedge fund batch workloads**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Quick Start](#-quick-start) • [Benchmarks](#-benchmark-categories) • [Decision Guide](#-decision-tree-which-image-should-you-use) • [Results](#-interpreting-results) • [Contributing](#-contributing)

</div>

---

## 📊 Executive Summary

> **TL;DR**: For most finance workloads, `python:3.14-slim` (~150MB) performs within 10% of larger images while being 20× smaller. Use Intel Python or Anaconda only for BLAS-heavy linear algebra on Intel CPUs.

This benchmark suite provides **empirical, reproducible data** to guide Docker base image selection for Python data workloads. We test three popular images across finance-oriented tasks:

| Image | Size | Python | BLAS | Best For |
|-------|------|--------|------|----------|
| `python:3.14-slim` | ~150MB | 3.14 | OpenBLAS | General workloads, CI/CD, AMD CPUs |
| `intel/python:latest` | ~2.8GB | 3.12 | Intel MKL | BLAS-heavy on Intel CPUs |
| `continuumio/anaconda3:latest` | ~3.5GB | 3.12 | Intel MKL | Conda environments, pre-installed packages |

### Real-World Impact

> *"We were running a critical end-of-day batch job on Intel Python—processing hundreds of millions of rows across our asset universe, computing risk metrics, and generating portfolio summaries. The code is multi-threaded to saturate all available CPUs, using ProcessPoolExecutor for the heavy lifting. The job consistently took 5 minutes. When we migrated to `python:3.14-slim`, the same job completed in under 3 minutes. No code changes. No algorithm tweaks. Just a different base image.*
>
> *The 40% runtime reduction isn't just about compute costs—though those savings are real. It's about decision velocity. When markets move, portfolio managers need updated risk numbers now, not in five minutes. Shaving two minutes off our batch window means traders get actionable intelligence faster, and that edge compounds across hundreds of decisions per day.*
>
> *Smaller image, faster pulls, faster execution, faster decisions. Sometimes the 'boring' choice wins."*
>
> — Technology Director, Systematic Hedge Fund

---

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- Bash shell (Linux/macOS/WSL)

### Run All Benchmarks

```bash
# Clone the repository
git clone https://github.com/jiripik/finance-python-bench.git
cd finance-python-bench

# Run full benchmark suite (takes 15-30 minutes)
./scripts/run_all.sh

# View results
open results/report.html  # macOS
xdg-open results/report.html  # Linux
```

### Configuration Options

```bash
# Environment variables
SCALE=1.0          # Data size multiplier (0.5 = smaller, 2.0 = larger)
THREAD_SETS="1 8"  # Thread configurations to test

# Example: Quick test with smaller data
SCALE=0.5 THREAD_SETS="1" ./scripts/run_all.sh

# Example: Production benchmark with larger data
SCALE=2.0 THREAD_SETS="1 4 8 16" ./scripts/run_all.sh
```

### Run Single Image Manually

```bash
# Inside a container with requirements installed
python benchmarks/run_suite.py \
  --label my-env \
  --output results/my-env.json \
  --scale 1.0 \
  --threads 8

# Generate HTML report
python benchmarks/report.py \
  --results results/*.json \
  --output results/report.html
```

---

## 🎯 Decision Tree: Which Image Should You Use?

### Step 1: What CPU Are You Running On?

#### AMD Processors (Ryzen, EPYC, Threadripper)

```
✅ Always use python:3.14-slim
```

**Why**: Intel MKL (used by `intel/python` and `anaconda`) contains CPU vendor detection that can throttle performance on AMD hardware. OpenBLAS is vendor-neutral and often *faster* than MKL on AMD.

#### Intel Processors (Xeon, Core)

```
├─► BLAS-heavy workload (>50% runtime in matrix ops)?
│     ├─► YES → Consider intel/python or anaconda
│     └─► NO  → Use python:3.14-slim
│
└─► Need conda package management?
      ├─► YES → Use anaconda (accept 3.5GB image)
      └─► NO  → Use python:3.14-slim
```

### Step 2: Threading Strategy

| Workload Type | Threading Works? | Recommended Approach |
|---------------|------------------|---------------------|
| **IO-bound** (file/network) | ✅ Yes | `ThreadPoolExecutor` |
| **CPU-bound Python** | ❌ GIL blocks | `ProcessPoolExecutor` |
| **NumPy/BLAS** | ✅ Internal threads | Set `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` |
| **Polars ETL** | ✅ Rust threads | Set `POLARS_MAX_THREADS` |
| **Pandas ETL** | ⚠️ Mostly single-threaded | Consider Polars instead |

### Quick Reference Matrix

| Your Situation | Recommended Image |
|----------------|-------------------|
| AMD CPU, any workload | `python:3.14-slim` |
| Intel CPU, general workload | `python:3.14-slim` |
| Intel CPU, heavy linear algebra | `intel/python` |
| Need conda environments | `anaconda3` |
| CI/CD pipelines | `python:3.14-slim` |
| Serverless/Lambda | `python:3.14-slim` |

---

## 📈 Benchmark Categories

The suite tests **5 workload categories** with **15+ individual benchmarks**, each designed to stress different aspects of Python performance.

### 1. IO-Bound Workloads

Tests dominated by disk/network throughput rather than CPU:

| Benchmark | Description | Key Metric |
|-----------|-------------|------------|
| `io_parquet_scan` | Polars lazy scan with aggregations | Disk read + Rust engine |
| `io_parquet_roundtrip` | PyArrow read → Zstd compress → write → read | Compression speed |
| `io_file_hashing` | SHA-256 hashing with chunked reads | IO throughput |

**Expected**: All images perform similarly—IO dominates.

### 2. Pandas/Polars ETL

Typical data engineering operations:

| Benchmark | Description | Key Metric |
|-----------|-------------|------------|
| `etl_pandas_groupby_join` | Merge trades/quotes + groupby aggregation | C-optimized joins |
| `etl_polars_lazy_agg` | Same join via Polars lazy API | Query optimization |
| `etl_pandas_resample` | 5-minute OHLC bars + rolling windows | Time-series ops |
| `etl_python_feature_loop` | Feature engineering with Python loops | Interpreter speed |
| `etl_polars_python_udf` | Python UDFs via `map_elements()` | Python callback overhead |

**Expected**: 5–10% variation. Polars stable; pandas may show slight MKL benefits.

### 3. NumPy BLAS-Heavy

Dense linear algebra—**where image choice matters most**:

| Benchmark | Description | Key Metric |
|-----------|-------------|------------|
| `blas_matrix_multiply` | 2500×2500 DGEMM | Peak FLOPS |
| `blas_eigenvalue` | Symmetric eigenvalue decomposition | LAPACK efficiency |
| `blas_svd` | 2000×800 SVD | Memory bandwidth |
| `blas_cholesky` | Positive-definite Cholesky | Cache utilization |

**Expected**: MKL 1.1×–2.0× faster on Intel CPUs. OpenBLAS may win on AMD.

### 4. Pure Python CPU-Bound

Stress tests for the Python interpreter:

| Benchmark | Description | Key Metric |
|-----------|-------------|------------|
| `python_cpu_bound` | sin/cos/exp loops with threading | GIL contention |
| `python_threaded_sum` | Parallel summation | Threading scalability |
| `python_monte_carlo_var` | VaR simulation with ProcessPool | Multi-process scaling |
| `python_orderbook_replay` | Order book event processing | Bytecode execution |

**Expected**: Python 3.14 10–20% faster than 3.12 on interpreter-heavy tasks.

### 5. Real-World Mixed Workloads

End-to-end pipelines simulating production jobs:

| Benchmark | Description | Key Metric |
|-----------|-------------|------------|
| `realworld_etl_pipeline` | 4-stage pipeline with threading | Full-stack performance |
| `realworld_risk_calc` | Monte Carlo risk with parallel scenarios | Production realism |

**Expected**: Similar across images—workloads combine IO, vectorized ops, and multi-processing.

---

## 📊 Interpreting Results

### The Report Structure

The generated `results/report.html` includes:

1. **Executive Summary** — Per-category recommendations with rationale
2. **Detailed Results Tables** — Task-by-task timing comparisons
3. **Threading Comparison** — 1-thread vs 8-thread speedup analysis
4. **Image Comparison Charts** — Visual performance breakdowns

### Understanding Recommendations

The recommendation engine uses a **10% threshold rule**:

```
IF performance_spread ≤ 1.10×:
    → Recommend smallest image (python:3.14-slim)
    → Rationale: "Performance similar; optimize for image size"
ELSE:
    → Recommend fastest image for category
    → Rationale: "Measurable performance benefit"
```

### Key Metrics to Watch

| Metric | What It Tells You |
|--------|-------------------|
| **Duration (s)** | Raw execution time |
| **Memory Delta (MB)** | Peak memory above baseline |
| **Speedup (8T vs 1T)** | Threading effectiveness |
| **Spread (max/min)** | How much images differ |

---

## 🔬 Technical Deep Dive

### Why BLAS Library Matters

**BLAS (Basic Linear Algebra Subprograms)** is the engine behind NumPy's matrix operations. When you call `numpy.dot()` or `@`, NumPy delegates to BLAS—not Python.

| Library | Bundled With | Optimized For | Notes |
|---------|--------------|---------------|-------|
| **OpenBLAS** | `python:slim` | All CPUs | Vendor-neutral, open-source |
| **Intel MKL** | `intel/python`, `anaconda` | Intel CPUs | Proprietary, AVX-512 optimized |

**The AMD caveat**: MKL includes CPU vendor checks that historically throttled non-Intel processors. While improved, MKL on AMD may still underperform OpenBLAS.

### Python Version Differences

| Feature | Python 3.12 (Intel/Anaconda) | Python 3.14 (slim) |
|---------|------------------------------|---------------------|
| Interpreter speed | Baseline | 10–20% faster |
| Comprehension inlining | ✅ | ✅ |
| Adaptive specialization | Basic | Improved |
| Inline caching | Basic | Enhanced |
| Free-threaded support | ❌ | Experimental |

### The GIL Reality

Python's Global Interpreter Lock means:
- **CPU-bound threads serialize** — No parallelism benefit
- **IO-bound threads work** — GIL released during waits
- **C extensions vary** — NumPy/Polars release GIL internally
- **Multiprocessing bypasses** — Separate interpreters, no GIL sharing

**Free-threaded Python** (3.13+) removes this limitation, but NumPy doesn't yet support it. This benchmark uses standard GIL-enabled builds.

---

## 📁 Project Structure

```
finance-python-bench/
├── benchmarks/
│   ├── __init__.py
│   ├── data_gen.py      # Synthetic data generation
│   ├── tasks.py         # Individual benchmark implementations
│   ├── threading_task.py # Threading-specific benchmarks
│   ├── run_suite.py     # Benchmark orchestrator
│   └── report.py        # HTML report generator
├── scripts/
│   └── run_all.sh       # Docker-based test runner
├── results/
│   ├── *.json           # Raw benchmark results
│   ├── report.html      # Generated HTML report
│   └── logs/            # Execution logs
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Extending the Suite

### Add a Custom Benchmark

```python
# benchmarks/tasks.py

def my_custom_benchmark(data_dir: Path, threads: int) -> dict:
    """
    Your benchmark description.
    
    Returns dict with 'duration_s' and optional 'memory_delta_mb'.
    """
    start = time.perf_counter()
    
    # Your benchmark code here
    result = expensive_computation()
    
    duration = time.perf_counter() - start
    return {
        "duration_s": duration,
        "memory_delta_mb": get_memory_delta(),
        "custom_metric": result
    }

# Register in TASK_REGISTRY
TASK_REGISTRY["my_custom_benchmark"] = my_custom_benchmark
```

### Add a Custom Docker Image

```bash
# scripts/run_all.sh

run_my_image() {
  local THREADS=$1
  docker run --rm \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    -e THREADS="$THREADS" \
    my-custom-image:latest \
    bash -lc "pip install -r requirements.txt && \
              python benchmarks/run_suite.py \
                --label my-image-${THREADS}t \
                --output results/my-image-${THREADS}t.json \
                --scale $SCALE \
                --threads $THREADS"
}
```

---

## 📋 Requirements

### Runtime
- Docker 20.10+
- 8GB+ RAM recommended
- 10GB+ disk space for images

### For Local Development
- Python 3.12+
- Dependencies in `requirements.txt`:
  - pandas, polars, numpy, pyarrow
  - jinja2 (for report generation)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Additional benchmark tasks (ML inference, time-series, etc.)
- [ ] More Docker images (Alpine, Debian, custom builds)
- [ ] ARM64 / Apple Silicon benchmarks
- [ ] Free-threaded Python support (once NumPy is ready)
- [ ] Cloud-specific benchmarks (cold start, network IO)

### Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests locally
python benchmarks/run_suite.py --label local --output results/local.json --scale 0.5
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Intel Distribution for Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
- [Anaconda](https://www.anaconda.com/)
- [Polars](https://pola.rs/) — blazingly fast DataFrames
- [PyArrow](https://arrow.apache.org/docs/python/) — columnar memory format

---

<div align="center">

**[⬆ Back to Top](#-finance-python-benchmark-suite)**

Made with 🐍 for the quant finance community

</div>
