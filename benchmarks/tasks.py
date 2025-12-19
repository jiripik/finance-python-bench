"""
Benchmark tasks for hedge-fund style batch jobs.

Organized by workload category to demonstrate Docker image selection:

1. IO-BOUND: Parquet scan, file hashing - all images similar
2. PANDAS/POLARS ETL: Joins, groupby, resample - all images similar
3. NUMPY BLAS-HEAVY: Matrix ops, linalg - Intel Python wins on Intel CPUs
4. PURE PYTHON CPU-BOUND: Python loops - Free-threaded Python wins with threading
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from benchmarks.threading_task import run_threaded_sum


def _maybe_limit_threads(threads: int) -> None:
    """Configure threading for BLAS/Polars libraries."""
    if threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        os.environ["POLARS_MAX_THREADS"] = str(threads)
    else:
        os.environ.pop("POLARS_MAX_THREADS", None)


def _check_free_threaded() -> bool:
    """Check if running on free-threaded (no-GIL) Python."""
    try:
        return not sys.flags.gil
    except AttributeError:
        return False


def _get_numpy_backend() -> str:
    """Detect NumPy BLAS backend (MKL, OpenBLAS, etc)."""
    try:
        config = np.__config__
        if hasattr(config, "show"):
            import io
            import contextlib

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                config.show()
            info = f.getvalue().lower()
            if "mkl" in info:
                return "MKL"
            elif "openblas" in info:
                return "OpenBLAS"
    except Exception:
        pass
    return "Unknown"


# =============================================================================
# CATEGORY 1: IO-BOUND WORKLOADS
# Expected: All Docker images perform similarly (~1.00x)
# Recommendation: python:slim (smallest, fastest pulls)
# =============================================================================


def io_parquet_scan(trades_path: Path, quotes_path: Path) -> Dict[str, Any]:
    """
    IO-bound: Scan Parquet files using Polars lazy evaluation.

    This is dominated by disk/network IO and the Rust engine.
    Docker image choice has minimal impact.
    """
    start = time.perf_counter()

    trades = pl.scan_parquet(trades_path)
    quotes = pl.scan_parquet(quotes_path)

    # Simple aggregation - mostly IO
    trades_count = trades.select(pl.len()).collect().item()
    quotes_count = quotes.select(pl.len()).collect().item()

    duration = time.perf_counter() - start

    return {
        "workload_hint": "io-insensitive",
        "message": "Image choice has minimal impact; prefer python:slim for production.",
        "trades_rows": trades_count,
        "quotes_rows": quotes_count,
        "scan_time_s": duration,
    }


def io_parquet_roundtrip(trades_path: Path) -> Dict[str, Any]:
    """
    IO-bound: Read and write Parquet with compression.

    Performance dominated by disk IO and compression algorithm.
    """
    start = time.perf_counter()

    table = pq.read_table(trades_path)
    out_path = trades_path.parent / "trades_roundtrip.parquet"
    pq.write_table(table, out_path, compression="zstd")
    out_rows = pq.read_table(out_path).num_rows
    out_path.unlink(missing_ok=True)

    duration = time.perf_counter() - start

    return {
        "workload_hint": "io-insensitive",
        "message": "IO and compression dominate; images perform similarly.",
        "rows": out_rows,
        "roundtrip_time_s": duration,
    }


def _hash_file(path: Path) -> str:
    """Hash a file - IO-bound operation."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def io_file_hashing(
    trades_path: Path, quotes_path: Path, threads: int
) -> Dict[str, Any]:
    """
    IO-bound: Hash multiple files.

    Performance dominated by disk read speed.
    Uses sequential processing when threads=1, otherwise uses ThreadPoolExecutor.
    """
    paths = [trades_path, quotes_path]

    start = time.perf_counter()

    if threads == 1:
        # Sequential processing for single-threaded mode
        hashes = [_hash_file(p) for p in paths]
    else:
        # Use threads for parallel IO - GIL isn't a bottleneck for IO
        num_workers = min(4, os.cpu_count() or 2)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            hashes = list(pool.map(_hash_file, paths))

    duration = time.perf_counter() - start

    return {
        "workload_hint": "io-insensitive",
        "message": "Disk read throughput bound; threading helps IO regardless of GIL.",
        "files_hashed": len(hashes),
        "hash_time_s": duration,
    }


# =============================================================================
# CATEGORY 2: PANDAS/POLARS ETL WORKLOADS
# Expected: All Docker images similar (~0.95x-1.05x)
# Recommendation: python:slim for production, anaconda for research
# =============================================================================


def etl_pandas_groupby_join(trades_path: Path, quotes_path: Path) -> Dict[str, Any]:
    """
    ETL: Pandas merge and groupby aggregation.

    Heavy data manipulation with joins. Performance similar across images
    because pandas operations are mostly C-optimized.
    """
    start = time.perf_counter()

    trades = pd.read_parquet(trades_path)
    quotes = pd.read_parquet(quotes_path)
    merged = trades.merge(quotes, on=["symbol", "ts"], how="inner")
    result = merged.groupby("symbol").agg(
        {"price": "mean", "size_x": "sum", "size_y": "sum"}
    )

    duration = time.perf_counter() - start

    return {
        "workload_hint": "etl-insensitive",
        "message": "Vectorized pandas dominates; images are within ~5-10% typically.",
        "rows_trades": len(trades),
        "rows_quotes": len(quotes),
        "rows_merged": len(merged),
        "rows_result": len(result),
        "etl_time_s": duration,
    }


def etl_polars_lazy_agg(trades_path: Path, quotes_path: Path) -> Dict[str, Any]:
    """
    ETL: Polars lazy join and aggregation.

    Uses Polars' Rust engine - very fast, minimal Python overhead.
    Docker image has minimal impact.
    """
    start = time.perf_counter()

    trades = pl.scan_parquet(trades_path)
    quotes = pl.scan_parquet(quotes_path)

    lazy_join = trades.join(quotes, on=["symbol", "ts"], how="inner", suffix="_q")
    summary = (
        lazy_join.group_by("symbol")
        .agg(
            [
                pl.col("price").mean().alias("avg_price"),
                pl.col("size").sum().alias("trade_volume"),
                pl.col("size_q").sum().alias("quote_depth"),
                (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread"),
            ]
        )
        .sort("trade_volume", descending=True)
    )
    df = summary.collect()

    duration = time.perf_counter() - start

    return {
        "workload_hint": "etl-insensitive",
        "message": "Polars Rust engine minimizes Python overhead; images similar.",
        "rows_result": df.shape[0],
        "top_symbol": df[0, "symbol"] if df.shape[0] else None,
        "etl_time_s": duration,
    }


def etl_pandas_resample(trades_path: Path) -> Dict[str, Any]:
    """
    ETL: Pandas time-series resampling and rolling windows.

    Common hedge fund operation for OHLC bars.
    """
    start = time.perf_counter()

    trades = pd.read_parquet(trades_path)
    trades["dt"] = pd.to_datetime(trades["ts"], unit="s")
    trades.set_index("dt", inplace=True)

    resampled = (
        trades.groupby("symbol").resample("5min").agg({"price": "mean", "size": "sum"})
    )
    rolling = resampled.groupby(level=0).rolling(window=12).mean()

    duration = time.perf_counter() - start

    return {
        "workload_hint": "etl-insensitive",
        "message": "Common OHLC resample; vectorized ops keep images close.",
        "rows": len(rolling),
        "resample_time_s": duration,
    }


def etl_python_feature_loop(trades_path: Path, scale: float) -> Dict[str, Any]:
    """
    ETL: Feature engineering with Python loops over DataFrame rows.

    This benchmark shows Python 3.14's interpreter speed advantage.
    Python 3.14 has ~10-15% faster bytecode execution which benefits
    workloads that iterate in Python rather than calling vectorized ops.

    Realistic scenario: Custom feature engineering that can't be vectorized,
    such as stateful computations, complex business logic, or event-driven features.
    """
    start = time.perf_counter()

    trades = pd.read_parquet(trades_path, columns=["symbol", "price", "size", "ts"])
    trades = trades.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # Limit rows for reasonable runtime but scale with parameter
    n_rows = min(len(trades), max(200_000, int(500_000 * scale)))
    trades = trades.head(n_rows)

    # Python-loop-heavy feature engineering (not vectorizable easily)
    # Simulates complex stateful business logic
    features = []
    prev_price: Dict[str, float] = {}
    prev_size: Dict[str, int] = {}
    cum_volume: Dict[str, int] = {}
    cum_notional: Dict[str, float] = {}
    trade_count: Dict[str, int] = {}

    for row in trades.itertuples(index=False):
        sym = row.symbol
        price = float(row.price)
        size = int(row.size)

        # Update cumulative state
        cum_volume[sym] = cum_volume.get(sym, 0) + size
        cum_notional[sym] = cum_notional.get(sym, 0.0) + price * size
        trade_count[sym] = trade_count.get(sym, 0) + 1

        # Compute features using Python logic
        vwap = cum_notional[sym] / cum_volume[sym] if cum_volume[sym] > 0 else price
        price_vs_vwap = (price - vwap) / vwap if vwap > 0 else 0.0

        # Momentum features (require previous values)
        prev_p = prev_price.get(sym, price)
        prev_s = prev_size.get(sym, size)
        price_change = (price - prev_p) / prev_p if prev_p > 0 else 0.0
        size_change = (size - prev_s) / prev_s if prev_s > 0 else 0.0

        # Complex business logic - categorize trade urgency
        if abs(price_change) > 0.01 and size > 2 * prev_s:
            urgency = "high"
        elif abs(price_change) > 0.005 or size > prev_s:
            urgency = "medium"
        else:
            urgency = "low"

        features.append(
            {
                "symbol": sym,
                "vwap": vwap,
                "price_vs_vwap": price_vs_vwap,
                "price_change": price_change,
                "size_change": size_change,
                "urgency": urgency,
                "cum_trades": trade_count[sym],
            }
        )

        prev_price[sym] = price
        prev_size[sym] = size

    result_df = pd.DataFrame(features)
    duration = time.perf_counter() - start

    return {
        "workload_hint": "python-cpu-interpret",
        "message": "Interpreter-driven Python loops; 3.14 may be modestly faster single-thread.",
        "rows_processed": n_rows,
        "symbols": len(prev_price),
        "features_computed": len(result_df.columns) * len(result_df),
        "high_urgency_pct": (result_df["urgency"] == "high").mean() * 100,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "loop_time_s": duration,
    }


def etl_polars_python_udf(
    trades_path: Path, quotes_path: Path, scale: float
) -> Dict[str, Any]:
    """
    ETL: Polars with Python UDFs (User Defined Functions).

    Polars is fast for built-in operations, but sometimes you need custom
    Python logic via map_elements(). This benchmark shows Python 3.14's
    faster bytecode execution when calling back into Python from Rust.

    This is a common pattern when:
    - Complex business logic can't be expressed in Polars expressions
    - Integrating with Python-only libraries
    - Prototyping before vectorizing
    """
    start = time.perf_counter()

    trades = pl.read_parquet(trades_path)
    quotes = pl.read_parquet(quotes_path)

    # Limit rows for reasonable runtime
    n_rows = min(len(trades), max(300_000, int(800_000 * scale)))
    trades = trades.head(n_rows)
    quotes = quotes.head(n_rows)

    # Join trades and quotes
    merged = trades.join(quotes, on=["symbol", "ts"], how="inner", suffix="_q")

    # Python UDF for complex spread analysis
    # This runs in Python, showing interpreter speed differences
    def analyze_spread(bid: float, ask: float, price: float, size: int) -> str:
        """Complex spread analysis that requires Python logic."""
        spread = ask - bid
        mid = (bid + ask) / 2
        spread_bps = (spread / mid) * 10000 if mid > 0 else 0

        # Position relative to spread
        if price <= bid:
            position = "at_bid"
        elif price >= ask:
            position = "at_ask"
        elif price < mid:
            position = "below_mid"
        else:
            position = "above_mid"

        # Spread regime classification
        if spread_bps < 5:
            regime = "tight"
        elif spread_bps < 20:
            regime = "normal"
        else:
            regime = "wide"

        # Size-weighted urgency
        if size > 1000 and regime == "tight":
            return f"{position}_large_tight"
        elif size > 500:
            return f"{position}_large_{regime}"
        else:
            return f"{position}_{regime}"

    # Apply Python UDF using map_elements
    result = merged.with_columns(
        pl.struct(["bid", "ask", "price", "size"])
        .map_elements(
            lambda x: analyze_spread(x["bid"], x["ask"], x["price"], x["size"]),
            return_dtype=pl.Utf8,
        )
        .alias("spread_analysis")
    )

    # Aggregate by analysis category
    summary = (
        result.group_by("spread_analysis")
        .agg(
            pl.col("size").sum().alias("total_volume"),
            pl.col("price").mean().alias("avg_price"),
            pl.len().alias("trade_count"),
        )
        .sort("total_volume", descending=True)
    )

    duration = time.perf_counter() - start

    return {
        "workload_hint": "python-callback",
        "message": "Python UDF callbacks show interpreter cost; images remain close.",
        "rows_processed": len(merged),
        "categories": len(summary),
        "top_category": summary[0, "spread_analysis"] if len(summary) > 0 else None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "udf_time_s": duration,
    }


# =============================================================================
# CATEGORY 3: NUMPY BLAS-HEAVY WORKLOADS
# Expected: intel/python wins on Intel CPUs (1.1x-2.0x for right kernels)
# Warning: Can be SLOWER on AMD CPUs!
# Recommendation: intel/python ONLY for Intel CPUs with proven benchmarks
# =============================================================================


def blas_matrix_multiply(scale: float) -> Dict[str, Any]:
    """
    BLAS-heavy: Large matrix multiplication (DGEMM).

    This is WHERE intel/python shines on Intel CPUs due to MKL optimization.
    Can be 1.1x-2.0x faster than OpenBLAS on Intel hardware.
    WARNING: May be slower on AMD CPUs!
    """
    size = max(1500, int(2500 * math.sqrt(scale)))

    start = time.perf_counter()

    rng = np.random.default_rng(123)
    a = rng.standard_normal((size, size), dtype=np.float64)
    b = rng.standard_normal((size, size), dtype=np.float64)
    _ = a @ b  # DGEMM - heavily optimized by MKL

    duration = time.perf_counter() - start

    return {
        "workload_hint": "blas-sensitive",
        "message": "Dense DGEMM favors MKL on Intel; consider intel/python on Intel nodes.",
        "matrix_size": size,
        "elements": size * size,
        "gflops": (2 * size**3) / duration / 1e9,
        "blas_backend": _get_numpy_backend(),
        "matmul_time_s": duration,
    }


def blas_eigenvalue(scale: float) -> Dict[str, Any]:
    """
    BLAS-heavy: Eigenvalue decomposition (DSYEV).

    Linear algebra kernel - benefits from MKL on Intel CPUs.
    """
    size = max(800, int(1200 * math.sqrt(scale)))

    start = time.perf_counter()

    rng = np.random.default_rng(456)
    a = rng.standard_normal((size, size), dtype=np.float64)
    # Make symmetric positive definite
    a = a @ a.T + np.eye(size) * size

    eigenvalues, _ = np.linalg.eigh(a)

    duration = time.perf_counter() - start

    return {
        "workload_hint": "blas-sensitive",
        "message": "Eigen decomposition benefits from MKL; benchmark on your hardware.",
        "matrix_size": size,
        "largest_eigenvalue": float(eigenvalues[-1]),
        "blas_backend": _get_numpy_backend(),
        "eigen_time_s": duration,
    }


def blas_svd(scale: float) -> Dict[str, Any]:
    """
    BLAS-heavy: Singular Value Decomposition (SVD).

    Common in finance for PCA, factor analysis.
    Benefits significantly from MKL on Intel CPUs.
    """
    rows = max(1000, int(2000 * math.sqrt(scale)))
    cols = max(500, int(800 * math.sqrt(scale)))

    start = time.perf_counter()

    rng = np.random.default_rng(789)
    a = rng.standard_normal((rows, cols), dtype=np.float64)

    _, s, _ = np.linalg.svd(a, full_matrices=False)
    # Explained variance ratio
    total_var = (s**2).sum()
    top10_var = (s[:10] ** 2).sum() / total_var if len(s) >= 10 else 1.0

    duration = time.perf_counter() - start

    return {
        "workload_hint": "blas-sensitive",
        "message": "SVD/PCA often faster with MKL on Intel; AMD may differ.",
        "matrix_shape": f"{rows}x{cols}",
        "singular_values": len(s),
        "top10_explained_var": float(top10_var),
        "blas_backend": _get_numpy_backend(),
        "svd_time_s": duration,
    }


def blas_cholesky(scale: float) -> Dict[str, Any]:
    """
    BLAS-heavy: Cholesky decomposition.

    Used in risk models, portfolio optimization.
    """
    size = max(1000, int(1800 * math.sqrt(scale)))

    start = time.perf_counter()

    rng = np.random.default_rng(321)
    a = rng.standard_normal((size, size), dtype=np.float64)
    # Make positive definite
    a = a @ a.T + np.eye(size) * size

    _ = np.linalg.cholesky(a)

    duration = time.perf_counter() - start

    return {
        "workload_hint": "blas-sensitive",
        "message": "Risk-model Cholesky is MKL-optimized on Intel; verify on AMD.",
        "matrix_size": size,
        "blas_backend": _get_numpy_backend(),
        "cholesky_time_s": duration,
    }


def python_threaded_sum(scale: float, threads: int) -> Dict[str, Any]:
    """
    Pure Python CPU-bound workload using threading.

    This benchmark is designed to highlight the performance gains from the
    free-threading model in Python 3.14. It calculates a sum of a series
    in parallel using multiple threads.

    - With GIL: Performance will not scale with the number of threads.
    - Free-threaded: Performance should scale well with the number of threads.
    """
    work_size = int(10_000_000 * scale)
    num_threads = threads if threads > 0 else min(8, os.cpu_count() or 2)

    if num_threads == 1:
        # Baseline for single-threaded execution
        result = run_threaded_sum(1, work_size)
        speedup = 1.0
        efficiency = 1.0
        verdict = "Single-threaded baseline"
    else:
        # Multi-threaded execution
        result = run_threaded_sum(num_threads, work_size)
        # Run single-threaded for comparison to calculate speedup
        single_thread_result = run_threaded_sum(1, work_size)

        speedup = (
            single_thread_result["duration_s"] / result["duration_s"]
            if result["duration_s"] > 0
            else 0.0
        )
        theoretical_max = float(num_threads)
        efficiency = speedup / theoretical_max if theoretical_max > 0 else 0.0
        verdict = "True parallelism!" if speedup > 1.5 else "GIL-limited"

    return {
        "workload_hint": "python-threading-sensitive",
        "message": "Demonstrates free-threading scaling for CPU-bound Python.",
        "threads": num_threads,
        "work_size": work_size,
        "duration_s": result["duration_s"],
        "speedup": speedup,
        "efficiency": efficiency,
        "free_threaded": _check_free_threaded(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "verdict": verdict,
    }


# =============================================================================
# CATEGORY 4: PURE PYTHON CPU-BOUND WORKLOADS
# Expected: Free-threaded Python 3.14 wins with threading (1.5x-3.0x speedup)
# GIL-limited Python: ~1.0x speedup (no benefit from threads)
# Recommendation: python:3.14-slim free-threaded for CPU-bound Python loops
# =============================================================================


def _pure_python_compute(data: List[float], iterations: int) -> float:
    """CPU-bound pure-Python work - benefits from no-GIL."""
    result = 0.0
    for _ in range(iterations):
        for val in data:
            result += math.sin(val) * math.cos(val) * math.exp(-abs(val) / 100)
    return result


def python_cpu_bound(scale: float, threads: int) -> Dict[str, Any]:
    """
    Pure Python CPU-bound workload.

    When threads=1: runs sequentially (baseline)
    When threads>1: runs with threading
      - With GIL: ~1.0x speedup (threads serialize)
      - Free-threaded: 1.5x-3.0x speedup (true parallelism)
    """
    n_threads = threads if threads > 0 else min(8, os.cpu_count() or 2)
    data_size = max(1500, int(4000 * scale))
    iterations = max(80, int(200 * scale))

    rng = np.random.default_rng(42)
    data = [float(x) for x in rng.standard_normal(data_size)]

    free_threaded = _check_free_threaded()

    if n_threads == 1:
        # Single-threaded: just run the work once
        start = time.perf_counter()
        _pure_python_compute(data, iterations)
        duration = time.perf_counter() - start

        return {
            "threads": n_threads,
            "data_size": data_size,
            "iterations": iterations,
            "sequential_time_s": duration,
            "threaded_time_s": duration,
            "speedup": 1.0,
            "theoretical_max_speedup": 1.0,
            "efficiency": 1.0,
            "free_threaded": free_threaded,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "workload_hint": "python-threading-sensitive",
            "message": "Threads help only with free-threaded Python; baseline single-thread shown.",
            "verdict": "Single-threaded baseline",
        }

    # Sequential baseline (same work as threaded will do)
    start = time.perf_counter()
    for _ in range(n_threads):
        _pure_python_compute(data, iterations)
    sequential_time = time.perf_counter() - start

    # Threaded execution
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [
            pool.submit(_pure_python_compute, data, iterations)
            for _ in range(n_threads)
        ]
        _ = [f.result() for f in futures]
    threaded_time = time.perf_counter() - start

    speedup = sequential_time / threaded_time if threaded_time > 0 else 0.0
    theoretical_max = float(n_threads)
    efficiency = speedup / theoretical_max if theoretical_max > 0 else 0.0

    return {
        "threads": n_threads,
        "data_size": data_size,
        "iterations": iterations,
        "sequential_time_s": sequential_time,
        "threaded_time_s": threaded_time,
        "speedup": speedup,
        "theoretical_max_speedup": theoretical_max,
        "efficiency": efficiency,
        "free_threaded": free_threaded,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "workload_hint": "python-threading-sensitive",
        "message": "GIL-limited shows ~1.0x; free-threaded shows 1.5x–3.0x speedup.",
        # Interpretation for report
        "verdict": "True parallelism!" if speedup > 1.5 else "GIL-limited",
    }


def _mc_var_worker(
    args: tuple,
) -> Dict[str, float]:
    """Monte Carlo VaR worker - mixed Python/NumPy."""
    mu, sigma, horizon_days, paths, seed = args
    rng = np.random.default_rng(seed)
    shocks = rng.normal(
        loc=mu / 252,
        scale=sigma / math.sqrt(252),
        size=(paths, horizon_days, mu.shape[0]),
    )
    pnl = shocks.sum(axis=(1, 2))
    var5 = np.percentile(pnl, 5)
    cutoff = int(0.05 * len(pnl))
    worst = np.partition(pnl, cutoff)[:cutoff]
    es5 = float(worst.mean()) if len(worst) else float(var5)
    return {"var5": float(var5), "es5": es5}


def python_monte_carlo_var(
    trades_path: Path, scale: float, threads: int
) -> Dict[str, Any]:
    """
    Monte Carlo VaR: Mixed Python/NumPy workload.

    Realistic hedge fund risk calculation.
    Benefits from both BLAS optimization AND free-threading.
    Uses sequential processing when threads=1.
    """
    trades = pd.read_parquet(trades_path, columns=["symbol", "price", "ts"])
    trades.sort_values(["symbol", "ts"], inplace=True)
    trades["ret"] = trades.groupby("symbol")["price"].pct_change().fillna(0.0)
    stats = (
        trades.groupby("symbol")["ret"]
        .agg(["mean", "std"])
        .sort_values("std", ascending=False)
        .head(6)
    )

    mu = stats["mean"].to_numpy()
    sigma = stats["std"].clip(lower=1e-6).to_numpy()
    horizon_days = 10
    paths_per_worker = max(50_000, int(120_000 * scale))
    workers = threads if threads > 0 else min(8, os.cpu_count() or 2)

    start = time.perf_counter()

    seeds = [7 + i for i in range(workers)]
    args = [(mu, sigma, horizon_days, paths_per_worker, seed) for seed in seeds]

    if workers == 1:
        # Sequential processing for single-threaded mode
        results = [_mc_var_worker(args[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_mc_var_worker, args))

    duration = time.perf_counter() - start

    var5_mean = float(np.mean([r["var5"] for r in results]))
    es5_mean = float(np.mean([r["es5"] for r in results]))

    return {
        "symbols": stats.index.tolist(),
        "workers": workers,
        "workers_type": "process" if workers > 1 else "sequential",
        "paths_per_worker": paths_per_worker,
        "total_paths": workers * paths_per_worker,
        "var5": var5_mean,
        "es5": es5_mean,
        "blas_backend": _get_numpy_backend(),
        "free_threaded": _check_free_threaded(),
        "workload_hint": "mixed",
        "message": "Mixed Python/NumPy; benefits from processes/BLAS more than image choice.",
        "mc_time_s": duration,
    }


def python_orderbook_replay(trades_path: Path, scale: float) -> Dict[str, Any]:
    """
    Pure Python: Order book replay simulation.

    Heavy Python loop - shows Python interpreter overhead.
    Free-threaded doesn't help here (single-threaded), but shows
    that free-threaded has minimal overhead for sequential code.
    """
    trades = pd.read_parquet(
        trades_path, columns=["symbol", "price", "size", "side", "ts"]
    )
    rows = min(len(trades), max(600_000, int(1_500_000 * scale)))
    sample = trades.sample(n=rows, random_state=123).sort_values("ts")

    start = time.perf_counter()

    positions: Dict[str, int] = {}
    last_px: Dict[str, float] = {}
    cash = 0.0

    for row in sample.itertuples(index=False):
        size = int(row.size)
        price = float(row.price)
        sym = row.symbol
        if row.side == "B":
            positions[sym] = positions.get(sym, 0) + size
            cash -= size * price
        else:
            positions[sym] = positions.get(sym, 0) - size
            cash += size * price
        last_px[sym] = price

    duration = time.perf_counter() - start

    gross_notional = sum(
        abs(pos) * last_px.get(sym, 0.0) for sym, pos in positions.items()
    )

    return {
        "rows_processed": rows,
        "symbols": len(positions),
        "gross_notional": gross_notional,
        "free_threaded": _check_free_threaded(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "workload_hint": "python-cpu-interpret",
        "message": "Heavy Python loop; single-thread shows interpreter cost; FT has minimal overhead.",
        "replay_time_s": duration,
    }


# =============================================================================
# CATEGORY 5: REAL-WORLD MIXED WORKLOADS
# Expected: Python 3.14 ~40% faster than Intel Python for dataframe+threading
# Recommendation: python:3.14-slim for mixed workloads
# =============================================================================


def realworld_etl_pipeline(
    trades_path: Path, quotes_path: Path, scale: float, threads: int
) -> Dict[str, Any]:
    """
    Real-World: Multi-stage ETL pipeline with threading.

    Simulates a typical hedge fund data processing job:
    1. Load and merge large datasets
    2. Feature engineering with rolling windows
    3. Parallel aggregation across symbols
    4. Final consolidation

    This benchmark shows where Python 3.14 shines - the combination of
    improved interpreter performance and better threading makes it
    significantly faster than older Python versions for mixed workloads.
    """
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    start = time.perf_counter()

    # Stage 1: Load and merge data
    trades = pd.read_parquet(trades_path)
    quotes = pd.read_parquet(quotes_path)

    # Scale up the data for realistic workload
    repeat_factor = max(1, int(3 * scale))
    if repeat_factor > 1:
        trades = pd.concat([trades] * repeat_factor, ignore_index=True)
        quotes = pd.concat([quotes] * repeat_factor, ignore_index=True)

    merged = trades.merge(
        quotes, on=["symbol", "ts"], how="left", suffixes=("", "_quote")
    )

    # Stage 2: Feature engineering with rolling windows
    symbols = merged["symbol"].unique()

    def process_symbol(sym: str) -> pd.DataFrame:
        """Process a single symbol - compute features."""
        df = merged[merged["symbol"] == sym].copy()
        df = df.sort_values("ts")

        # Rolling statistics
        df["price_sma_20"] = df["price"].rolling(20, min_periods=1).mean()
        df["price_std_20"] = df["price"].rolling(20, min_periods=1).std().fillna(0)
        df["volume_sma_20"] = df["size"].rolling(20, min_periods=1).mean()

        # Derived features
        df["price_zscore"] = (df["price"] - df["price_sma_20"]) / (
            df["price_std_20"] + 1e-8
        )
        df["volume_ratio"] = df["size"] / (df["volume_sma_20"] + 1)

        # VWAP calculation
        df["cum_vol"] = df["size"].cumsum()
        df["cum_pv"] = (df["price"] * df["size"]).cumsum()
        df["vwap"] = df["cum_pv"] / df["cum_vol"]

        return df

    # Stage 3: Process symbols (parallel or sequential based on threads)
    num_workers = threads if threads > 0 else min(4, os.cpu_count() or 2)

    if num_workers == 1:
        # Sequential processing for single-threaded mode
        processed_dfs = [process_symbol(sym) for sym in symbols]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            processed_dfs = list(pool.map(process_symbol, symbols))

    # Stage 4: Consolidation and final aggregation
    result_df = pd.concat(processed_dfs, ignore_index=True)

    # Final summary stats
    summary = result_df.groupby("symbol").agg(
        {
            "price": ["mean", "std", "min", "max"],
            "size": "sum",
            "price_zscore": ["mean", "std"],
            "vwap": "last",
        }
    )

    duration = time.perf_counter() - start

    return {
        "input_rows": len(merged),
        "output_rows": len(result_df),
        "symbols_processed": len(symbols),
        "workers": num_workers,
        "pipeline_time_s": duration,
        "free_threaded": _check_free_threaded(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "workload_hint": "mixed",
        "message": "Mixed ETL with threading; prefer python:slim; 3.14 may help modestly.",
    }


def realworld_risk_calc(
    trades_path: Path, scale: float, threads: int
) -> Dict[str, Any]:
    """
    Real-World: Portfolio risk calculation with parallel processing.

    Simulates computing VaR, Greeks, and other risk metrics across
    a portfolio of positions. Uses a mix of pandas operations and
    numerical computation with threading.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    start = time.perf_counter()

    trades = pd.read_parquet(trades_path)

    # Build position aggregates
    positions = (
        trades.groupby("symbol").agg({"size": "sum", "price": "last"}).reset_index()
    )
    positions["notional"] = positions["size"] * positions["price"]

    # Scale up for realistic workload
    n_scenarios = max(1000, int(5000 * scale))
    n_workers = threads if threads > 0 else min(4, os.cpu_count() or 2)

    def compute_scenario_risk(scenario_id: int) -> Dict[str, float]:
        """Compute risk metrics for a single scenario."""
        np.random.seed(scenario_id)

        # Simulate price shocks
        shocks = np.random.normal(0, 0.02, len(positions))
        shocked_prices = positions["price"].values * (1 + shocks)
        shocked_notional = positions["size"].values * shocked_prices

        # P&L
        pnl = np.sum(shocked_notional - positions["notional"].values)

        # Additional risk metrics
        var_contrib = np.abs(positions["notional"].values * shocks)
        max_loss = np.min(positions["notional"].values * shocks)

        return {
            "scenario_id": scenario_id,
            "pnl": pnl,
            "max_loss": max_loss,
            "total_var_contrib": np.sum(var_contrib),
        }

    scenarios = list(range(n_scenarios))

    if n_workers == 1:
        # Sequential processing for single-threaded mode
        results = [compute_scenario_risk(s) for s in scenarios]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(compute_scenario_risk, scenarios))

    duration = time.perf_counter() - start

    pnls = [r["pnl"] for r in results]
    var_5 = np.percentile(pnls, 5)
    es_5 = np.mean([p for p in pnls if p <= var_5])

    return {
        "positions": len(positions),
        "scenarios": n_scenarios,
        "workers": n_workers,
        "var_5pct": float(var_5),
        "expected_shortfall_5pct": float(es_5),
        "calc_time_s": duration,
        "free_threaded": _check_free_threaded(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "workload_hint": "mixed",
        "message": "Portfolio risk with threading; image choice minor; processes dominate.",
    }


# =============================================================================
# TASK BUILDER
# =============================================================================

TaskFunc = Callable[[], Dict[str, Any]]


def build_tasks(
    trades_path: Path, quotes_path: Path, scale: float, threads: int
) -> List[Dict[str, Any]]:
    """
    Build benchmark tasks organized by workload category.

    Categories map to Docker image recommendations:
    - io: All images similar → python:slim (smallest)
    - etl: All images similar → python:slim (production) / anaconda (research)
    - blas: Intel wins on Intel CPUs → intel/python (with caveats)
    - python: Free-threaded wins → python:3.14-slim free-threaded
    """
    _maybe_limit_threads(threads)

    return [
        # Category 1: IO-Bound (all images ~same)
        {
            "name": "io_parquet_scan",
            "category": "io",
            "fn": lambda: io_parquet_scan(trades_path, quotes_path),
        },
        {
            "name": "io_parquet_roundtrip",
            "category": "io",
            "fn": lambda: io_parquet_roundtrip(trades_path),
        },
        {
            "name": "io_file_hashing",
            "category": "io",
            "fn": lambda: io_file_hashing(trades_path, quotes_path, threads),
        },
        # Category 2: Pandas/Polars ETL (all images ~same)
        {
            "name": "etl_pandas_groupby_join",
            "category": "etl",
            "fn": lambda: etl_pandas_groupby_join(trades_path, quotes_path),
        },
        {
            "name": "etl_polars_lazy_agg",
            "category": "etl",
            "fn": lambda: etl_polars_lazy_agg(trades_path, quotes_path),
        },
        {
            "name": "etl_pandas_resample",
            "category": "etl",
            "fn": lambda: etl_pandas_resample(trades_path),
        },
        {
            "name": "etl_python_feature_loop",
            "category": "etl",
            "fn": lambda: etl_python_feature_loop(trades_path, scale),
        },
        {
            "name": "etl_polars_python_udf",
            "category": "etl",
            "fn": lambda: etl_polars_python_udf(trades_path, quotes_path, scale),
        },
        # Category 3: NumPy BLAS-Heavy (intel/python wins on Intel CPUs)
        {
            "name": "blas_matrix_multiply",
            "category": "blas",
            "fn": lambda: blas_matrix_multiply(scale),
        },
        {
            "name": "blas_eigenvalue",
            "category": "blas",
            "fn": lambda: blas_eigenvalue(scale),
        },
        {
            "name": "blas_svd",
            "category": "blas",
            "fn": lambda: blas_svd(scale),
        },
        {
            "name": "blas_cholesky",
            "category": "blas",
            "fn": lambda: blas_cholesky(scale),
        },
        # Category 4: Pure Python CPU-Bound (free-threaded wins)
        {
            "name": "python_cpu_bound",
            "category": "python",
            "fn": lambda: python_cpu_bound(scale, threads),
        },
        {
            "name": "python_threaded_sum",
            "category": "python",
            "fn": lambda: python_threaded_sum(scale, threads),
        },
        {
            "name": "python_monte_carlo_var",
            "category": "python",
            "fn": lambda: python_monte_carlo_var(trades_path, scale, threads),
        },
        {
            "name": "python_orderbook_replay",
            "category": "python",
            "fn": lambda: python_orderbook_replay(trades_path, scale),
        },
        # Category 5: Real-World Mixed Workloads (Python 3.14 wins)
        {
            "name": "realworld_etl_pipeline",
            "category": "realworld",
            "fn": lambda: realworld_etl_pipeline(
                trades_path, quotes_path, scale, threads
            ),
        },
        {
            "name": "realworld_risk_calc",
            "category": "realworld",
            "fn": lambda: realworld_risk_calc(trades_path, scale, threads),
        },
    ]
