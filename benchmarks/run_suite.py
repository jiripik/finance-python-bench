from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import psutil
from tqdm import tqdm

# Allow running as a script without installing the package
if __package__ is None:  # pragma: no cover - CLI convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.data_gen import generate_datasets
from benchmarks.tasks import build_tasks


def _median(values: List[float]) -> float:
    """Calculate median of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def _measure_task(name: str, fn, iterations: int = 3) -> Dict[str, Any]:
    """Run a task multiple times and return median metrics."""
    proc = psutil.Process()
    durations: List[float] = []
    rss_deltas: List[float] = []
    status = "ok"
    notes = None
    result: Dict[str, Any] | None = None

    for i in range(iterations):
        rss_before = proc.memory_info().rss
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # pragma: no cover - benchmark harness
            status = "error"
            notes = f"{type(exc).__name__}: {exc}"
            break
        duration = time.perf_counter() - start
        rss_after = proc.memory_info().rss
        durations.append(duration)
        rss_deltas.append((rss_after - rss_before) / (1024 * 1024))

    median_duration = _median(durations)
    median_rss_delta = _median(rss_deltas)

    return {
        "name": name,
        "status": status,
        "duration_s": median_duration,
        "rss_delta_mb": median_rss_delta,
        "result": result,
        "notes": notes,
        "iterations": len(durations),
        "all_durations": durations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finance benchmarks in the current environment"
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label for the current environment/image (e.g., python-slim)",
    )
    parser.add_argument(
        "--output", default="results/run.json", help="Path to JSON output"
    )
    parser.add_argument(
        "--data-dir", default=".bench_data", help="Directory to hold generated data"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Workload scale factor"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Thread cap for BLAS/polars/joblib (0=library default)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3, takes average)",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*DataFrame.swapaxes.*",
        category=FutureWarning,
    )

    args = parse_args()
    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    data_info = generate_datasets(base_dir, scale=args.scale)
    trades_path = data_info["trades"][0]
    quotes_path = data_info["quotes"][0]

    tasks = build_tasks(
        trades_path, quotes_path, scale=args.scale, threads=args.threads
    )

    results: List[Dict[str, Any]] = []
    for task in tqdm(tasks, desc=f"benchmarks ({args.iterations}x avg)"):
        measured = _measure_task(task["name"], task["fn"], iterations=args.iterations)
        measured["category"] = task.get("category", "unknown")
        results.append(measured)

    metadata = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scale": args.scale,
        "threads": args.threads,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "env": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "POLARS_MAX_THREADS",
            ]
            if os.environ.get(key) is not None
        },
        "data_dir": str(base_dir),
    }

    payload = {"metadata": metadata, "results": results}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
