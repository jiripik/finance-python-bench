from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import jinja2
import pandas as pd


# Workload category definitions - recommendations computed dynamically from results
WORKLOAD_CATEGORIES = {
    "io": {
        "title": "📁 IO-Bound Workloads",
        "subtitle": "Parquet scan, file hashing, compression",
        "color": "#6b7280",
        "icon": "📁",
    },
    "etl": {
        "title": "🔄 Pandas/Polars ETL",
        "subtitle": "Joins, groupby, resample, aggregations",
        "color": "#3b82f6",
        "icon": "🔄",
    },
    "blas": {
        "title": "⚡ NumPy BLAS-Heavy",
        "subtitle": "Matrix ops, SVD, eigenvalue, Cholesky",
        "color": "#dc2626",
        "icon": "⚡",
    },
    "python": {
        "title": "🧵 Pure Python CPU-Bound",
        "subtitle": "Python loops, Monte Carlo, order book replay",
        "color": "#7c3aed",
        "icon": "🧵",
    },
    "realworld": {
        "title": "🏢 Real-World Mixed Workloads",
        "subtitle": "Multi-stage ETL pipelines, risk calculations with threading",
        "color": "#059669",
        "icon": "🏢",
    },
}

# Approximate Docker image sizes in MB (smaller is better as tiebreaker)
IMAGE_SIZES = {
    "anaconda": 3500,  # ~3.5GB
    "intel/python": 2800,  # ~2.8GB
    "python:3.14-gil": 150,  # ~150MB, GIL enabled
    "python:3.14-freethreaded": 150,  # ~150MB, free-threaded (no GIL)
    "python:3.14-slim": 150,  # ~150MB (legacy label)
}


TEMPLATE = jinja2.Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Docker Image Selection Guide - Finance Benchmarks</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; background: #f6f7fb; color: #1f2933; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 32px; border-bottom: 2px solid #4f46e5; padding-bottom: 8px; }
    h3 { margin-top: 24px; color: #374151; }
    .meta { color: #4b5563; margin-bottom: 24px; }
    
    /* Executive Summary */
    .exec-summary { background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; padding: 24px; border-radius: 12px; margin-bottom: 32px; }
    .exec-summary h2 { color: white; border: none; margin: 0 0 16px 0; }
    .exec-summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; }
    .exec-card { background: rgba(255,255,255,0.15); border-radius: 8px; padding: 16px; }
    .exec-card h4 { margin: 0 0 8px 0; font-size: 16px; }
    .exec-card .rec { font-size: 18px; font-weight: bold; color: #fef08a; }
    .exec-card .why { font-size: 12px; opacity: 0.9; margin-top: 4px; }
    
    /* Category sections */
    .category-section { margin: 32px 0; padding: 24px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .category-header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
    .category-icon { font-size: 32px; }
    .category-title { flex: 1; }
    .category-title h3 { margin: 0; font-size: 20px; }
    .category-title .subtitle { font-size: 14px; color: #6b7280; }
    .category-rec { background: #dcfce7; border: 2px solid #16a34a; border-radius: 8px; padding: 12px 16px; }
    .category-rec.warning { background: #fef3c7; border-color: #d97706; }
    .category-rec strong { color: #15803d; }
    .category-rec.warning strong { color: #92400e; }
    
    /* Results tables */
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; }
    th { background: #f3f4f6; font-weight: 600; }
    tr:nth-child(even) { background: #f9fafb; }
    .ok { color: #15803d; font-weight: 600; }
    .error { color: #b91c1c; font-weight: 600; }
    .best { background: #d1fae5; }
    .winner-cell { background: #bbf7d0; font-weight: bold; }
    
    /* Speedup indicators */
    .speedup { font-weight: bold; padding: 2px 8px; border-radius: 4px; }
    .speedup-good { background: #d1fae5; color: #15803d; }
    .speedup-neutral { background: #f3f4f6; color: #6b7280; }
    .speedup-bad { background: #fee2e2; color: #b91c1c; }
    
    /* Free-threading badge */
    .ft-badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .ft-yes { background: #7c3aed; color: white; }
    .ft-no { background: #e5e7eb; color: #6b7280; }
    
    /* BLAS backend badge */
    .blas-badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .blas-mkl { background: #0369a1; color: white; }
    .blas-openblas { background: #6b7280; color: white; }
    
    /* Comparison matrix */
    .comparison-matrix { margin: 24px 0; }
    .comparison-matrix th { text-align: center; min-width: 100px; }
    .comparison-matrix td { text-align: center; }
    .comparison-matrix .task-name { text-align: left; font-weight: 500; }
    
    /* Threading summary */
    .threading-highlight { background: linear-gradient(135deg, #7c3aed, #a855f7); color: white; padding: 20px; border-radius: 12px; margin: 24px 0; }
    .threading-highlight h3 { margin: 0 0 12px 0; color: white; }
    .threading-stats { display: flex; gap: 32px; flex-wrap: wrap; }
    .threading-stat { text-align: center; }
    .threading-stat .value { font-size: 28px; font-weight: bold; }
    .threading-stat .label { font-size: 12px; opacity: 0.9; }
    
    /* Warnings */
    .warning-box { background: #fef3c7; border-left: 4px solid #d97706; padding: 12px 16px; margin: 16px 0; border-radius: 0 8px 8px 0; }
    .warning-box strong { color: #92400e; }
    
    /* Section headers for thread groupings */
    .section-header { color: white; padding: 20px 24px; border-radius: 12px; margin: 32px 0 16px 0; }
    .section-header h2 { color: white; margin: 0; border: none; padding: 0; }
    .section-header .subtitle { font-size: 14px; opacity: 0.9; margin-top: 4px; }
    
    /* Version comparison */
    .version-cell { font-family: monospace; font-size: 13px; }
    
    /* Thread comparison section */
    .thread-comparison { background: linear-gradient(135deg, #0891b2, #06b6d4); color: white; padding: 24px; border-radius: 12px; margin: 32px 0; }
    .thread-comparison h2 { color: white; border: none; margin: 0 0 8px 0; padding: 0; }
    .thread-comparison .subtitle { opacity: 0.9; margin-bottom: 16px; }
    .thread-comparison table { background: rgba(255,255,255,0.1); border-radius: 8px; }
    .thread-comparison th, .thread-comparison td { border-color: rgba(255,255,255,0.2); color: white; }
    .thread-comparison th { background: rgba(255,255,255,0.15); }
    .thread-comparison .prefer-1t { background: rgba(59, 130, 246, 0.3); }
    .thread-comparison .prefer-8t { background: rgba(16, 185, 129, 0.3); }
    .thread-comparison .similar { background: rgba(255,255,255,0.05); }
    
    code { background: #eef2ff; padding: 2px 4px; border-radius: 4px; }
    @media (max-width: 768px) { .exec-summary-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>🐳 Docker Image Selection Guide</h1>
  <div class="meta">Finance Python Benchmarks • Generated from {{ file_count }} runs • Which base image should YOU use?</div>

    <!-- Executive Summary aligned with Goal.txt policy -->
    <div class="exec-summary">
        <h2>Executive Summary</h2>
        <div class="exec-summary-grid">
            {% for cat_id, cat in categories.items() %}
            <div class="exec-card">
                <h4>{{ cat.title|replace(cat.icon, '')|trim }}</h4>
                <div class="rec">Recommend: {{ cat.recommendation }}</div>
                <div class="why">{{ cat.reason }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

  <!-- Results by Thread Count -->
  {% for tc in thread_counts %}
  <div class="section-header" style="background: linear-gradient(135deg, {% if tc == 1 %}#3b82f6, #1d4ed8{% else %}#10b981, #047857{% endif %});">
    <h2>{% if tc == 1 %}🔹 Single-Threaded Results (1 Thread){% else %}🔸 Multi-Threaded Results ({{ tc }} Threads){% endif %}</h2>
    <div class="subtitle">{% if tc == 1 %}Baseline performance{% else %}Parallel workload performance{% endif %}</div>
  </div>

  {% for cat_id, cat in categories.items() %}
  {% if results_by_threads[tc].get(cat_id) %}
  <div class="category-section">
    <div class="category-header">
      <div class="category-icon">{{ cat.icon }}</div>
      <div class="category-title">
        <h3>{{ cat.title|replace(cat.icon, '')|trim }}</h3>
        <div class="subtitle">{{ cat.subtitle }}</div>
      </div>
    </div>
    
    {% if cat_id == 'blas' %}
    <div class="warning-box">
      <strong>⚠️ CPU Architecture Warning:</strong> Intel MKL is optimized for Intel CPUs. 
      On AMD Ryzen/EPYC, MKL may be <strong>slower</strong> than OpenBLAS due to CPU vendor checks.
      Always benchmark on YOUR hardware before committing to intel/python.
    </div>
    {% endif %}

    <!-- Comparison Matrix for this category -->
    <table class="comparison-matrix">
      <tr>
        <th>Task</th>
        {% for img in images %}
        <th>{{ img }}</th>
        {% endfor %}
        <th>Winner</th>
      </tr>
      {% for task_name, task_data in results_by_threads[tc][cat_id].items() %}
      <tr>
        <td class="task-name">{{ task_name }}</td>
        {% for img in images %}
          {% set result = task_data.get(img) %}
          {% if result %}
            {% set is_fastest = (img == task_data.winner) %}
            <td class="{% if is_fastest %}winner-cell{% endif %}">
              {{ '%.2f' % result.duration_s }}s
              {% if result.blas_backend %}
                <br><span class="blas-badge {% if 'MKL' in result.blas_backend %}blas-mkl{% else %}blas-openblas{% endif %}">{{ result.blas_backend }}</span>
              {% endif %}
              {% if result.speedup is defined and result.speedup > 0 %}
                <br><span class="speedup {% if result.speedup > 1.5 %}speedup-good{% elif result.speedup < 0.9 %}speedup-bad{% else %}speedup-neutral{% endif %}">{{ '%.1f' % result.speedup }}× speedup</span>
              {% endif %}
              {% if result.free_threaded is defined %}
                <br><span class="ft-badge {% if result.free_threaded %}ft-yes{% else %}ft-no{% endif %}">{% if result.free_threaded %}FT{% else %}GIL{% endif %}</span>
              {% endif %}
            </td>
          {% else %}
            <td>-</td>
          {% endif %}
        {% endfor %}
        <td><strong>{{ task_data.winner }}</strong></td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}
  {% endfor %}
  {% endfor %}

  <!-- Single-Threaded vs Multi-Threaded Comparison -->
  {% if thread_comparison %}
  <div class="thread-comparison">
    <h2>📊 When to Use Single-Threaded vs Multi-Threaded</h2>
    <div class="subtitle">Comparing 1-thread vs 8-thread performance to guide your threading decisions</div>
    
    <table>
      <tr>
        <th>Task</th>
        <th>Category</th>
        <th>Best Image (1t)</th>
        <th>Time (1t)</th>
        <th>Best Image (8t)</th>
        <th>Time (8t)</th>
        <th>Speedup</th>
        <th>Recommendation</th>
      </tr>
      {% for task_name, data in thread_comparison.items() %}
      <tr class="{% if data.speedup < 0.9 %}prefer-1t{% elif data.speedup > 1.2 %}prefer-8t{% else %}similar{% endif %}">
        <td>{{ task_name }}</td>
        <td>{{ data.category }}</td>
        <td>{{ data.best_1t }}</td>
        <td>{{ '%.2f' % data.time_1t }}s</td>
        <td>{{ data.best_8t }}</td>
        <td>{{ '%.2f' % data.time_8t }}s</td>
        <td><strong>{{ '%.2f' % data.speedup }}×</strong></td>
        <td>{% if data.speedup < 0.9 %}🔹 Use 1 thread{% elif data.speedup > 1.2 %}🔸 Use 8 threads{% else %}➖ Either works{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
    
    <div style="margin-top: 16px; font-size: 14px;">
      <strong>Key Insights:</strong>
      <ul style="margin: 8px 0; padding-left: 20px;">
        <li><strong>🔹 Blue rows:</strong> Single-threaded is faster (threading overhead dominates)</li>
        <li><strong>🔸 Green rows:</strong> Multi-threaded is faster (≥1.2× speedup from parallelism)</li>
        <li><strong>Gray rows:</strong> Similar performance - choose based on other factors</li>
      </ul>
    </div>
  </div>
  {% endif %}

  <!-- Raw Results -->
  <details style="margin-top: 32px;">
    <summary style="cursor: pointer; font-weight: 600; padding: 12px; background: #f3f4f6; border-radius: 8px;">
      📋 Raw Results (click to expand)
    </summary>
    <table style="margin-top: 16px;">
      <tr><th>Image</th><th>Task</th><th>Category</th><th>Duration (s)</th><th>RSS (MB)</th><th>Status</th><th>Details</th></tr>
      {% for row in all_rows %}
        <tr>
          <td>{{ row.label }}</td>
          <td>{{ row.name }}</td>
          <td>{{ row.category }}</td>
          <td>{{ '%.3f' % row.duration_s }}</td>
          <td>{{ '%.1f' % row.rss_delta_mb }}</td>
          <td class="{{ row.status }}">{{ row.status }}</td>
          <td style="font-size: 11px;">{{ row.extra or '' }}</td>
        </tr>
      {% endfor %}
    </table>
  </details>
</body>
</html>
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HTML report from benchmark JSON files"
    )
    parser.add_argument(
        "--results", nargs="+", required=True, help="List of JSON result files"
    )
    parser.add_argument(
        "--output", default="results/report.html", help="Path to HTML report"
    )
    return parser.parse_args()


def normalize_image_label(label: str) -> str:
    """Normalize Docker image label for comparison (strip thread suffix)."""
    label_lower = label.lower()
    # Remove thread suffix like -1t, -8t
    import re

    label_lower = re.sub(r"-\d+t$", "", label_lower)
    label = re.sub(r"-\d+t$", "", label, flags=re.IGNORECASE)

    if "anaconda" in label_lower:
        return "anaconda"
    elif "intel" in label_lower:
        return "intel/python"
    elif "3.14-freethreaded" in label_lower or "3.14t" in label_lower:
        return "python:3.14-freethreaded"
    elif "3.14-gil" in label_lower:
        return "python:3.14-gil"
    elif "3.14" in label:
        # Legacy: default to GIL variant for old labels
        return "python:3.14-gil"
    elif "python-slim" in label_lower:
        return "python:slim"
    return label


def extract_thread_count(label: str) -> int:
    """Extract thread count from label like 'anaconda-8t' -> 8."""
    import re

    match = re.search(r"-(\d+)t$", label.lower())
    return int(match.group(1)) if match else 1


def load_rows(paths: List[Path]) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load benchmark results and extract threading summary."""
    records: List[Dict[str, Any]] = []
    threading_results: List[Dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        label = (
            payload["metadata"]["label"]
            if "metadata" in payload
            else payload.get("label", path.stem)
        )
        normalized_label = normalize_image_label(label)
        thread_count = extract_thread_count(label)

        for row in payload.get("results", []):
            result_data = row.get("result") or {}
            extra_parts = []

            # Extract BLAS backend info
            blas_backend = result_data.get("blas_backend")
            if blas_backend:
                extra_parts.append(f"BLAS: {blas_backend}")

            # Extract free-threaded status
            free_threaded = result_data.get("free_threaded", False)
            if "free_threaded" in result_data:
                extra_parts.append(f"FT: {free_threaded}")

            # Extract speedup for threading benchmarks
            speedup = result_data.get("speedup")
            if speedup is not None:
                extra_parts.append(f"speedup: {speedup:.2f}x")
                # Add to threading summary for the highlight section
                if row.get("name") == "python_cpu_bound_threaded":
                    threading_results.append(
                        {
                            "label": normalized_label,
                            "speedup": speedup,
                            "threads": result_data.get("threads", 1),
                            "thread_count": thread_count,
                            "free_threaded": free_threaded,
                        }
                    )

            # Extract GFLOPS for matrix ops
            gflops = result_data.get("gflops")
            if gflops:
                extra_parts.append(f"GFLOPS: {gflops:.1f}")

            records.append(
                {
                    "label": label,
                    "normalized_label": normalized_label,
                    "thread_count": thread_count,
                    "name": row.get("name"),
                    "category": row.get("category"),
                    "duration_s": row.get("duration_s", 0),
                    "rss_delta_mb": row.get("rss_delta_mb", 0),
                    "status": row.get("status"),
                    "notes": row.get("notes"),
                    "extra": " | ".join(extra_parts) if extra_parts else "",
                    "blas_backend": blas_backend,
                    "free_threaded": free_threaded,
                    "speedup": speedup,
                }
            )

    return pd.DataFrame.from_records(records), threading_results


def build_category_results(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build results organized by category and task.

    Returns: {category: {task_name: {image_label: result_dict}}}
    """
    category_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for category in df["category"].unique():
        if pd.isna(category):
            continue
        cat_df = df[df["category"] == category]
        category_results[category] = {}

        for task_name in cat_df["name"].unique():
            task_df = cat_df[cat_df["name"] == task_name]
            task_results = {}

            for _, row in task_df.iterrows():
                img = row["normalized_label"]
                if (
                    img not in task_results
                    or row["duration_s"] < task_results[img]["duration_s"]
                ):
                    task_results[img] = {
                        "duration_s": row["duration_s"],
                        "blas_backend": row.get("blas_backend"),
                        "free_threaded": row.get("free_threaded"),
                        "speedup": row.get("speedup"),
                        "status": row["status"],
                    }

            # Determine winner
            if task_results:
                numeric_results = {
                    k: v
                    for k, v in task_results.items()
                    if isinstance(v["duration_s"], (int, float))
                }
                if numeric_results:
                    winner = min(
                        numeric_results, key=lambda k: numeric_results[k]["duration_s"]
                    )
                    task_results["winner"] = winner
                else:
                    task_results["winner"] = "-"

            category_results[category][task_name] = task_results

    return category_results


def build_version_comparison(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build overall version comparison table."""
    comparison: Dict[str, Dict[str, Any]] = {}

    for task_name in df["name"].unique():
        task_df = df[df["name"] == task_name]
        category = task_df["category"].iloc[0] if not task_df.empty else ""

        times = {}
        for _, row in task_df.iterrows():
            img = row["normalized_label"]
            if img not in times or row["duration_s"] < times[img]:
                times[img] = row["duration_s"]

        numeric_times = {k: v for k, v in times.items() if isinstance(v, (int, float))}
        if numeric_times:
            fastest = min(numeric_times, key=numeric_times.get)
        else:
            fastest = "-"

        comparison[task_name] = {
            "category": category,
            "times": times,
            "fastest": fastest,
        }

    return comparison


def compute_category_recommendations(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Compute recommendations for each category based on actual benchmark results.

    For each category:
    1. Count how many tasks each image wins
    2. If there's a clear winner (wins majority), recommend it
    3. If results are close, recommend smallest image
    4. Generate reason based on actual performance data
    """
    recommendations: Dict[str, Dict[str, str]] = {}

    for category in df["category"].unique():
        if pd.isna(category):
            continue

        cat_df = df[df["category"] == category]

        # Count wins per image
        wins: Dict[str, int] = {}
        total_times: Dict[str, List[float]] = {}

        for task_name in cat_df["name"].unique():
            task_df = cat_df[cat_df["name"] == task_name]

            times = {}
            for _, row in task_df.iterrows():
                img = row["normalized_label"]
                if img not in times or row["duration_s"] < times[img]:
                    times[img] = row["duration_s"]
                if img not in total_times:
                    total_times[img] = []
                total_times[img].append(row["duration_s"])

            if times:
                numeric_times = {
                    k: v
                    for k, v in times.items()
                    if isinstance(v, (int, float)) and v > 0
                }
                if numeric_times:
                    winner = min(numeric_times, key=numeric_times.get)
                    wins[winner] = wins.get(winner, 0) + 1

        if not wins:
            recommendations[category] = {
                "recommendation": "any image",
                "reason": "No benchmark data available.",
            }
            continue

        # Analyze results
        total_tasks = sum(wins.values())
        sorted_winners = sorted(
            wins.items(), key=lambda x: (-x[1], IMAGE_SIZES.get(x[0], 9999))
        )

        top_image, top_wins = sorted_winners[0]
        win_pct = top_wins / total_tasks if total_tasks > 0 else 0

        # Goal.txt policy: if results are close, prefer smallest image (python:slim lane);
        # BLAS-heavy may prefer Intel image on Intel CPUs; Python CPU-bound may prefer 3.14 for FT.
        avg_times = {img: (sum(t) / len(t)) for img, t in total_times.items() if t}
        ratio = None
        if avg_times:
            best_avg = min(avg_times.values())
            worst_avg = max(avg_times.values())
            ratio = (worst_avg / best_avg) if best_avg > 0 else None

        # Define closeness threshold
        close_threshold = 1.10

        # Category-specific guidance
        if category in ("io", "etl", "realworld"):
            # Prefer smallest image when results are close
            if ratio is not None and ratio <= close_threshold:
                # Prefer python slim variant among present images
                candidates = list(avg_times.keys())
                smallest = min(candidates, key=lambda x: IMAGE_SIZES.get(x, 9999))
                recommendations[category] = {
                    "recommendation": smallest,
                    "reason": f"Performance similar across images (≤{close_threshold:.2f}× spread). {smallest} recommended as smallest, ~{IMAGE_SIZES.get(smallest, '?')}MB.",
                }
            else:
                # Clear winner fallback
                second_info = ""
                if len(sorted_winners) > 1:
                    second_img, second_wins = sorted_winners[1]
                    second_info = f" ({second_img}: {second_wins}/{total_tasks})"
                recommendations[category] = {
                    "recommendation": top_image,
                    "reason": f"Won {top_wins}/{total_tasks} tasks in this category{second_info}.",
                }
        elif category == "blas":
            # If Intel/python is among winners or shows MKL backend frequently, recommend it with caveat
            # Otherwise fall back to smallest or fastest depending on spread
            intel_present = any(img.startswith("intel/") for img in avg_times.keys())
            if intel_present and (
                win_pct >= 0.5 or (ratio is not None and ratio > close_threshold)
            ):
                recommendations[category] = {
                    "recommendation": "intel/python",
                    "reason": "Dense linear algebra benefits observed. Recommend intel/python on Intel CPUs; benchmark on your hardware (MKL vs OpenBLAS).",
                }
            else:
                # Results close: prefer smallest
                if ratio is not None and ratio <= close_threshold:
                    smallest = min(
                        avg_times.keys(), key=lambda x: IMAGE_SIZES.get(x, 9999)
                    )
                    recommendations[category] = {
                        "recommendation": smallest,
                        "reason": f"BLAS results similar across images (≤{close_threshold:.2f}× spread). {smallest} is smallest image.",
                    }
                else:
                    # Clear fastest
                    recommendations[category] = {
                        "recommendation": top_image,
                        "reason": f"Won {top_wins}/{total_tasks} tasks in BLAS category.",
                    }
        elif category == "python":
            # Prefer Python 3.14 slim only if free-threaded speedup observed; else prefer smallest
            cat_df = df[df["category"] == category]
            ft_rows = cat_df[
                (cat_df["name"] == "python_cpu_bound") & (cat_df["thread_count"] == 8)
            ]
            ft_advantage = False
            if not ft_rows.empty:
                # Check for any image reporting free_threaded True and speedup > 1.2
                for _, r in ft_rows.iterrows():
                    if bool(r.get("free_threaded")) and (
                        float(r.get("speedup") or 0.0) > 1.2
                    ):
                        ft_advantage = True
                        break

            if ft_advantage and any(
                img.startswith("python:3.14-slim") for img in avg_times.keys()
            ):
                recommendations[category] = {
                    "recommendation": "python:3.14-slim",
                    "reason": "Observed >1.2× speedup with free-threaded Python; prefer 3.14 for CPU-bound threaded Python where compatible.",
                }
            else:
                smallest = min(avg_times.keys(), key=lambda x: IMAGE_SIZES.get(x, 9999))
                recommendations[category] = {
                    "recommendation": smallest,
                    "reason": f"No free-threaded advantage observed; performance similar. Recommend smallest image ({smallest}).",
                }
        else:
            # Generic fallback
            second_info = ""
            if len(sorted_winners) > 1:
                second_img, second_wins = sorted_winners[1]
                second_info = f" ({second_img}: {second_wins}/{total_tasks})"
            recommendations[category] = {
                "recommendation": top_image,
                "reason": f"Won {top_wins}/{total_tasks} tasks in this category{second_info}.",
            }

    return recommendations


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.results]
    df, threading_summary = load_rows(paths)

    if df.empty:
        raise SystemExit("No results found")

    # Get unique images for column headers
    images = sorted(df["normalized_label"].unique())

    # Get unique thread counts
    thread_counts = sorted(df["thread_count"].unique())

    # Build category-organized results for each thread count
    results_by_threads: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for tc in thread_counts:
        tc_df = df[df["thread_count"] == tc]
        results_by_threads[tc] = build_category_results(tc_df)

    # Build version comparison for each thread count
    comparison_by_threads: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for tc in thread_counts:
        tc_df = df[df["thread_count"] == tc]
        comparison_by_threads[tc] = build_version_comparison(tc_df)

    # Split threading summary by thread count
    threading_summary_1t = [
        t for t in threading_summary if t.get("thread_count", 1) == 1
    ]
    threading_summary_8t = [
        t for t in threading_summary if t.get("thread_count", 1) == 8
    ]

    # Build 1t vs 8t comparison for threading guidance
    thread_comparison: Dict[str, Dict[str, Any]] = {}
    if 1 in thread_counts and 8 in thread_counts:
        for task_name in df["name"].unique():
            # Get best time for 1 thread
            df_1t = df[(df["name"] == task_name) & (df["thread_count"] == 1)]
            df_8t = df[(df["name"] == task_name) & (df["thread_count"] == 8)]

            if df_1t.empty or df_8t.empty:
                continue

            category = df_1t["category"].iloc[0]

            # Find best performer for each thread count
            best_1t_idx = df_1t["duration_s"].idxmin()
            best_8t_idx = df_8t["duration_s"].idxmin()

            time_1t = df_1t.loc[best_1t_idx, "duration_s"]
            time_8t = df_8t.loc[best_8t_idx, "duration_s"]
            best_img_1t = df_1t.loc[best_1t_idx, "normalized_label"]
            best_img_8t = df_8t.loc[best_8t_idx, "normalized_label"]

            # Speedup: how much faster is 8t vs 1t (>1 means 8t is faster)
            speedup = time_1t / time_8t if time_8t > 0 else 1.0

            thread_comparison[task_name] = {
                "category": category,
                "time_1t": time_1t,
                "time_8t": time_8t,
                "best_1t": best_img_1t,
                "best_8t": best_img_8t,
                "speedup": speedup,
            }

    # Compute dynamic recommendations based on results
    # Use 8-thread results for recommendations (more realistic production scenario)
    # Fall back to 1-thread if 8-thread not available
    if 8 in thread_counts:
        rec_df = df[df["thread_count"] == 8]
    elif thread_counts:
        rec_df = df[df["thread_count"] == thread_counts[0]]
    else:
        rec_df = df

    category_recommendations = compute_category_recommendations(rec_df)

    # Merge recommendations into categories dict
    categories_with_recs = {}
    for cat_id, cat_info in WORKLOAD_CATEGORIES.items():
        cat_with_rec = dict(cat_info)
        if cat_id in category_recommendations:
            cat_with_rec["recommendation"] = category_recommendations[cat_id][
                "recommendation"
            ]
            cat_with_rec["reason"] = category_recommendations[cat_id]["reason"]
        else:
            cat_with_rec["recommendation"] = "see results"
            cat_with_rec["reason"] = "No benchmark data for this category."
        categories_with_recs[cat_id] = cat_with_rec

    html = TEMPLATE.render(
        file_count=len(paths),
        all_rows=df.to_dict(orient="records"),
        categories=categories_with_recs,
        thread_counts=thread_counts,
        results_by_threads=results_by_threads,
        comparison_by_threads=comparison_by_threads,
        thread_comparison=thread_comparison if thread_comparison else None,
        images=images,
        threading_summary_1t=threading_summary_1t if threading_summary_1t else None,
        threading_summary_8t=threading_summary_8t if threading_summary_8t else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
