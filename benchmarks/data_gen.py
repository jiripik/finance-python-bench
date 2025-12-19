"""Synthetic data generators for benchmark scenarios.

All datasets are synthetic and deterministic for repeatability. Sizes scale
linearly with the ``scale`` argument so users can tune runtime to hit the
5+ minute target for their hardware.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def _ensure_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _build_trade_frame(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    symbols = np.array(
        [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "JPM",
            "V",
            "XOM",
        ]
    )
    df = pd.DataFrame(
        {
            "symbol": rng.choice(symbols, size=rows),
            "price": rng.lognormal(mean=0.02, sigma=0.15, size=rows) * 100,
            "size": rng.integers(1, 5_000, size=rows),
            "side": rng.choice(["B", "S"], size=rows),
            "ts": rng.integers(1_700_000_000, 1_700_172_800, size=rows),
        }
    )
    return df


def _build_quote_frame(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    symbols = np.array(
        [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "JPM",
            "V",
            "XOM",
        ]
    )
    mid = rng.lognormal(mean=0.01, sigma=0.1, size=rows) * 100
    spread = rng.lognormal(mean=-3.0, sigma=0.3, size=rows)
    df = pd.DataFrame(
        {
            "symbol": rng.choice(symbols, size=rows),
            "bid": mid - spread,
            "ask": mid + spread,
            "size": rng.integers(1, 10_000, size=rows),
            "ts": rng.integers(1_700_000_000, 1_700_172_800, size=rows),
        }
    )
    return df


def generate_datasets(
    base_dir: Path, scale: float = 1.0, seed: int = 42
) -> Dict[str, Tuple[Path, float]]:
    """Generate synthetic parquet datasets.

    Parameters
    ----------
    base_dir: Path
        Directory to hold generated files.
    scale: float
        Linear scale factor. 1.0 generates ~5-8 million rows per table; increase
        to drive longer runtimes.
    seed: int
        Random seed for repeatability.

    Returns
    -------
    Dict[str, Tuple[Path, float]]
        Mapping of dataset names to (path, seconds_to_build).
    """

    base_dir = _ensure_dir(base_dir)
    rng = np.random.default_rng(seed)

    trade_rows = max(1_500_000, int(5_000_000 * scale))
    quote_rows = max(2_000_000, int(8_000_000 * scale))

    trades_path = base_dir / "trades.parquet"
    quotes_path = base_dir / "quotes.parquet"
    info: Dict[str, Tuple[Path, float]] = {}

    if not trades_path.exists():
        start = time.perf_counter()
        trades = _build_trade_frame(rng, trade_rows)
        trades.to_parquet(trades_path, compression="snappy")
        info["trades"] = (trades_path, time.perf_counter() - start)
    else:
        info["trades"] = (trades_path, 0.0)

    if not quotes_path.exists():
        start = time.perf_counter()
        quotes = _build_quote_frame(rng, quote_rows)
        table = pa.Table.from_pandas(quotes, preserve_index=False)
        pq.write_table(table, quotes_path, compression="snappy")
        info["quotes"] = (quotes_path, time.perf_counter() - start)
    else:
        info["quotes"] = (quotes_path, 0.0)

    return info
