"""Data loader module for fetching and managing trading data.

Orchestrates downloading via *yfinance*, hybrid calendar-day alignment,
feature engineering (delegated to ``features.py``), and persistence to CSV.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.features import compute_all_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------
FINTECH: list[str] = ["SQ", "PYPL", "COIN", "HOOD", "V", "MA"]
CRYPTO: list[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]
MACRO: list[str] = ["^TNX", "DX-Y.NYB"]

STOCK_TICKERS: list[str] = FINTECH + MACRO          # market-hours assets
CRYPTO_TICKERS: list[str] = CRYPTO                   # 24/7 assets
ALL_TICKERS: list[str] = STOCK_TICKERS + CRYPTO_TICKERS

START_DATE: str = "2020-01-01"
OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE: str = "chimera_data_v1.csv"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV data for *ticker* from Yahoo Finance.

    Returns ``None`` when no data is available (e.g. delisted symbol).
    """
    try:
        logger.info("Downloading %s …", ticker)
        df: pd.DataFrame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning("Empty data for %s – skipping.", ticker)
            return None

        # yfinance may return MultiIndex columns for single tickers – flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.index = pd.to_datetime(df.index)
        return df

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Hybrid alignment
# ---------------------------------------------------------------------------
def _align_to_calendar(
    df: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    is_crypto: bool,
) -> pd.DataFrame:
    """Reindex *df* to a full calendar-day index.

    * **Stocks / Macro**: forward-fill prices; fill Volume with 0.
    * **Crypto**: retain original 24/7 data; only reindex (no fill needed
      beyond what the full join provides).
    """
    df = df.reindex(full_index)

    if is_crypto:
        # Crypto trades every day; gaps are genuine missing data – still ffill
        # for any rare holes (exchange downtime, etc.)
        price_cols = [c for c in df.columns if c != "Volume"]
        df[price_cols] = df[price_cols].ffill()
        df["Volume"] = df["Volume"].fillna(0)
    else:
        # Stocks: carry Friday close into Sat/Sun
        price_cols = [c for c in df.columns if c != "Volume"]
        df[price_cols] = df[price_cols].ffill()
        df["Volume"] = df["Volume"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def build_dataset(
    start: str = START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """Download, align, feature-engineer, and merge the full universe.

    Returns
    -------
    pd.DataFrame
        Columns are MultiIndex ``(feature, ticker)`` after merge, then
        flattened to ``ticker_feature`` for easy CSV storage.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # --- 1. Download raw data -------------------------------------------
    raw: dict[str, pd.DataFrame] = {}
    for ticker in ALL_TICKERS:
        df = _download_ticker(ticker, start, end)
        if df is not None:
            raw[ticker] = df

    if not raw:
        raise pd.errors.EmptyDataError("No data downloaded for any ticker.")

    # --- 2. Build full calendar index -----------------------------------
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(df.index for df in raw.values())))
    )
    full_index = pd.date_range(all_dates.min(), all_dates.max(), freq="D")

    # --- 3. Align & compute features per ticker -------------------------
    featured: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        is_crypto = ticker in CRYPTO_TICKERS
        aligned = _align_to_calendar(df, full_index, is_crypto=is_crypto)
        enriched = compute_all_features(aligned)
        featured[ticker] = enriched

    # --- 4. Merge into a single wide DataFrame --------------------------
    merged = pd.concat(featured, axis=1)
    # Flatten MultiIndex columns → "TICKER_Feature"
    merged.columns = [f"{ticker}_{col}" for ticker, col in merged.columns]

    # --- 5. Drop rows with any remaining NaN (warm-up + target horizon) -
    rows_before = len(merged)
    merged.dropna(inplace=True)
    rows_after = len(merged)
    logger.info(
        "Dropped %d warm-up / target NaN rows (%d → %d).",
        rows_before - rows_after,
        rows_before,
        rows_after,
    )

    return merged


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save *df* to CSV inside *output_dir*; returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILE
    df.to_csv(out_path)
    logger.info("Dataset saved → %s  (%s rows × %s cols)", out_path, *df.shape)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """End-to-end: download → align → feature-engineer → save → validate."""
    logger.info("=" * 60)
    logger.info("Chimera Alpha Engine  –  Phase 1: Data Pipeline")
    logger.info("=" * 60)

    df = build_dataset()
    save_dataset(df)

    # ---- Validation ----------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Shape
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

    # Correlation proof: COIN (stock) vs BTC-USD (crypto) closes
    coin_close = f"COIN_Close"
    btc_close = f"BTC-USD_Close"
    if coin_close in df.columns and btc_close in df.columns:
        corr_cols = [coin_close, btc_close]
        corr_matrix = df[corr_cols].corr()
        print(f"\nCorrelation matrix  (COIN vs BTC-USD Close):\n{corr_matrix}")
    else:
        missing = [c for c in (coin_close, btc_close) if c not in df.columns]
        print(f"\n⚠  Cannot compute correlation – missing columns: {missing}")

    print()


if __name__ == "__main__":
    main()
