"""
🧬 Chimera Alpha Engine — Phase 3: Live Inference Dashboard

Streamlit application that:
  1. Downloads live OHLCV data via yfinance
  2. Engineers the EXACT features used during TFT training (Phase 1)
  3. Runs inference through the saved TFT model
  4. Renders a Plotly fan chart with quantile prediction bands

Launch:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torchmetrics
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# pytorch-forecasting / Lightning imports (must match training environment)
# ---------------------------------------------------------------------------
import lightning.pytorch as pl  # noqa: F401  — unified Lightning 2.x
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    GroupNormalizer,
)

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="🧬 Chimera Alpha Engine",
    page_icon="🧬",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_tft_model.ckpt"

TICKER_OPTIONS: list[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD",
    "PYPL", "COIN", "HOOD",
]

CRYPTO_TICKERS: set[str] = {"BTC-USD", "ETH-USD", "SOL-USD"}

# Features the model was trained on (must match Phase 2 / Cell 3 exactly)
TIME_VARYING_UNKNOWN_REALS: list[str] = [
    "Close",
    "Log_Returns",
    "Trend_Spread",
    "Stoch_K",
    "Stoch_D",
    "RVOL",
]

# Quantile indices produced by QuantileLoss (7 quantiles):
# 0=P2, 1=P10, 2=P25, 3=P50 (median), 4=P75, 5=P90, 6=P98
Q_P10, Q_P25, Q_P50, Q_P75, Q_P90 = 1, 2, 3, 4, 5


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Engineering  (mirrors src/features.py EXACTLY)
# ═══════════════════════════════════════════════════════════════════════════

def load_market_data(ticker: str, days: int = 300) -> pd.DataFrame | None:
    """Download OHLCV data and compute every feature used during training.

    The formulas here are copied verbatim from ``src/features.py`` to
    prevent any concept drift between training and inference.

    Returns ``None`` when yfinance fails or returns empty data.
    """
    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    # Flatten MultiIndex columns if present (yfinance quirk)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel("Ticker")

    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # ------------------------------------------------------------------
    # 1. Trend — EMA Spread  (src/features.py  → add_trend_features)
    #    Trend_Spread = (EMA_13 - EMA_21) / Close
    # ------------------------------------------------------------------
    ema_13 = df["Close"].ewm(span=13, adjust=False).mean()
    ema_21 = df["Close"].ewm(span=21, adjust=False).mean()
    df["Trend_Spread"] = (ema_13 - ema_21) / df["Close"]

    # ------------------------------------------------------------------
    # 2. Momentum — Full Stochastic Oscillator  (add_momentum_features)
    #    %K = SMA_3( (C - L14) / (H14 - L14) * 100 )
    #    %D = SMA_3(%K)
    # ------------------------------------------------------------------
    window, smooth_k, smooth_d = 14, 3, 3
    lowest_low = df["Low"].rolling(window=window, min_periods=1).min()
    highest_high = df["High"].rolling(window=window, min_periods=1).max()
    raw_k = (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
    df["Stoch_K"] = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    df["Stoch_D"] = df["Stoch_K"].rolling(window=smooth_d, min_periods=1).mean()

    # ------------------------------------------------------------------
    # 3. Volume — Relative Volume  (add_volume_features)
    #    RVOL = Volume / (SMA_20_Volume + 1e-9)
    # ------------------------------------------------------------------
    sma_vol = df["Volume"].rolling(window=20, min_periods=1).mean()
    df["RVOL"] = df["Volume"] / (sma_vol + 1e-9)

    # ------------------------------------------------------------------
    # 4. Volatility — Log Returns  (add_volatility_features)
    #    Log_Returns = ln(Close_t / Close_{t-1})
    # ------------------------------------------------------------------
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # ------------------------------------------------------------------
    # 5. Target — Forward 5-day return (placeholder = 0 for inference)
    #    During training: Target_Return_5D = Close_{t+5}/Close_t - 1
    #    At inference we don't know the future, so fill with 0.
    # ------------------------------------------------------------------
    df["Target_Return_5D"] = 0.0

    # ------------------------------------------------------------------
    # 6. Metadata columns required by TimeSeriesDataSet
    # ------------------------------------------------------------------
    df["Ticker"] = str(ticker)
    df["Asset_Class"] = "CRYPTO" if ticker in CRYPTO_TICKERS else "FINTECH"

    # Reset the DatetimeIndex into a plain "Date" column to avoid
    # ambiguity (yfinance names the index "Date" too).
    df = df.reset_index()
    if "Date" not in df.columns and "index" in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)

    # time_idx: continuous integer sequence (matches training setup)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["time_idx"] = range(len(df))

    # Cast categoricals to string
    df["Ticker"] = df["Ticker"].astype(str)
    df["Asset_Class"] = df["Asset_Class"].astype(str)

    # Fill residual NaNs (warm-up rows)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Loading  (cached & CPU-compatible)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = "models/best_tft_model.ckpt"

    # 1. Check if file exists first
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file NOT found at: {os.path.abspath(model_path)}")
        st.warning("Please verify you downloaded 'best_tft_model.ckpt' and placed it in the 'models/' folder.")
        return None

    try:
        # 2. Primary: let Lightning handle it with CPU mapping
        model = TemporalFusionTransformer.load_from_checkpoint(
            model_path,
            map_location=torch.device("cpu"),
        )
        model.eval()
        return model

    except Exception as primary_err:
        # 3. Fallback — "Nuclear Option"
        # Lightning's load_from_checkpoint can touch CUDA internals
        # (trainer state, device metadata) BEFORE map_location is applied.
        # On a CPU-only PyTorch build this raises a non-RuntimeError.
        # Work around it by deserialising the checkpoint ourselves and
        # manually reconstructing the model from hparams + state_dict.
        st.info(
            f"⚡ Primary load raised `{type(primary_err).__name__}`. "
            "Trying manual CPU reconstruction…"
        )
        try:
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )
            model = TemporalFusionTransformer(**checkpoint["hyper_parameters"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            return model
        except Exception as fallback_err:
            st.error(f"❌ Fallback also failed: {fallback_err}")
            st.error(f"(Original error was: {primary_err})")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Plotly Fan Chart  (The Cockpit)
# ═══════════════════════════════════════════════════════════════════════════

def build_fan_chart(
    hist_df: pd.DataFrame,
    future_dates: list[pd.Timestamp],
    quantiles: np.ndarray,
    ticker: str,
) -> go.Figure:
    """Build a Plotly figure with historical close + quantile prediction fan.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Last 60 days of historical data (must have ``Date`` and ``Close``).
    future_dates : list[pd.Timestamp]
        Five dates: Today+1 … Today+5.
    quantiles : np.ndarray
        Shape ``(5, 7)`` — 5 prediction steps × 7 quantile levels, in
        absolute price (not returns).
    ticker : str
        Ticker symbol for chart title.
    """
    fig = go.Figure()

    # --- Historical close ------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#636EFA", width=2),
        )
    )

    # --- Connector from last close to first P50 -------------------------
    last_date = hist_df["Date"].iloc[-1]
    last_close = float(hist_df["Close"].iloc[-1])
    fig.add_trace(
        go.Scatter(
            x=[last_date, future_dates[0]],
            y=[last_close, float(quantiles[0, Q_P50])],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # --- Outer fan: P10 → P90  (Light Blue — Risk/Reward Zone) ----------
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=(
                list(quantiles[:, Q_P90])
                + list(quantiles[:, Q_P10])[::-1]
            ),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.15)",
            line=dict(width=0),
            name="P10 – P90  (Risk/Reward)",
            hoverinfo="skip",
        )
    )

    # --- Inner fan: P25 → P75  (Dark Blue — Probable Zone) ---------------
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=(
                list(quantiles[:, Q_P75])
                + list(quantiles[:, Q_P25])[::-1]
            ),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.35)",
            line=dict(width=0),
            name="P25 – P75  (Probable Zone)",
            hoverinfo="skip",
        )
    )

    # --- Median line: P50  (Orange Dashed — Median Target) ---------------
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=list(quantiles[:, Q_P50]),
            mode="lines+markers",
            name="P50  (Median Target)",
            line=dict(color="orange", width=2.5, dash="dash"),
            marker=dict(size=5, color="orange"),
        )
    )

    fig.update_layout(
        title=f"🧬 {ticker} — 5-Day TFT Forecast (Quantile Fan)",
        xaxis_title="Date",
        yaxis_title="Price (USD)" if "USD" in ticker else "Price",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=540,
        margin=dict(l=60, r=30, t=80, b=50),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Inference Loop
# ═══════════════════════════════════════════════════════════════════════════

def predict_next_5_days(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[pd.Timestamp]] | None:
    """Run TFT inference and return quantile predictions + future dates.

    Uses ``TimeSeriesDataSet.from_parameters`` with the model's stored
    dataset parameters so the encoder configuration matches training
    exactly.

    Returns
    -------
    (quantiles, future_dates)
        quantiles : np.ndarray of shape ``(5, 7)``
        future_dates : list of 5 Timestamps (Today+1 … Today+5)
    """
    try:
        # Build an inference dataset using the model's saved parameters
        # so encoder length, normaliser, categoricals, etc. all match.
        inference_dataset = TimeSeriesDataSet.from_parameters(
            model.dataset_parameters,
            df,
            predict=True,
            stop_randomization=True,
        )

        dataloader = inference_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0,
        )

        # ── Force every sub-module to CPU ("Nuclear Option") ─────────
        # The checkpoint was saved on CUDA.  torchmetrics Metric._apply()
        # creates  torch.zeros(1, device=self.device)  with the OLD device
        # *before* the move-function runs → crashes on CPU-only PyTorch.
        #
        # Fix: temporarily monkey-patch Metric._apply so it sets
        # device=cpu FIRST, then delegates to the standard nn.Module._apply
        # (which doesn't create dummy tensors).  Restore afterwards.
        _original_metric_apply = torchmetrics.Metric._apply

        def _cpu_safe_apply(self, fn, *args, **kwargs):
            self._device = torch.device("cpu")
            return nn.Module._apply(self, fn, *args, **kwargs)

        torchmetrics.Metric._apply = _cpu_safe_apply
        try:
            model.to(torch.device("cpu"))

            # Use model.predict() (NOT trainer.predict) so that
            # pytorch_forecasting sets up its internal prediction callbacks.
            # trainer_kwargs forces the internally-created Trainer to CPU.
            output = model.predict(
                dataloader,
                mode="quantiles",
                return_x=True,
                trainer_kwargs=dict(accelerator="cpu", logger=False),
            )

            # model.predict with return_x may return a tuple or a single
            # tensor depending on the pytorch-forecasting version.
            if isinstance(output, (tuple, list)):
                raw_predictions = output[0]
            else:
                raw_predictions = output
        finally:
            torchmetrics.Metric._apply = _original_metric_apply

        # Extract quantiles — shape: (n_samples, prediction_length, n_quantiles)
        # For a single ticker we take the last sample.
        if raw_predictions.ndim == 3:
            quantiles = raw_predictions[-1].cpu().numpy()  # (5, 7)
        else:
            quantiles = raw_predictions.cpu().numpy()       # (5, 7)

        # Future dates: next 5 calendar days from last date in data
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(5)]

        return quantiles, future_dates

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.exception(e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Application
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎛️ Flight Control")
        st.divider()

        ticker = st.selectbox(
            "Select Asset",
            options=TICKER_OPTIONS,
            index=0,
        )

        run_analysis = st.button("🔄 Run Analysis", use_container_width=True)

        st.divider()
        st.caption(
            "**Chimera Alpha Engine** v0.3  \n"
            "Phase 3 — Live Inference  \n"
            f"Model: `{MODEL_PATH.name}`"
        )

    # ── Header ───────────────────────────────────────────────────────────
    st.title("🧬 Chimera Alpha Engine: Live Inference")
    st.markdown("---")

    # ── Load Model ───────────────────────────────────────────────────────
    model = load_model()

    if model is None:
        st.warning(
            f"⚠️ **Model not found** at `{MODEL_PATH}`.  \n"
            "Please copy your trained `best_tft_model.ckpt` into the "
            "`models/` directory and refresh the page.",
            icon="🔍",
        )
        st.info(
            "💡 **Tip:** After training on Kaggle, download the `.ckpt` "
            "file and place it at:  \n"
            f"`{MODEL_PATH}`"
        )
        st.stop()

    # ── Data Download & Feature Engineering ──────────────────────────────
    with st.spinner(f"📡 Downloading & processing **{ticker}** data…"):
        df = load_market_data(ticker, days=300)

    if df is None or df.empty:
        st.error(
            f"❌ No data returned for **{ticker}**.  \n"
            "Check your internet connection or try a different symbol."
        )
        st.stop()

    # ── Metric Row (placeholders — AI Signal updated after inference) ────
    current_price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    change_pct = ((current_price - prev_close) / prev_close) * 100

    metric_cols = st.columns(3)
    metric_cols[0].metric("💰 Current Price", f"${current_price:,.2f}")
    metric_cols[1].metric(
        "📈 Daily Change",
        f"{change_pct:+.2f}%",
        delta=f"{change_pct:+.2f}%",
    )
    signal_placeholder = metric_cols[2].empty()  # filled after inference

    st.markdown("---")

    # ── Inference ────────────────────────────────────────────────────────
    with st.spinner("🧠 Running TFT inference…"):
        result = predict_next_5_days(model, df)

    if result is None:
        st.stop()

    quantiles, future_dates = result

    # Convert quantile *returns* → absolute price levels
    price_quantiles = current_price * (1.0 + quantiles)

    # ── AI Signal ────────────────────────────────────────────────────────
    median_price_day5 = float(price_quantiles[-1, Q_P50])

    if median_price_day5 > current_price:
        signal_placeholder.metric("🤖 AI Signal", "🟢 BULLISH")
    elif median_price_day5 < current_price:
        signal_placeholder.metric("🤖 AI Signal", "🔴 BEARISH")
    else:
        signal_placeholder.metric("🤖 AI Signal", "🟡 NEUTRAL")

    # ── Plotly Fan Chart ─────────────────────────────────────────────────
    hist_df = df.tail(60).copy()
    fig = build_fan_chart(hist_df, future_dates, price_quantiles, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # ── Trade Plan ───────────────────────────────────────────────────────
    st.subheader("📋 Trade Plan")

    p10_day5 = float(price_quantiles[-1, Q_P10])
    p90_day5 = float(price_quantiles[-1, Q_P90])
    risk = current_price - p10_day5
    reward = p90_day5 - current_price
    rr_ratio = reward / risk if risk > 0 else float("inf")

    trade_plan = pd.DataFrame(
        {
            "Metric": [
                "Entry (Current Price)",
                "Stop Loss (P10, Day 5)",
                "Take Profit (P90, Day 5)",
                "Median Target (P50, Day 5)",
                "Risk (Entry → SL)",
                "Reward (Entry → TP)",
                "Risk / Reward Ratio",
            ],
            "Value": [
                f"${current_price:,.2f}",
                f"${p10_day5:,.2f}",
                f"${p90_day5:,.2f}",
                f"${median_price_day5:,.2f}",
                f"${risk:,.2f}",
                f"${reward:,.2f}",
                f"1 : {rr_ratio:.2f}",
            ],
        }
    )
    st.dataframe(trade_plan, use_container_width=True, hide_index=True)

    # ── Raw Quantile Predictions ─────────────────────────────────────────
    with st.expander("📊 Raw Quantile Predictions"):
        pred_df = pd.DataFrame(
            price_quantiles,
            columns=["P2", "P10", "P25", "P50 (Median)", "P75", "P90", "P98"],
            index=[f"Day {i + 1}" for i in range(5)],
        )
        pred_df.insert(
            0, "Date", [d.strftime("%Y-%m-%d") for d in future_dates]
        )
        st.dataframe(pred_df, use_container_width=True)

    # ── Latest Feature Values ────────────────────────────────────────────
    with st.expander("🔬 Latest Feature Values"):
        latest = df.iloc[-1]
        feat_data = {
            "Feature": TIME_VARYING_UNKNOWN_REALS,
            "Value": [
                f"{float(latest[f]):.6f}" for f in TIME_VARYING_UNKNOWN_REALS
            ],
        }
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
