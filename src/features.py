"""Feature engineering module for technical indicators and signals.

All functions are pure: they take a DataFrame (with OHLCV columns) and
return it augmented with new feature columns.  No I/O happens here.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Trend  –  EMA Spread
# ---------------------------------------------------------------------------
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalised EMA-13 / EMA-21 spread.

    ``Trend_Spread = (EMA_13 - EMA_21) / Close``
    """
    df = df.copy()
    ema_13 = df["Close"].ewm(span=13, adjust=False).mean()
    ema_21 = df["Close"].ewm(span=21, adjust=False).mean()
    df["Trend_Spread"] = (ema_13 - ema_21) / df["Close"]
    return df


# ---------------------------------------------------------------------------
# 2. Momentum  –  Stochastic %K / %D
# ---------------------------------------------------------------------------
def add_momentum_features(
    df: pd.DataFrame,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Full Stochastic Oscillator (%K, %D) on 0–100 scale.

    * ``%K = SMA_smooth( (Close - Low_window) / (High_window - Low_window) * 100 )``
    * ``%D = SMA_smooth(%K)``
    * ``Is_Oversold = 1  if  %K < 20  else 0``
    """
    df = df.copy()
    lowest_low = df["Low"].rolling(window=window, min_periods=1).min()
    highest_high = df["High"].rolling(window=window, min_periods=1).max()

    raw_k = (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
    df["Stoch_K"] = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    df["Stoch_D"] = df["Stoch_K"].rolling(window=smooth_d, min_periods=1).mean()
    df["Is_Oversold"] = (df["Stoch_K"] < 20).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Volume  –  Relative Volume (RVOL)
# ---------------------------------------------------------------------------
def add_volume_features(df: pd.DataFrame, sma_window: int = 20) -> pd.DataFrame:
    """Relative volume vs. its 20-day simple moving average.

    ``RVOL = Volume / (SMA_20_Volume + 1e-9)``
    """
    df = df.copy()
    sma_vol = df["Volume"].rolling(window=sma_window, min_periods=1).mean()
    df["RVOL"] = df["Volume"] / (sma_vol + 1e-9)
    return df


# ---------------------------------------------------------------------------
# 4. Volatility  –  Log Returns
# ---------------------------------------------------------------------------
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Logarithmic daily returns.

    ``Log_Returns = ln(Close_t / Close_{t-1})``
    """
    df = df.copy()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ---------------------------------------------------------------------------
# 5. Label  –  Forward 5-day return (target variable)
# ---------------------------------------------------------------------------
def add_target_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Future *horizon*-day simple return used as the prediction target.

    ``Target_Return_5D = Close_{t+5} / Close_t  - 1``
    """
    df = df.copy()
    df[f"Target_Return_{horizon}D"] = (
        df["Close"].shift(-horizon) / df["Close"]
    ) - 1
    return df


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------
def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply **every** feature engineering step in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with columns
        ``['Open', 'High', 'Low', 'Close', 'Volume']``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame augmented with all computed features.
    """
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volume_features(df)
    df = add_volatility_features(df)
    df = add_target_label(df)
    return df
