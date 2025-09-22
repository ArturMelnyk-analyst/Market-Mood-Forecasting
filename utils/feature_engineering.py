import pandas as pd
import numpy as np

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataset:
    - Target: next week's SP500 drop (binary)
    - Lags: Mood_Index, SP500_Returns, VIX_Change, Google_Sentiment_Index, Unemployment
    - Macro lags: include lag1 and lag2
    - Past-only rolling stats + stability ratios
    - Guardrails: ensure no forbidden columns (Mood_Zone_Cat, contemporaneous macros for modeling)
    """

    # --- Target ---
    df["Target_NextWeekDrop"] = (df["sp500_returns"].shift(-1) < 0).astype(int)

    # --- Lags for domain features ---
    df["Mood_Index_Lag1"] = df["Mood_Index"].shift(1)
    df["Google_Trend_Lag1"] = df["Google_Sentiment_Index"].shift(1)
    df["Unemployment_Lag1"] = df["Unemployment"].shift(1)

    # --- Macro lags ---
    df["sp500_returns_lag1"] = df["sp500_returns"].shift(1)
    df["sp500_returns_lag2"] = df["sp500_returns"].shift(2)
    df["vix_change_lag1"]    = df["vix_change"].shift(1)
    df["vix_change_lag2"]    = df["vix_change"].shift(2)

    # --- Past-only rolling stats ---
    def past_roll(col: pd.Series, window: int, fn: str = "mean"):
        s = col.shift(1).rolling(window=window, min_periods=window//2)
        if fn == "mean":
            return s.mean()
        if fn == "std":
            return s.std(ddof=0)
        if fn == "median":
            return s.median()
        raise ValueError("Unsupported fn")

    for w in (4, 8, 12):
        # S&P500
        df[f"sp500_ret_roll{w}_lag_mean"] = past_roll(df["sp500_returns"], w, "mean")
        df[f"sp500_ret_roll{w}_lag_std"]  = past_roll(df["sp500_returns"], w, "std")
        # VIX
        df[f"vix_change_roll{w}_lag_mean"] = past_roll(df["vix_change"], w, "mean")
        df[f"vix_change_roll{w}_lag_std"]  = past_roll(df["vix_change"], w, "std")

    # Stability ratios
    eps = 1e-9
    for w in (4, 8, 12):
        df[f"sp500_ret_roll{w}_stability"] = (
            df[f"sp500_ret_roll{w}_lag_mean"] / (df[f"sp500_ret_roll{w}_lag_std"] + eps)
        )
        df[f"vix_change_roll{w}_stability"] = (
            df[f"vix_change_roll{w}_lag_mean"] / (df[f"vix_change_roll{w}_lag_std"] + eps)
        )

    # --- Drop rows with NA in critical columns ---
    df = df.dropna(subset=[
        "Mood_Index", "Mood_Index_Lag1",
        "sp500_returns_lag1", "vix_change_lag1",
        "Google_Trend_Lag1", "Unemployment_Lag1",
        "Target_NextWeekDrop"
    ]).copy()

    # --- Guardrails ---
    assert "Mood_Zone_Cat" not in df.columns, "Forbidden column Mood_Zone_Cat found!"
    print("Guard OK → 'Mood_Zone_Cat' not present, raw macros EDA-only.")

    return df
