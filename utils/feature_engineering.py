def add_lag_features(df):
    """
    Adds lagged features to the input dataframe and drops NA rows.
    """
    df["Target_NextWeekDrop"] = (df["SP500_Returns"].shift(-1) < 0).astype(int)
    df["Mood_Index_Lag1"] = df["Mood_Index"].shift(1)
    df["SP500_Returns_Lag1"] = df["SP500_Returns"].shift(1)
    df["VIX_Change_Lag1"] = df["VIX_Change"].shift(1)
    df["Google_Trend_Lag1"] = df["Google_Sentiment_Index"].shift(1)
    df["Unemployment_Lag1"] = df["Unemployment"].shift(1)

    df = df.dropna(subset=[
        "Mood_Index", "Mood_Index_Lag1",
        "SP500_Returns_Lag1", "VIX_Change_Lag1",
        "Google_Trend_Lag1", "Unemployment_Lag1",
        "Target_NextWeekDrop"
    ]).copy()
    
    return df