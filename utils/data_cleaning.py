from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_mood_index(df, vix_col='VIX_Close', google_col='Google_Sentiment_Index', unemp_col='Unemployment'):
    scaler = MinMaxScaler()
    norm_values = scaler.fit_transform(df[[vix_col, google_col, unemp_col]])
    norm_df = pd.DataFrame(norm_values, columns=["VIX_Norm", "Google_Norm", "Unemp_Norm"])
    norm_df.index = df.index
    df = df.copy()
    df[["VIX_Norm", "Google_Norm", "Unemp_Norm"]] = norm_df
    df["Mood_Index"] = df[["VIX_Norm", "Google_Norm", "Unemp_Norm"]].mean(axis=1)
    df["Mood_Zone"] = df["Mood_Index"].apply(lambda val: "Calm" if val < 0.4 else "Cautious" if val < 0.7 else "Panic")
    return df