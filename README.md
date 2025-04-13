📉 Forecasting Market Fear: A Behavioral Finance Approach
Using Google Trends, VIX & Economic Indicators

📌 Objective
Design an early-warning system that forecasts potential S&P 500 downturns using:

Public sentiment (Google Trends)

Market volatility (VIX)

Unemployment rates

By engineering a Mood Index, this project predicts whether the market will drop the following week — focusing on risk detection rather than pure accuracy.

🧠 Background
This project merges behavioral finance and predictive modeling by:

Tracking public fear through online search behavior

Leveraging volatility and macroeconomic signals

Using machine learning to build an early risk detector

📁 Data Sources
Dataset	Source	Time Range
S&P 500 (^GSPC)	Yahoo Finance	2004–2025
VIX (Volatility Index)	Yahoo Finance	2004–2025
Google Trends	“stock market crash”	2004–2025
Unemployment Rate	FRED Economic Data	2004–2025
🔧 Project Workflow
Data Preparation

Resampled to weekly frequency (Fridays)

Cleaned and aligned all sources

Built a Mood_Index combining VIX, Google Trends & Unemployment (normalized)

Exploratory Analysis

Visualized sentiment, volatility, and market performance

Annotated major market events (e.g. 2008, COVID, 2022 inflation)

Statistical Analysis

Lag correlations (e.g. sentiment → VIX)

Granger causality testing

Predictive Modeling

Binary target: Market drop next week? (1 = yes, 0 = no)

Feature engineering: lags, volatility, rolling averages

Models:

Logistic Regression (baseline)

Random Forest

XGBoost (final model) with GridSearchCV (216 param combos)

Threshold optimization (maximize recall for market drop)

✅ Final Model: Tuned XGBoostClassifier
Metric	Value
Accuracy	41.6%
F1 Score (↓)	0.545
Recall (↓)	90.6% ✅
Precision (↓)	38.9%
🎯 Focus: Maximize recall for downturn detection, not pure accuracy.

💡 Key Features Implemented
get_mood_index(df) — custom function to create behavioral risk signal

Lag features: sentiment, volatility, unemployment

Volatility windows: 3-week rolling std for returns

GridSearchCV: hyperparameter tuning

SHAP Explainability: feature impact for stakeholders

📊 Visualizations
Mood Index vs. S&P 500 (with Panic Zones)
![image](https://github.com/user-attachments/assets/ef202c22-6118-41a3-94a5-a39be03795f9)


Annotated market events (2008 crisis, COVID crash, SVB failure)
![image](https://github.com/user-attachments/assets/3d8f917d-dff6-4e97-8ae4-42df5451dcce)



SHAP beeswarm plot for feature influence
![image](https://github.com/user-attachments/assets/c8ba95f3-4945-418c-ade2-8219bca3f3fa)


📦 Deliverables
market_fear_prediction.ipynb — full notebook

final_model_xgb.pkl — trained & tuned model

get_mood_index() — reusable scoring function

Visual assets: charts, SHAP, annotated mood timeline

📚 Key Learnings
Interpretability matters more than raw accuracy in risk prediction

Behavioral data is a powerful, underused signal

Lag-based features and threshold tuning significantly boost performance

SHAP provides transparency for model adoption in finance

🚀 Future Work
Add social sentiment (Reddit, Twitter, News APIs)

Include interest rates, inflation (CPI)

Deploy as an interactive dashboard (Streamlit, Dash)

📍 Summary
A machine learning approach to behavioral finance and market forecasting.
Built for analysts, investors, and fintech teams seeking to understand market mood shifts.
