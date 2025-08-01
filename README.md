\# 📊 Market Mood Forecasting



Forecasting potential S\&P 500 downturns using a custom Market Mood Index built from behavioral and macroeconomic indicators (Google Trends, VIX, Unemployment Rate).  

This project is designed as a \*\*portfolio-ready, explainable financial model\*\*, optimized for interpretability over raw accuracy.



---



\## 📌 Objective

\- Build an \*\*early-warning signal\*\* for market risk.  

\- Use \*\*Google Trends\*\*, \*\*VIX\*\*, and \*\*Unemployment Rate\*\* to construct a \*\*Mood Index\*\*.  

\- Predict whether the market will drop the following week.



---



\## 📂 PR-Based Workflow

| PR | Description |

|----|-------------|

| PR#0 | Repo setup (.gitignore, README, structure) |

| PR#1 | Load raw datasets → `/data/raw` |

| PR#2 | Clean \& align datasets → create Mood Index (`mood\_features.py`) |

| PR#2.1 | \*\*HOTFIX\*\*: Utility rename (`feature\_engineering.py`) |

| PR#3 | EDA (sentiment trends, volatility overlays) |

| PR#4 | Feature Engineering (lags, normalization, target creation) |

| PR#5 | Modeling (model comparison + threshold tuning) |

| PR#6 | Model Explainability (SHAP summary, beeswarm, dependence, waterfall, Top 10 features) |

| PR#7 | Final consolidated notebook |

| PR#8 | Documentation (README, Technical\_Documentation, architecture.md) |

| PR#9 | Gradio App (Planned) |

| PR#10 | Hugging Face Deployment (Planned) |



---



\## 📊 Modeling \& Results

\- \*\*Models tested:\*\* Logistic Regression, Random Forest, KNN, LightGBM, XGBoost  

\- \*\*Best model:\*\* XGBoost  

&nbsp; - F1 Mean: \*\*0.424\*\*  

&nbsp; - ROC AUC Mean: \*\*0.5169\*\*  

&nbsp; - Optimal Threshold: \*\*0.24\*\* (F1 ≈ 0.59, Recall ≈ 98%)  



\### Model Comparison

| Model              | F1 Mean | ROC AUC |

|--------------------|---------|---------|

| XGBoost            | 0.424   | 0.5169 |

| LightGBM           | 0.378   | 0.4957 |

| KNN                | 0.364   | 0.5255 |

| Logistic Regression| 0.322   | 0.4843 |

| Random Forest      | 0.295   | 0.5017 |

| Dummy              | 0.000   | 0.5000 |



---



\## 🧠 Explainability (SHAP)

Interpretability is the main focus:  

\- \*\*Summary Plot\*\* (global feature impact)  

\- \*\*Beeswarm Plot\*\* (distribution of feature effects)  

\- \*\*Dependence Plot\*\* (top feature: `VIX\_Change`)  

\- \*\*Top 10 Features Bar Chart\*\*  



---



\## 📦 Project Structure

Market-Mood-Forecasting/

│

├── data/

│ ├── raw/ # Original datasets

│ ├── cleaned/ # Cleaned datasets

│

├── images/

│ ├── eda/ # EDA visuals

│ ├── feature\_engineering/

│ ├── modeling/

│ ├── model\_explain/

│

├── models/

│ └── XGBoost\_model.pkl # Final model

│

├── notebooks/

│ ├── 05\_modeling.ipynb

│ ├── 06\_model\_explain.ipynb

│ ├── 07\_final\_notebook.ipynb

│

├── utils/

│ ├── mood\_features.py

│ ├── feature\_engineering.py

│

├── architecture.md

├── Technical\_Documentation.pdf

└── README.md





---



\## 🎯 Intended Usage

\- \*\*For portfolio demonstration\*\*: Model interpretability > accuracy.  

\- \*\*Not for high-frequency trading\*\* — best used as part of a \*\*risk dashboard\*\*.  

\- Planned deployment via:

&nbsp; - \*\*Gradio App\*\* (PR#9)

&nbsp; - \*\*Hugging Face Spaces\*\* (PR#10)



---



\## 📚 Key Learnings

\- Temporal feature engineering improved stability.

\- SHAP analysis provided transparency for model behavior.

\- Eliminating API dependencies increased reproducibility.

\- Clear PR-based workflow improves portfolio credibility.



---



\## 📚 Summary

**Note:** Multiple model families were tested and tuned (Logistic Regression, Random Forest, KNN, LightGBM, XGBoost).  
Performance is limited by the predictive signal in available data, not model capacity.  
Further optimization would lead to marginal gains — focus is on interpretability and deployment.



---



\## 📖 Full Documentation

For complete details (EDA, features, modeling, explainability, visuals, limitations):  

📄 \[Technical Documentation (PDF)](./Technical\_Documentation.pdf)



---



## 📷 Sample Visuals
Below are selected outputs from EDA, feature engineering, modeling, and explainability stages:

**Mood Index vs S&P 500**  
![Mood Index vs S&P 500](https://raw.githubusercontent.com/ArturMelnyk-analyst/Market-Mood-Forecasting/main/images/eda/mood_vs_sp500.png)

**Lag Correlation**  
![Lag Correlation](https://raw.githubusercontent.com/ArturMelnyk-analyst/Market-Mood-Forecasting/main/images/feature_engineering/lag_correlation_google_vix.png)

**ROC Curve**  
![ROC Curve](https://raw.githubusercontent.com/ArturMelnyk-analyst/Market-Mood-Forecasting/main/images/modeling/roc_curve.png)

**SHAP Summary**  
![SHAP Summary](https://raw.githubusercontent.com/ArturMelnyk-analyst/Market-Mood-Forecasting/main/images/model_explain/shap_summary_plot.png)


