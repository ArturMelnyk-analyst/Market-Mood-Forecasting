🗂️ Architecture.md — Market Mood Forecasting

## 📌 Project Structure
This project forecasts potential S&P 500 downturns using sentiment, volatility, and macroeconomic signals. It combines:

Multiple external APIs → local versioned .csv

Custom feature engineering: Mood Index, lags, rolling stats

Exploratory and statistical analysis

Predictive modeling with threshold tuning

SHAP explainability for transparent results

Final Gradio app prototype with optional deployment to Hugging Face Spaces


## 📊 Project Pipeline Flow

Below is the end-to-end workflow from raw data collection to deployment.  
This shows how each stage connects — matching your pull request structure.

```mermaid
graph TD
  A[External APIs<br>Yahoo Finance / FRED / PyTrends] --> B[Raw Data CSVs<br>PR#1 Load Data]
  B --> C[Data Cleaning & Alignment<br>Resample, Mood Index, Lags<br>PR#2]
  C --> D[🔥 Hotfix<br>Save Cleaned CSV<br>PR#2.1]
  D --> E[EDA & Statistical Tests<br>Plots, Annotations, Correlations<br>PR#3 & PR#4]
  E --> F[XGBoost Modeling<br>GridSearchCV, Threshold Tuning<br>PR#5]
  F --> G[Explainability<br>SHAP Plots: Beeswarm, Waterfall, Heatmap<br>PR#6]
  G --> H[Final Notebook<br>Markdown TOC, Saved Visuals<br>PR#7]
  H --> I[Documentation Assets<br>README.md, Technical Doc, architecture.md<br>PR#8]
  H --> J[Gradio App Prototype<br>app.py, Local Inference<br>PR#9 (Planned)]
  J --> K[Hugging Face Spaces<br>Live Public Deployment<br>PR#10 (Planned)]


## 🗂️ Folder Structure Diagram

```mermaid
graph TD
  Root --> data[📂 data/]
  Root --> models[📂 models/]
  Root --> images[📂 images/]
  Root --> notebooks[📂 notebooks/]
  Root --> app.py[🗒️ app.py]
  Root --> README.md[🗒️ README.md]
  Root --> Technical_Documentation.pdf[📕 Technical_Documentation.pdf]
  Root --> architecture.md[🗒️ architecture.md]


### ✨ **Pro tip:**
> _Tip: Mermaid diagram rendering may require VS Code Preview or GitHub plugin._


## 📁 Folders & Key Files

| Path                          | Description                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `/data/`                      | Raw `.csv` files saved after hotfix to ensure reproducibility.                                                               |
| `/models/`                    | Final `xgb_market_mood.pkl` model; `.gitkeep` used initially to track empty folders.                                         |
| `/images/`                    | Versioned plots: EDA, feature engineering, modeling, SHAP explainability.                                                    |
| `/notebooks/`                 | Jupyter notebooks for each PR step (`01_load_data.ipynb`, `02_clean_transform.ipynb`, etc.), plus `07_final_notebook.ipynb`. |
| `/app.py`                     | *(Planned)* Gradio prototype for risk prediction and local SHAP display.                                                     |
| `/requirements.txt`           | Python packages for local environment & Gradio deployment.                                                                   |
| `README.md`                   | Project overview, workflow, visuals, and links to `Technical_Documentation.pdf`.                                             |
| `Technical_Documentation.pdf` | Detailed workflow, version control structure, Visual Appendix.                                                               |
| `architecture.md`             | This file: high-level data flow & file connections.                                                                          |


##🔗 Version Control & Workflow
✅ Pull Request Structure (0–10)

PR#0: Initial repo setup — folders, .gitignore, README.md, LICENSE.

PR#1: Load raw datasets; store .csv in /data/.

PR#2: Cleaning, alignment, Mood Index.

PR#2.1: 🔥 Hotfix — save reproducible .csv to break API dependency.

PR#3: EDA — sentiment spikes, S&P overlays, major event annotations.

PR#4: Lag correlation, Granger causality, feature checks.

PR#5: XGBoost model, GridSearchCV, threshold tuning, ROC, feature importance.

PR#6: SHAP explainability plots: beeswarm, summary bar, waterfall, heatmap.

PR#7: Final combined notebook with Markdown TOC & visuals.

PR#8: README.md, Technical_Documentation.pdf, architecture.md final.

PR#9: (Planned) Gradio app for live risk prediction.

PR#10: (Planned) Hugging Face Spaces deployment with public badge.


##✅ Key Reproducibility Highlights
API calls replaced with local .csv after hotfix.

.gitignore updated to hide raw .csv but keep cleaned_data.csv.

Final notebook uses only versioned .csv and final xgb_market_mood.pkl.

##📌 Key Takeaway:
“Combining multiple volatile data sources makes reproducibility essential — versioned data, consistent file structure, and clear version control ensure this behavioral finance workflow stays robust and understandable for any reviewer or user.”

##🚀 Future-Proof: Gradio & Hugging Face
Gradio App: Takes user inputs for key features, displays drop probability + SHAP impact for transparency.

Hugging Face Spaces: Deploys the Gradio prototype with a public link and badge in the README.md.


##📌 How To Navigate
📄 README.md → Quick intro, PR flow, example visuals.

📚 Technical_Documentation.pdf → Full details, PR steps, Visual Appendix.

🗂️ architecture.md → This diagram & file map.
