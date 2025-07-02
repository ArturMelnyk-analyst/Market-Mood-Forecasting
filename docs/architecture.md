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
  A[External APIs — Yahoo Finance, FRED, PyTrends] --> B[Raw Data CSVs — PR#1]
  B --> C[Data Cleaning & Mood Index — PR#2]
  C --> D[🔥 HOTFIX — Save Cleaned CSV — PR#2.1]
  D --> E[♻️ HOTFIX — Improved Saved Visuals — PR#2.2]
  E --> F[EDA & Statistical Tests — PR#3 & PR#4]
  F --> G[XGBoost Modeling — PR#5]
  G --> H[Explainability — SHAP — PR#6]
  H --> I[Final Notebook — PR#7]
  I --> J[Documentation — README.md, Tech Doc, Arch.md — PR#8]
  I --> K[Gradio Prototype — PR#9 Planned]
  K --> L[Hugging Face Spaces — PR#10 Planned]


## 🗂️ Folder Structure Diagram

```mermaid
graph TD
  Root --> data[📂 data/]
  Root --> docs[📂 docs/]
  Root --> images[📂 images/]
  Root --> models[📂 models/]
  Root --> notebooks[📂 notebooks/]
  Root --> utils[📂 utils/]
  Root --> app.py[🗒️ app.py] 
  Root --> README.md[🗒️ README.md]
  Root --> requirements.txt[🗒️ requirements.txt]
  Root --> LICENSE[📜 LICENSE]
  Root --> .gitignore[⚙️ .gitignore]
  Root --> .env.example.txt[⚙️ .env.example.txt

### ✨ **Pro tip:**
> _Tip: Mermaid diagram rendering may require VS Code Preview or GitHub plugin._


## 📁 Folders & Key Files

PathDescription/data/Raw .csv files saved after hotfix to ensure reproducibility./docs/Project documentation assets. Contains Technical_Documentation.pdf and architecture.md./images/Versioned plots for EDA, feature engineering, modeling, SHAP explainability./models/Final xgb_market_mood.pkl model; .gitkeep used initially to track empty folders./notebooks/Jupyter notebooks for each PR step, plus 07_final_notebook.ipynb./utils/Any helper scripts or functions if needed.app.py(Planned) Gradio prototype for live risk prediction and local SHAP display.README.mdProject overview, workflow, example visuals, and links to full documentation.requirements.txtPython packages needed for local dev and Gradio deployment.LICENSEProject license file..gitignoreVersion control exclusions to hide raw files but keep cleaned data and model..env.example.txtExample environment config file for local runs.##🔗 Version Control & Workflow
✅ Pull Request Structure (0–10)

PR#0: Initial repo setup — folders, .gitignore, README.md, LICENSE.

PR#1: Load raw datasets; store .csv in /data/.

PR#2: Cleaning, alignment, Mood Index.

PR#2.1: 🔥 Hotfix — save reproducible .csv to break API dependency.

PR#2.2: Hotfix — Regenerate saved visuals with clearer axes, larger fonts, and high-res images for presentation

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
