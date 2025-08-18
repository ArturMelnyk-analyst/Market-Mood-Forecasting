\# 🗂️ Architecture.md — Market Mood Forecasting



\## 📌 Project Overview

The \*\*Market Mood Forecasting\*\* project predicts potential short-term S\&P 500 downturns by combining market volatility, search sentiment, and macroeconomic indicators.



This document outlines the \*\*project structure\*\*, \*\*pull request flow\*\*, and \*\*file organization\*\* for reproducibility.



---



\## 📊 Project Pipeline \& PR Flow

Below is the step-by-step development flow aligned with the PR structure.



```mermaid

graph TD
  A[PR#0: Repo Setup] --> B[PR#1: Load Raw Datasets]
  B --> C[PR#2: Data Cleaning & Mood Index]
  C --> D[PR#2.1: HOTFIX Utility Name Update]
  D --> E[PR#3: Exploratory Data Analysis]
  E --> F[PR#4: Feature Engineering]
  F --> G[PR#5: Modeling XGBoost + Comparison]
  G --> H[PR#6: Model Explainability - SHAP]
  H --> I[PR#7: Final Consolidated Notebook]
  I --> J[PR#8: Documentation & Assets - README, Tech Docs, Arch.md]
  J --> K[PR#9: Gradio App]
  K --> L[PR#10: Hugging Face Spaces Deployment]



📂 Folder Structure



```mermaid

graph TD
  Root --> data[📂 data/]
  data --> raw[📂 raw/]
  data --> cleaned[📂 cleaned/]
  Root --> docs[📂 docs/]
  Root --> images[📂 images/]
  images --> eda[📂 eda/]
  images --> feature_eng[📂 feature_engineering/]
  images --> modeling[📂 modeling/]
  images --> model_explain[📂 model_explain/]
  Root --> models[📂 models/]
  Root --> notebooks[📂 notebooks/]
  Root --> utils[📂 utils/]
  Root --> app_py[🗒️ app.py - planned PR#9]
  Root --> readme[🗒️ README.md]
  Root --> requirements[🗒️ requirements.txt]
  Root --> license[📜 LICENSE]
  Root --> gitignore[⚙️ .gitignore]
  Root --> env_example[⚙️ .env.example.txt]





📁 Folders \& Key Files



| Path                          | Description                                                                                     |

| ----------------------------- | ----------------------------------------------------------------------------------------------- |

| `/data/raw`                   | Original `.csv` datasets as loaded in PR#1                                                      |

| `/data/cleaned`               | Cleaned datasets from PR#2                                                                      |

| `/docs`                       | Documentation assets (Tech Doc, architecture.md)                                                |

| `/images/eda`                 | EDA visuals from PR#3                                                                           |

| `/images/feature\_engineering` | Feature engineering visuals from PR#4                                                           |

| `/images/modeling`            | ROC, PR curves, model visuals from PR#5                                                         |

| `/images/model\_explain`       | SHAP explainability plots from PR#6 (summary, beeswarm, dependence, waterfall, top 10 features) |

| `/models`                     | Final trained model (`XGBoost\_model.pkl`) from PR#5                                             |

| `/notebooks`                  | PR-aligned notebooks (`05\_modeling.ipynb`, `06\_model\_explain.ipynb`, `07\_final\_notebook.ipynb`) |

| `/utils`                      | Helper scripts for feature engineering and mood index                                           |

| `app.py`                      | Gradio interface for                                                               |

| `README.md`                   | Overview and workflow                                                                           |

| `requirements.txt`            | Package dependencies for local use and deployment                                               |

| `.gitignore`                  | Files/folders excluded from version control                                                     |

| `.gitattributes`              | LFS/Hugging Face export settings (to be added PR#10)                                            |





🔗 Version Control \& PR Workflow



| PR     | Status         | Description                                                                     |

| ------ | -------------- | ------------------------------------------------------------------------------- |

| PR#0   | ✅ Done         | Initial repo setup — structure, `.gitignore`, `README.md`                       |

| PR#1   | ✅ Done         | Load raw datasets to `/data/raw`, save reproducible `.csv`                      |

| PR#2   | ✅ Done         | Data cleaning, Mood Index calculation, save to `/data/cleaned`                  |

| PR#2.1 | ✅ Done         | HOTFIX: Utility name correction                                                 |

| PR#3   | ✅ Done         | Exploratory Data Analysis (EDA)                                                 |

| PR#4   | ✅ Done         | Feature Engineering                                                             |

| PR#5   | ✅ Done         | Modeling (XGBoost + model comparison)                                           |

| PR#6   | ✅ Done         | SHAP Explainability (summary, beeswarm, dependence, waterfall, top 10 features) |

| PR#7   | ✅ Done         | Final Consolidated Notebook                                                     |

| PR#8   | ✅ Done         | Documentation \& assets (README.md, Tech Doc, Arch.md)                          |

| PR#9   | ✅ Done         | Gradio app for prediction interface                                             |

| PR#10  | ✅ Done         | Hugging Face Spaces deployment                                                  |





📌 Reproducibility Highlights

No API dependencies in this version.



All datasets versioned in /data/raw and /data/cleaned.



Final model saved in /models/XGBoost\_model.pkl.



SHAP visuals saved to /images/model\_explain for PR#8 docs.





📌 How to Navigate

README.md → High-level summary, workflow, example visuals.



Technical\_Documentation.pdf → Detailed PR-by-PR description, metrics, and visuals.



architecture.md → Project architecture, folder structure, and PR sequence.





