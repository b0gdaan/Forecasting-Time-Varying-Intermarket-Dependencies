# Forecasting Time-Varying Intermarket Dependencies

**Master's thesis** · Forecasting rolling correlations between Bitcoin and conventional financial assets using machine learning.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/Models-8-orange)

---

## What it does

Predicts **1-step-ahead rolling Pearson correlations** between `BTC-USD` and six assets across four time windows using walk-forward evaluation.

| Dimension | Values |
|---|---|
| **Base asset** | BTC-USD |
| **Paired with** | S&P 500 · NASDAQ · Gold · Silver · Dollar Index · ETH |
| **Rolling windows** | 14 · 30 · 60 · 90 days |
| **Models** | Naive · AR(1) · ElasticNet · Ridge · RandomForest · GBM · XGBoost · DCC-GARCH |
| **Target transform** | Fisher-z (arctanh) for variance stabilization |
| **Evaluation** | Walk-forward (expanding window), MAE / RMSE / R², Diebold–Mariano tests |
| **Data** | Yahoo Finance · 2016–2026 · ~10 years |

---

## Results

- **XGBoost** achieved the lowest RMSE across most BTC–asset pairs at the 30-day window
- **DCC-GARCH** outperformed ML models on short windows (14d) for BTC–S&P500
- Fisher-z transformation consistently improved model stability vs raw correlations
- Diebold–Mariano tests confirmed statistical significance of XGBoost gains over AR(1) baseline

---

## Project Structure

```
├── main.py                  # Entry point — runs the full pipeline
├── config.yaml              # All settings (assets, windows, models, GPU)
├── requirements.txt
├── setup_dirs.py            # Creates output folders
│
├── thesis_app/
│   ├── pipeline.py          # Core: data → features → models → metrics → DM tests
│   └── dcc.py               # DCC-GARCH(1,1) econometric benchmark
│
├── notebooks/
│   ├── 01_EDA_Dataset.ipynb       # Price/return analysis, ADF tests, Fisher-z illustration
│   ├── 02_GridSearch.ipynb        # Hyperparameter tuning (TimeSeriesSplit CV)
│   ├── 03_Model_Comparison.ipynb  # RMSE/R² heatmaps, model ranking
│   ├── 04_DM_Tests_Visuals.ipynb  # Diebold–Mariano tests, publication-quality figures
│   └── 05_XGB_vs_DCC.ipynb        # Deep dive: XGBoost vs DCC-GARCH error analysis
│
├── data/
│   ├── raw/prices.csv             # Auto-downloaded on first run
│   └── processed/returns.csv
│
└── outputs/
    ├── figures/                   # All plots (PNG, 130 dpi)
    ├── predictions/               # Per-experiment forecast CSVs
    ├── results/
    │   ├── metrics.csv            # MAE / RMSE / R² per model
    │   └── dm_tests.csv           # Diebold–Mariano test results
    └── tables/
        ├── metrics_table.tex      # LaTeX table (thesis-ready)
        └── dm_tests.tex
```

---

## Setup

```bash
git clone https://github.com/b0gdaan/Forecasting-Time-Varying-Intermarket-Dependencies.git
cd Forecasting-Time-Varying-Intermarket-Dependencies

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run

```bash
# Full pipeline: download data → all models → metrics → DM tests → figures
python main.py
```

First run downloads ~10 years of price data from Yahoo Finance (~30 sec).  
Subsequent runs use cached `data/raw/prices.csv`.

```bash
# Then explore results in notebooks
jupyter notebook
```

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `rolling_windows` | `[14,30,60,90]` | Correlation window sizes (days) |
| `use_fisher_transform` | `true` | Fisher-z transform on target |
| `use_dcc_garch` | `true` | Include DCC-GARCH benchmark |
| `use_xgboost` | `true` | Include XGBoost |
| `xgb_device` | `cuda` | `"cuda"` for GPU · `"cpu"` for CPU-only |
| `min_train_size` | `800` | Minimum training observations |
| `refit_every` | `20` | Walk-forward refit frequency (days) |

**No GPU?** Set `xgb_device: "cpu"` in `config.yaml`.  
**No `arch` package?** Set `use_dcc_garch: false`.

---

## Tech Stack

`Python 3.11` · `XGBoost` · `scikit-learn` · `pandas` · `statsmodels` · `arch (DCC-GARCH)` · `matplotlib` · `Yahoo Finance API`
