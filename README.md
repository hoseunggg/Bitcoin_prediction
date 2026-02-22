Bitcoin Prediction (On-chain + ML)
=================================

This repository contains code and materials accompanying our published paper:

- Paper (DOI): [Bitcoin forecasting with machine learning and on-chain information](https://doi.org/10.1080/10293523.2026.2616575)
- PDF (in this repo): [Bitcoin forecasting with machine learning and on-chain](./Bitcoin%20forecasting%20with%20machine%20learning%20and%20on-chain.pdf)
- 
Goal
----
This repository explores short-horizon Bitcoin forecasting (e.g., next-day return) using
machine learning with on-chain + market/macro signals. It includes training, evaluation,
and basic interpretability (feature importance / SHAP) workflows.

What this project does
----------------------
- Builds a dataset aligned on a daily timeline (UTC recommended).
- Trains ML models (e.g., Random Forest, XGBoost, SVR, MLP) and a simple baseline.
- Evaluates out-of-sample performance using time-aware splits (rolling/expanding windows).
- Explains predictions using feature importance (Permutation Importance) and SHAP.

Project Structure
-----------------
```txt
.
├─ Bitcoin forecasting with machine learning and on-chain.pdf
├─ config.py         # path
├─ data_preprocess.py
├─ figure.py
├─ model.py
├─ table.py
├─ train.py
├─ train_category.py
├─ .vscode/
├─ dataset/          # input + intermediate CSVs
├─ figure/           # summary figures / CSV outputs for plots
├─ result/           # evaluation outputs (RMSE, DM tests, error breakdowns)
└─ __pycache__/
```
Data notes
----------
Typical feature groups:
- On-chain: fees, transaction volume, active addresses, mining stats (hashrate/difficulty), etc.
- Market: volatility proxies, returns, volume, etc.
- Macro: rates, spreads, risk indicators, etc.

Important:
- Avoid look-ahead bias. For a target at time t+1, do not use features only known after time t.
- Keep time ordering for splits; do not randomly shuffle.

Run
---
1) (Optional) Preprocess data:
   python data_preprocess.py

2) Train / evaluate:
   python train.py

3) Category-based experiments (if used):
   python train_category.py

Evaluation
----------
- RMSE / MAE
- Out-of-sample R^2
- Directional accuracy (sign hit rate)

Interpretability
----------------
- Permutation Feature Importance (PFI): importance via performance drop after shuffling a feature.
- SHAP: decomposes predictions into per-feature contributions (global + local explanations).

Reproducibility checklist
-------------------------
- Fix random seeds
- Use a consistent preprocessing pipeline (scaling/winsorizing/normalization)
- Use rolling/expanding evaluation, and save metrics + model artifacts
- Log feature sets, date ranges, and hyperparameters for each run

