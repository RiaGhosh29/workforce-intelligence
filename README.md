# Workforce Intelligence

AI-Assisted Workforce Intelligence Dashboard — Predictive Attrition, Workforce Cost Simulation & AI Displacement Scoring.

This repository contains the code and artefacts for a business analytics project that combines an explainable attrition prediction model, a workforce cost simulation engine, and an AI displacement risk scoring framework into a single interactive dashboard.

## Live dashboard
The deployed application is available at: https://workforce-intelligence.streamlit.app/

## Analytical notebook
The exploratory data analysis and model training process is documented in a separate notebook:
`AI_Workforce_Intelligence_Analysis.ipynb` — [Open in Colab](https://colab.research.google.com/drive/10pPYOC33Jj_6XcA9DQBL9bhxKlCHB0hk)

## Data sources
- **IBM HR Analytics Employee Attrition & Performance** — a publicly available benchmark HR dataset used to train and evaluate the attrition prediction model, and as the basis for the workforce cost simulation and AI displacement scoring modules.
- **O*NET occupational database** — used to map IBM job roles to their nearest O*NET-SOC occupational equivalents, providing the automation-probability and task-repetitiveness inputs used by the AI displacement scoring framework.

## Model development process
The attrition model was developed by benchmarking baseline classifiers, selecting XGBoost as the final model, and tuning it via cross-validated hyperparameter search. SHAP was used to generate global and local explainability outputs. The serialized model, preprocessing pipeline, and feature list are stored as `xgb_attrition_model.pkl`, `preprocessor_xgb.pkl`, and `feature_names.pkl` respectively, with precomputed SHAP values in `shap_values_test.npy`.

## Repository contents
- `app.py` — Streamlit dashboard application (Overview, Attrition Risk, Cost Simulator, AI Displacement, and 2x2 Matrix pages)
- `xgb_attrition_model.pkl`, `preprocessor_xgb.pkl`, `feature_names.pkl` — trained model artefacts
- `shap_values_test.npy` — precomputed SHAP values for explainability
- `workforce_intelligence_scored.csv`, `cost_simulation_summary.csv`, `displacement_scores.csv`, `matrix_quadrant_summary.csv` — output datasets consumed by the dashboard
- `requirements.txt`, `runtime.txt` — Python dependencies and runtime version
- `.devcontainer/` — reproducible development container configuration

## How to run locally
1. Clone the repository: `git clone https://github.com/RiaGhosh29/workforce-intelligence.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`
4. Open the local URL shown in the terminal (typically `http://localhost:8501`)

## Authors
Ria Ghosh, Izabela Martirosyan
