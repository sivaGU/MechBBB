# MechBBB merged Streamlit deployment

Python **3.10** recommended (`runtime.txt`: python-3.10.14).

## Local launch

```bash
cd MechBBB_Streamlit_Deployment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Equivalent entrypoint:

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Sidebar: Home / Documentation / Demo Prediction Tool / MechBBB-ML Prediction.

## Streamlit Cloud

1. Push this folder as the app root (or set the Cloud app root to `MechBBB_Streamlit_Deployment`).
2. Main file: `streamlit_app.py` (or `app.py`).
3. Python version: 3.10 (see `runtime.txt`).
4. Dependencies: `requirements.txt`. System packages: `packages.txt` if needed for RDKit builds.

## Model (revision Model C)

- Stage1 → 2061 features → 5 Stage2 seeds → mean → isotonic → calibrated P(BBB+)
- Primary threshold **0.51**; secondary **0.81** / **0.92**
- Artifacts under `artifacts/` (not historical `stage1_efflux.joblib` / `model_seed*.pkl`)
- No train_fps AD; no ±2×SE CI in this build

## Tests

```bash
cd MechBBB_Streamlit_Deployment
python -m pytest tests/test_inference.py -q
```
