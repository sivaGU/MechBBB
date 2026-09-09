# GUI_MERGE_AUDIT

UTC: 2026-09-08T18:10:17.334477+00:00

## Sources

| Role | Path | Archive SHA256 |
|------|------|----------------|
| Historical UI | `sources_readonly/MechBBB-main (1)/MechBBB-main` | MechBBB-main (1).zip = `85d0413ded039450ebade97ab9dc6a37bffce90493200e8c33153ef89e405038` |
| Frozen inference | `sources_readonly/MechBBB_Streamlit_Deployment/MechBBB_Streamlit_Deployment` | MechBBB_Streamlit_Deployment.zip = `4078fe2f85eddda1c9307e12b97a16584a03015758f3925424127d186024a71a` |
| Merged DEST | `GUI_merged_revision_v2/MechBBB_Streamlit_Deployment` | (this build) |

## What was taken from historical UI

- `streamlit_app.py` (patched), `demo_ligands.py`, `similarity_module.py`, `example_inputs.csv`, `packages.txt`, `runtime.txt`
- Historical look preserved: blue-teal CSS, sidebar nav Home/Documentation/Demo/Prediction, RDKit 2D drawing + optional CACTUS fallback
- **Not** copied: historical `artifacts/`, historical `src/mechbbb/predict.py` (or other historical models)

## What was taken from frozen package

- Entire `artifacts/` (revision Model C)
- `src/mechbbb/{chem.py,features.py,__init__.py}` + replaced `predict.py` (threshold isolation)
- `fixtures/frozen_reference_subset.csv`, `requirements.txt`, `.gitignore`, `.streamlit/`

## Key patches

1. Primary threshold **0.51**; radio for 0.51 / 0.81 / 0.92 / user-adjusted
2. Removed ±2×SE CI, Uncertainty Analysis, "BBB permeability confidence" framing
3. Show calibrated P(BBB+), raw ensemble mean, class, threshold separately
4. Stage1 labeled as uncalibrated model outputs
5. Similarity/AD disabled (`get_train_fps` → None; no `train_fps.npz`)
6. Home/Docs corrected (no thr 0.35 as MCC-optimal; revision artifact paths)
7. Demo caption: illustrative/literature-reported, unverified; revision Model C predictions
8. Batch CSV columns without CI
9. `predict()` never mutates shared `default_threshold`
10. `app.py` thin entry → `streamlit_app.main()`; page config/CSS inside `main()`

## Verification

- Frozen 18-ref equivalence: **0** class mismatches; max |Δ cal| ~1e-16
- Pytest: **6 passed**
- Smoke: `load_predictor('.')` + predict `CCO` from DEST cwd: **PASS**
