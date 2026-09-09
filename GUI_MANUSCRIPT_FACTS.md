# GUI_MANUSCRIPT_FACTS

UTC: 2026-09-08T18:10:17.334477+00:00

**GUI synchronized with frozen revision Model C:** YES (merged historical UI + frozen inference)

## Verified implemented functionality

- **Inputs:** SMILES text; structure file upload (best-effort); batch CSV with SMILES/smiles/smi column
- **Batch:** row order preserved; per-row errors; downloadable CSV without CI columns
- **Models:** Stage1 final_refit_v3 boosters; Stage2 Model C seeds 0–4; validation isotonic calibrator
- **Outputs:** Stage1 scores (uncalibrated), raw ensemble mean, calibrated P(BBB+), binary class at selected threshold, InChIKey, canonical SMILES
- **Primary threshold:** 0.51 (calibrated)
- **Optional thresholds:** 0.81 high-sensitivity, 0.92 high-specificity, plus user-adjusted slider
- **Similarity warning:** not enabled (`get_train_fps` returns None)

## Do not claim

- That the public Streamlit Cloud site is already updated
- That Stage1 scores are experimental transporter measurements
- That 0.35 is the revision MCC-optimal threshold
- That ±2×SE ensemble intervals are validated uncertainty estimates
- That train_fps similarity is a validated applicability domain

## Archive provenance

- MechBBB-main (1).zip SHA256 = `85d0413ded039450ebade97ab9dc6a37bffce90493200e8c33153ef89e405038`
- MechBBB_Streamlit_Deployment.zip SHA256 = `4078fe2f85eddda1c9307e12b97a16584a03015758f3925424127d186024a71a`
