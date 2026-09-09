# GUI_PRESERVED_FEATURES

UTC: 2026-09-08T18:10:17.334477+00:00

## Preserved from historical MechBBB GUI

- Blue-teal CSS theme and sidebar gradient navigation
- Pages: Home, Documentation, Demo Prediction Tool, MechBBB-ML Prediction
- `render_ligand_structure` (RDKit Cairo 2D primary)
- `get_mol_for_drawing` (SMILES-only 2D path)
- `fetch_structure_image_from_database` (optional NCI CACTUS fallback, timeout=10s)
- Structure file parsers (SDF/MOL/PDB/PDBQT/MOL2/CSV) for SMILES extraction
- Demo page with 50 ligands (25 CNS+ / 25 CNS−)
- Ligand preview slot on Prediction page
- Batch CSV upload + download

## Intentionally changed / disabled

- Default threshold 0.35 → **0.51** (revision MCC-optimal on calibrated scale)
- Ensemble ±2×SE CI / Uncertainty Analysis panels → **removed**
- Applicability-domain / `train_fps.npz` similarity → **disabled** (`get_train_fps` returns None)
- Inference backend → **frozen revision Model C** only (no historical joblib/pkl names)
