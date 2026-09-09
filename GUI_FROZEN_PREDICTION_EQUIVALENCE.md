# GUI_FROZEN_PREDICTION_EQUIVALENCE

UTC: 2026-09-08T18:10:17.334477+00:00

## Reference sources (not regenerated)

- Fixture: `fixtures/frozen_reference_subset.csv` (18 compounds from BBBP_test + strict_B3DB locked evaluation)
- Upstream revision CSVs (not re-run for this merge): `revision/03_stage2_rebuild/evaluation/bbbp_test_predictions_model_C.csv`, `revision/04_strict_b3db/predictions/strict_b3db_predictions_model_C.csv`

## Tolerances

| Quantity | Max allowed |abs diff| |
|----------|----------------------|
| Calibrated P(BBB+) | 1e-05 (pytest); observed ~1e-16 |
| Classification @ 0.51 | **0 mismatches required** |

## Results (this merge build)

- Reference compounds tested: **18**
- Max |Δ calibrated|: **1.110e-16**
- Classification mismatches: **0**
- Equivalence pass: **True**
- Pytest `tests/test_inference.py`: **6 passed**

## Threshold isolation

- `predict(smiles, threshold=...)` uses per-request threshold
- `default_threshold` on the shared predictor is never assigned by `predict_single` / `predict_batch`
