# GUI_FUNCTIONAL_TEST_REPORT

UTC: 2026-09-08T18:10:17.334477+00:00

## Pytest (`tests/test_inference.py`)

| Test | Result |
|------|--------|
| test_load_and_dim | PASS |
| test_invalid | PASS |
| test_frozen_subset_classes | PASS (18 refs, 0 mismatches) |
| test_batch_alignment | PASS |
| test_threshold_isolation | PASS |
| test_no_historical_artifact_names_loaded | PASS |

**Summary: 6 passed**

## Smoke (DEST cwd only)

```text
load_predictor('.') ; predict_single('CCO')
→ is_valid=True, prob≈0.9841, bbb_class=BBB+, threshold=0.51, n_features=2061
```

## Demo audit

- `DEMO_LIGAND_AUDIT.csv`: 50 compounds; RDKit parse failures: **0**
- Optional `demo_predictions_revision.csv`: 50/50 valid predictions at thr=0.51
- Labels: `illustrative_literature_reported_unverified` (no invented citations)

## Unresolved / known limitations

- Similarity / AD: disabled; `train_fps.npz` not shipped
- Demo labels are illustrative only; not independently verified against primary literature in this merge
- CACTUS image fallback requires network; drawing failure does not affect prediction
- No live Streamlit Cloud deploy performed
