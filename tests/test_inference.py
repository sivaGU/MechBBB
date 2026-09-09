"""Equivalence and workflow tests for merged MechBBB deployment package."""
from pathlib import Path

import pandas as pd

from src.mechbbb.predict import (
    HIGH_SENS_THRESHOLD,
    HIGH_SPEC_THRESHOLD,
    PRIMARY_THRESHOLD,
    load_predictor,
    predict_batch,
    predict_single,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_and_dim():
    p = load_predictor(ROOT)
    r = predict_single("CCO", threshold=PRIMARY_THRESHOLD, predictor=p)
    assert r.is_valid
    assert r.n_features == 2061
    assert not hasattr(r, "prob_std_error") or getattr(r, "prob_std_error", None) is None


def test_invalid():
    p = load_predictor(ROOT)
    r = predict_single("not_smiles", predictor=p)
    assert not r.is_valid
    assert r.error


def test_frozen_subset_classes():
    p = load_predictor(ROOT)
    ref = pd.read_csv(ROOT / "fixtures" / "frozen_reference_subset.csv")
    mism = 0
    max_diff = 0.0
    for _, row in ref.iterrows():
        r = predict_single(row["canonical_smiles"], threshold=PRIMARY_THRESHOLD, predictor=p)
        assert r.is_valid
        pred = 1 if r.prob >= PRIMARY_THRESHOLD else 0
        if pred != int(row["y_pred_locked_thr"]):
            mism += 1
        diff = abs(r.prob - float(row["y_prob_calibrated"]))
        max_diff = max(max_diff, diff)
        assert diff < 1e-5
    assert mism == 0
    assert max_diff < 1e-5


def test_batch_alignment():
    p = load_predictor(ROOT)
    smiles = ["CCO", "bad", "", "c1ccccc1"]
    out = predict_batch(smiles, predictor=p)
    assert len(out) == 4
    assert out[0].is_valid and out[3].is_valid
    assert not out[1].is_valid and not out[2].is_valid


def test_threshold_isolation():
    """Two thresholds: same calibrated prob, possibly different class; shared default unchanged."""
    p = load_predictor(ROOT)
    default_before = p.default_threshold
    r1 = predict_single("CCO", threshold=PRIMARY_THRESHOLD, predictor=p)
    r2 = predict_single("CCO", threshold=HIGH_SPEC_THRESHOLD, predictor=p)
    assert r1.is_valid and r2.is_valid
    assert abs(r1.prob - r2.prob) < 1e-12
    assert abs(r1.prob_raw - r2.prob_raw) < 1e-12
    assert r1.threshold == PRIMARY_THRESHOLD
    assert r2.threshold == HIGH_SPEC_THRESHOLD
    assert p.default_threshold == default_before
    # At least for ethanol, high-spec (0.92) should not flip class vs primary unless near boundary;
    # still require that class is derived from each threshold independently.
    assert r1.bbb_class == ("BBB+" if r1.prob >= PRIMARY_THRESHOLD else "BBB-")
    assert r2.bbb_class == ("BBB+" if r2.prob >= HIGH_SPEC_THRESHOLD else "BBB-")
    r3 = predict_single("CCO", threshold=HIGH_SENS_THRESHOLD, predictor=p)
    assert abs(r3.prob - r1.prob) < 1e-12
    assert p.default_threshold == default_before


def test_no_historical_artifact_names_loaded():
    """Loader must resolve revision layout, not historical stage1_efflux / model_seed*.pkl names."""
    art = ROOT / "artifacts"
    assert (art / "stage1" / "efflux.joblib").exists()
    assert (art / "stage2_modelC" / "seed0.txt").exists()
    assert (art / "calibration" / "isotonic_model_C.joblib").exists()
    assert not (art / "stage1_efflux.joblib").exists()
    assert not (art / "train_fps.npz").exists()
    for i in range(5):
        assert not (art / "stage2_modelC" / f"model_seed{i}.pkl").exists()
    p = load_predictor(ROOT)
    assert p.default_threshold == PRIMARY_THRESHOLD or abs(p.default_threshold - PRIMARY_THRESHOLD) < 1e-9
