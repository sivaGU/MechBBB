"""Frozen revision Model C inference for MechBBB GUI.

Pipeline (matches revision/03_stage2_rebuild/run_locked_evaluation.py):
  Stage1 Booster.predict -> p_efflux/p_influx/p_pampa
  Stage2 five Boosters .predict -> mean raw
  isotonic on ensemble mean -> calibrated P(BBB+)
  primary threshold 0.51 on calibrated scale

Per-request thresholds are passed into predict(); shared predictor state is never mutated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import joblib
import lightgbm as lgb
import numpy as np

from .chem import standardize_and_identify
from .features import compute_morgan, compute_physchem10

PRIMARY_THRESHOLD = 0.51
HIGH_SENS_THRESHOLD = 0.81
HIGH_SPEC_THRESHOLD = 0.92


@dataclass
class PredictResult:
    is_valid: bool
    smiles: str
    canonical_smiles: str
    inchikey: str
    prob_raw: float
    prob: float  # calibrated P(BBB+) — primary displayed probability
    bbb_class: str
    p_efflux: Optional[float]
    p_influx: Optional[float]
    p_pampa: Optional[float]
    threshold: float
    error: str
    seed_probs: Optional[List[float]] = None
    n_features: int = 0
    imputed_stage1: bool = False


class MechBBBRevisionPredictor:
    def __init__(
        self,
        stage1_efflux,
        stage1_influx,
        stage1_pampa,
        stage2_boosters: List,
        calibrator,
        threshold: float = PRIMARY_THRESHOLD,
        stage1_medians: Optional[dict] = None,
    ):
        self.stage1_efflux = stage1_efflux
        self.stage1_influx = stage1_influx
        self.stage1_pampa = stage1_pampa
        self.stage2_boosters = stage2_boosters
        self.calibrator = calibrator
        # Default operating point only; never mutated by predict_single/batch.
        self.default_threshold = float(threshold)
        self.stage1_medians = stage1_medians or {
            "p_efflux": 0.396630553345084,
            "p_influx": 0.0973444295171783,
            "p_pampa": 0.8446839961072456,
        }

    def _featurize_2058(self, smiles_list: Sequence[str]) -> np.ndarray:
        pc = compute_physchem10(list(smiles_list)).values.astype(np.float64)
        fp = compute_morgan(list(smiles_list), radius=2, n_bits=2048).astype(np.float64)
        return np.hstack([pc, fp])

    def predict(self, smiles: str, threshold: Optional[float] = None) -> PredictResult:
        thr = self.default_threshold if threshold is None else float(threshold)
        mid = standardize_and_identify(smiles)
        if mid is None:
            return PredictResult(
                is_valid=False,
                smiles=smiles or "",
                canonical_smiles="",
                inchikey="",
                prob_raw=0.0,
                prob=0.0,
                bbb_class="BBB-",
                p_efflux=None,
                p_influx=None,
                p_pampa=None,
                threshold=thr,
                error="Invalid or unparseable SMILES",
            )
        canon = mid.canonical_smiles
        X1 = self._featurize_2058([canon])
        if X1.shape != (1, 2058) or np.isnan(X1).any():
            return PredictResult(
                is_valid=False,
                smiles=smiles,
                canonical_smiles=canon,
                inchikey=mid.inchikey,
                prob_raw=0.0,
                prob=0.0,
                bbb_class="BBB-",
                p_efflux=None,
                p_influx=None,
                p_pampa=None,
                threshold=thr,
                error="Could not compute PhysChem10+ECFP4 features",
            )
        p_efflux = float(self.stage1_efflux.predict(X1)[0])
        p_influx = float(self.stage1_influx.predict(X1)[0])
        p_pampa = float(self.stage1_pampa.predict(X1)[0])
        imputed = False
        for name, val in (("p_efflux", p_efflux), ("p_influx", p_influx), ("p_pampa", p_pampa)):
            if not np.isfinite(val):
                if name == "p_efflux":
                    p_efflux = float(self.stage1_medians["p_efflux"])
                elif name == "p_influx":
                    p_influx = float(self.stage1_medians["p_influx"])
                else:
                    p_pampa = float(self.stage1_medians["p_pampa"])
                imputed = True
        M = np.array([[p_efflux, p_influx, p_pampa]], dtype=np.float64)
        X2 = np.hstack([X1, M])
        if X2.shape != (1, 2061):
            return PredictResult(
                is_valid=False,
                smiles=smiles,
                canonical_smiles=canon,
                inchikey=mid.inchikey,
                prob_raw=0.0,
                prob=0.0,
                bbb_class="BBB-",
                p_efflux=p_efflux,
                p_influx=p_influx,
                p_pampa=p_pampa,
                threshold=thr,
                error=f"Feature dim mismatch: {X2.shape}",
            )
        seed_probs = [float(b.predict(X2)[0]) for b in self.stage2_boosters]
        raw = float(np.mean(seed_probs))
        cal = float(self.calibrator.predict(np.array([raw]))[0])
        bbb_class = "BBB+" if cal >= thr else "BBB-"
        return PredictResult(
            is_valid=True,
            smiles=smiles,
            canonical_smiles=canon,
            inchikey=mid.inchikey,
            prob_raw=raw,
            prob=cal,
            bbb_class=bbb_class,
            p_efflux=p_efflux,
            p_influx=p_influx,
            p_pampa=p_pampa,
            threshold=thr,
            error="",
            seed_probs=seed_probs,
            n_features=2061,
            imputed_stage1=imputed,
        )


def _load_booster(path: Path):
    path = Path(path)
    if path.suffix == ".txt":
        return lgb.Booster(model_file=str(path))
    obj = joblib.load(path)
    if isinstance(obj, lgb.Booster):
        return obj
    raise TypeError(f"Unsupported model type at {path}: {type(obj)}")


def load_predictor(artifact_dir: Union[str, Path] = ".") -> MechBBBRevisionPredictor:
    base = Path(artifact_dir)
    if (base / "artifacts" / "stage1").exists():
        art = base / "artifacts"
    elif (base / "artifacts_revision" / "stage1").exists():
        art = base / "artifacts_revision"
    elif (base / "stage1").exists():
        art = base
    else:
        raise FileNotFoundError(
            f"Could not locate revision artifacts under {base} "
            "(expected artifacts/stage1 or artifacts_revision/stage1). "
            "Required layout: artifacts/stage1/{efflux,influx,pampa}.joblib, "
            "artifacts/stage2_modelC/seed{{0..4}}.txt, "
            "artifacts/calibration/isotonic_model_C.joblib"
        )

    s1 = art / "stage1"
    efflux = _load_booster(s1 / "efflux.joblib")
    influx = _load_booster(s1 / "influx.joblib")
    pampa = _load_booster(s1 / "pampa.joblib")

    s2 = art / "stage2_modelC"
    boosters = []
    for i in range(5):
        txt = s2 / f"seed{i}.txt"
        jb = s2 / f"seed{i}.joblib"
        if txt.exists():
            boosters.append(_load_booster(txt))
        elif jb.exists():
            boosters.append(_load_booster(jb))
        else:
            raise FileNotFoundError(f"Missing Stage2 seed{i} under {s2}")

    cal_path = art / "calibration" / "isotonic_model_C.joblib"
    calibrator = joblib.load(cal_path)

    thr = PRIMARY_THRESHOLD
    thr_path = art / "calibration" / "locked_threshold_model_C.json"
    if thr_path.exists():
        thr = float(json.loads(thr_path.read_text(encoding="utf-8"))["mcc_optimal"]["threshold"])
    cfg_path = art / "threshold.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("probability_scale") == "isotonic_calibrated_ensemble_mean":
            thr = float(cfg.get("threshold", thr))

    medians = {
        "p_efflux": 0.396630553345084,
        "p_influx": 0.0973444295171783,
        "p_pampa": 0.8446839961072456,
    }
    med_path = art / "stage1_train_medians.json"
    if med_path.exists():
        medians.update(
            {
                k: v
                for k, v in json.loads(med_path.read_text(encoding="utf-8")).items()
                if k.startswith("p_")
            }
        )

    return MechBBBRevisionPredictor(
        stage1_efflux=efflux,
        stage1_influx=influx,
        stage1_pampa=pampa,
        stage2_boosters=boosters,
        calibrator=calibrator,
        threshold=thr,
        stage1_medians=medians,
    )


def predict_single(
    smiles: str,
    threshold: Optional[float] = None,
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[MechBBBRevisionPredictor] = None,
) -> PredictResult:
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return predictor.predict(smiles, threshold=threshold)


def predict_batch(
    smiles_list: List[str],
    threshold: Optional[float] = None,
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[MechBBBRevisionPredictor] = None,
) -> List[PredictResult]:
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return [predictor.predict(s, threshold=threshold) for s in smiles_list]
