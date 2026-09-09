"""
MechBBB-ML Streamlit GUI. Two-stage mechanistically augmented BBB permeability classifier (Model C).

Merged revision deployment: historical UI + frozen revision Model C inference.

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Ensure project root (this folder) is on path for src.mechbbb
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_DIR = PROJECT_ROOT

import io
import urllib.request
import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Native Cairo drawing needs libXrender on Linux. Streamlit Cloud APT may be broken;
# keep this import optional so the app can start and still run predictions.
try:
    from rdkit.Chem.Draw import rdMolDraw2D as _rdMolDraw2D
except Exception as _draw_import_err:  # ImportError or OSError (missing .so)
    _rdMolDraw2D = None
    logging.warning(
        "RDKit Cairo drawing unavailable (%s). "
        "Structure preview will use CACTUS fallback or show a nonfatal message.",
        _draw_import_err,
    )

from src.mechbbb.predict import (
    predict_single,
    predict_batch,
    load_predictor,
    PRIMARY_THRESHOLD,
    HIGH_SENS_THRESHOLD,
    HIGH_SPEC_THRESHOLD,
)
from demo_ligands import CNS_PENETRATING_LIGANDS, NON_CNS_PENETRATING_LIGANDS


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2, CSV (first row only).
    """
    try:
        ext = file_extension.lower()
        if ext == ".sdf":
            from io import StringIO
            sdf_data = StringIO(file_content.decode("utf-8"))
            supplier = Chem.SDMolSupplier(sdf_data)
            for m in supplier:
                if m is not None:
                    return Chem.MolToSmiles(m, canonical=True)
        elif ext == ".mol":
            mol = Chem.MolFromMolBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdbqt":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".mol2":
            try:
                mol = Chem.MolFromMol2Block(file_content.decode("utf-8"))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        elif ext == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            col = next((c for c in df.columns if c.lower() in ("smiles", "smi") or c == "SMILES"), None)
            if col and len(df) > 0:
                return str(df[col].iloc[0]).strip()
    except Exception:
        pass
    return None


def get_mol_with_3d(smiles: str, file_content: Optional[bytes] = None, file_extension: Optional[str] = None):
    """
    Get an RDKit mol with 3D coordinates for visualization.
    Uses uploaded file coords if present, else generates 3D from SMILES.
    Returns Chem.Mol or None.
    """
    mol = None
    if file_content is not None and file_extension is not None:
        ext = file_extension.lower()
        try:
            text = file_content.decode("utf-8")
            if ext == ".sdf":
                from io import StringIO
                supplier = Chem.SDMolSupplier(StringIO(text))
                mols = [m for m in supplier if m is not None]
                for m in mols:
                    if m.GetNumConformers() > 0:
                        mol = m
                        break
                else:
                    mol = mols[0] if mols else None
            elif ext == ".mol":
                mol = Chem.MolFromMolBlock(text)
            elif ext in (".pdb", ".pdbqt"):
                mol = Chem.MolFromPDBBlock(text)
            elif ext == ".mol2":
                mol = Chem.MolFromMol2Block(text)
            if mol is not None and mol.GetNumConformers() == 0:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                mol = None
        except Exception:
            mol = None
    if mol is None and smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except Exception:
                return None
    return mol


def fetch_structure_image_from_database(smiles: str, width: int = 400, height: int = 400) -> Optional[bytes]:
    """
    Optional fallback: fetch a 2D structure image from NCI CACTUS.
    Returns PNG image bytes or None on failure. Drawing failure must not affect prediction.
    """
    if not smiles or not str(smiles).strip():
        return None
    try:
        encoded = urllib.parse.quote(str(smiles).strip(), safe="")
        url = (
            f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/image"
            f"?width={width}&height={height}&format=png"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MechBBB-ML-GUI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
            if not data or len(data) < 100:
                return None
            return data
    except Exception:
        return None


def get_mol_for_drawing(smiles: Optional[str] = None, file_content: Optional[bytes] = None, file_extension: Optional[str] = None):
    """
    Get an RDKit mol for 2D structure drawing (no 3D embedding required).

    2D depictions must always come from SMILES (preferably canonical) so bond
    orders and aromaticity are correct. Uploaded file blocks (PDB/PDBQT/MOL2)
    are intentionally ignored for drawing — use get_mol_with_3d for coordinates.
    file_content / file_extension are accepted for API compatibility but unused.
    """
    del file_content, file_extension  # 2D path is SMILES-only; 3D uses get_mol_with_3d
    if smiles is None:
        return None
    smiles_str = str(smiles).strip()
    if not smiles_str:
        return None
    return Chem.MolFromSmiles(smiles_str)


def validate_demo_ligands() -> list:
    """
    Sanity-check demo SMILES with RDKit. Returns list of (name, smiles) that fail
    MolFromSmiles so bad entries cannot silently render as blank/wrong.
    """
    bad = []
    for name, smi in CNS_PENETRATING_LIGANDS + NON_CNS_PENETRATING_LIGANDS:
        if Chem.MolFromSmiles(smi) is None:
            bad.append((name, smi))
            logging.warning("Demo ligand SMILES failed RDKit parse: %s | %s", name, smi)
    return bad


def native_rdkit_drawing_available() -> bool:
    """True when rdMolDraw2D (Cairo) imported successfully (needs libXrender on Linux)."""
    return _rdMolDraw2D is not None


def render_ligand_structure(mol, size: int = 400) -> Optional[bytes]:
    """
    Draw the ligand as a 2D chemical structure using RDKit Cairo when available.
    Returns PNG bytes or None. Original drawing path retained for when system libs return.
    Prediction must never depend on this function.
    """
    if mol is None or _rdMolDraw2D is None:
        return None
    try:
        draw_size = max(300, int(size))
        drawer = _rdMolDraw2D.MolDraw2DCairo(draw_size, int(draw_size * 0.78))
        opts = drawer.drawOptions()
        opts.bondLineWidth = 3.0
        opts.padding = 0.02
        opts.baseFontSize = 0.95
        opts.minFontSize = 14
        opts.maxFontSize = 30
        opts.clearBackground = True

        draw_mol = Chem.Mol(mol)
        if draw_mol.GetNumConformers() > 0:
            draw_mol.RemoveAllConformers()
        AllChem.Compute2DCoords(draw_mol)

        _rdMolDraw2D.PrepareAndDrawMolecule(drawer, draw_mol)
        drawer.FinishDrawing()
        return bytes(drawer.GetDrawingText())
    except Exception:
        return None


def resolve_structure_preview(
    smiles: Optional[str],
    *,
    size: int = 400,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[bytes]:
    """
    Primary: local RDKit Cairo. Fallback: NCI CACTUS for canonical/standardized SMILES.
    Returns None if both fail (caller shows nonfatal 'Structure preview unavailable').
    """
    smiles_str = (smiles or "").strip()
    if not smiles_str:
        return None
    mol = get_mol_for_drawing(smiles_str)
    img = render_ligand_structure(mol, size=size) if mol is not None else None
    if img is not None:
        return img
    w = width if width is not None else size
    h = height if height is not None else int(size * 0.78)
    return fetch_structure_image_from_database(smiles_str, width=w, height=h)


CUSTOM_CSS = """
<style>
    /* Blue-teal palette: light powder -> midnight azure */
    :root {
        --light-powder-blue: #E0F4F8;
        --soft-sky-blue: #B3E5F0;
        --light-azure: #80D4E8;
        --medium-steel-blue: #4DB8D0;
        --deep-cerulean: #2A9DB5;
        --rich-teal-blue: #1E7A8C;
        --midnight-azure: #0D4F5C;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    section.main,
    .main,
    [data-testid="stAppViewContainer"] > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    div[data-testid="stAppViewContainer"] > div > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    .main .block-container,
    section.main .block-container {
        background-color: #ffffff !important;
        padding: 1.2rem 0.9rem 1.2rem 1.2rem;
        margin: 0.8rem 0.35rem 0.8rem 0.65rem;
        max-width: 1500px;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(13, 79, 92, 0.08);
    }
    
    .result-panel {
        border: 1px solid #d8ebef;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fdff 100%);
        margin-bottom: 0.55rem;
    }
    
    .result-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1E7A8C;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }
    
    .result-value {
        font-size: 1.48rem;
        font-weight: 800;
        color: #0D4F5C;
        line-height: 1.12;
    }
    
    .result-subtext {
        font-size: 0.82rem;
        color: #4c5f64;
        margin-top: 0.18rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D4F5C 0%, #0A3D47 100%);
        color: #ffffff;
        min-width: 200px !important;
        max-width: 280px !important;
        width: 280px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 280px !important;
        min-width: 200px !important;
        max-width: 280px !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #0A3D47;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4DB8D0 0%, #2A9DB5 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(77, 184, 208, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%);
        box-shadow: 0 4px 8px rgba(42, 157, 181, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton > button:focus {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%);
        box-shadow: 0 0 0 0.3rem rgba(77, 184, 208, 0.35);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%);
        color: white;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%);
    }
    
    h1, h2, h3 {
        color: #1E7A8C;
        font-weight: 700;
    }
    
    /* Slightly tighter section spacing (prediction-style subheaders) */
    .main h3 {
        margin-top: 0.65rem !important;
        margin-bottom: 0.35rem !important;
    }
    
    .main h4 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }

    .section-heading-compact {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E7A8C;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
    }
    
    a {
        color: #1E7A8C;
        text-decoration: none;
    }
    
    a:hover {
        color: #2A9DB5;
        text-decoration: underline;
    }
    
    [data-testid="stMetricValue"] {
        color: #1E7A8C;
        font-weight: 600;
        font-size: 1.42rem !important;
        line-height: 1.2 !important;
    }
    
    /* Tighten label → value gap in metrics */
    [data-testid="stMetric"] label p {
        margin-bottom: 0.1rem !important;
    }
    
    /* Progress bars sit closer to the value above */
    [data-testid="stProgress"] {
        margin-top: 0.2rem !important;
        margin-bottom: 0.35rem !important;
    }

    /* Tighten ligand preview vertical spacing */
    [data-testid="stImage"] {
        margin-top: 0.15rem !important;
        margin-bottom: 0.1rem !important;
    }

    [data-testid="stCaptionContainer"] {
        margin-top: 0.05rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Prediction summary info box: slightly less vertical padding */
    .main [data-baseweb="notification"] {
        padding-top: 0.55rem !important;
        padding-bottom: 0.55rem !important;
    }
    
    .stSuccess {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        border-left: 4px solid #4DB8D0;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        border-left: 4px solid #4DB8D0;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
        border-left: 4px solid #2A9DB5;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stError {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
        border-left: 4px solid #1E7A8C;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stRadio > label,
    .stSelectbox > label,
    .stTextInput > label,
    .stSlider > label,
    .stFileUploader > label {
        color: #1E7A8C;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        color: #1E7A8C;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
    }
    
    .stDataFrame {
        border: 2px solid #4DB8D0;
        border-radius: 4px;
    }
    
    hr {
        border-color: #4DB8D0;
        border-width: 2px;
    }
    
    .stSlider .stSlider > div > div {
        background-color: #4DB8D0;
    }
    
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%) !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%) !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255, 255, 255, 0.9);
    }
    
    [data-testid="stSidebar"] .stSuccess {
        background: linear-gradient(90deg, rgba(77, 184, 208, 0.3) 0%, rgba(42, 157, 181, 0.25) 100%);
        border-left: 4px solid #4DB8D0;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background: linear-gradient(90deg, rgba(77, 184, 208, 0.25) 0%, rgba(42, 157, 181, 0.2) 100%);
        border-left: 4px solid #4DB8D0;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .main .block-container > div {
        background-color: #ffffff;
    }
</style>
"""


# ============================================================================
# PREDICTOR (cached)
# ============================================================================

@st.cache_resource
def get_predictor():
    return load_predictor(HANDOFF_DIR)


@st.cache_resource
def get_train_fps():
    """Similarity / AD fingerprints are not shipped in this revision deployment."""
    return None


DEFAULT_THRESHOLD = PRIMARY_THRESHOLD


def _select_threshold(key_prefix: str = "pred") -> float:
    """Sidebar radio for primary / high-sens / high-spec / user-adjusted operating points."""
    st.sidebar.markdown("### Settings")
    mode = st.sidebar.radio(
        "Threshold mode",
        [
            f"Primary MCC-optimal ({PRIMARY_THRESHOLD})",
            f"High sensitivity ({HIGH_SENS_THRESHOLD})",
            f"High specificity ({HIGH_SPEC_THRESHOLD})",
            "User-adjusted",
        ],
        index=0,
        key=f"{key_prefix}_thr_mode",
    )
    if mode.startswith("Primary"):
        thr = PRIMARY_THRESHOLD
    elif mode.startswith("High sensitivity"):
        thr = HIGH_SENS_THRESHOLD
    elif mode.startswith("High specificity"):
        thr = HIGH_SPEC_THRESHOLD
    else:
        thr = st.sidebar.slider(
            "User-adjusted operating point (calibrated P(BBB+))",
            0.01,
            0.99,
            float(PRIMARY_THRESHOLD),
            0.01,
            key=f"{key_prefix}_thr_slider",
        )
        st.sidebar.info("User-adjusted operating point — not a confidence category.")
    st.sidebar.info(
        f"**Active threshold:** `{thr:.2f}` on calibrated P(BBB+).  \n"
        f"Revision Model C (5 seeds → mean → isotonic).  \n"
        f"Historical thr 0.35 is **not** the revision MCC-optimal threshold."
    )
    return float(thr)


# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    st.title("MechBBB-ML - Blood-Brain Barrier Permeability Studio")
    st.caption(
        "Two-stage mechanistically augmented BBB permeability classifier (revision Model C)."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        f"""
        - **Model focus:** BBB permeability classification (Model C)
        - **Architecture:** Stage-1 (efflux/influx/PAMPA) + Stage-2 (PhysChem+ECFP+mech)
        - **Primary threshold:** {PRIMARY_THRESHOLD} (MCC-optimal on calibrated validation scale)
        - **Secondary points:** {HIGH_SENS_THRESHOLD} (high sens), {HIGH_SPEC_THRESHOLD} (high spec)
        - **Status:** All pages available
        """
    )
    st.sidebar.success("Interactive ligand screening available!")

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams struggle to predict whether small molecules cross the blood-brain barrier.
        MechBBB-ML (Model C) is a two-stage mechanistically augmented classifier that first predicts
        auxiliary ADME-related scores (efflux, influx, PAMPA) and then combines them with physicochemical
        and fingerprint features to predict BBB permeability. This approach improves both
        external generalization and interpretability.
        """
    )

    st.markdown(
        f"""
        ### Model highlights
        - **Stage-1:** LightGBM models trained on auxiliary mechanistic datasets (BBBP excluded) yield uncalibrated p_efflux, p_influx, p_pampa (model scores, not experimental measurements).
        - **Stage-2:** Model C = PhysChem + ECFP4 + Stage-1 scores; 5-seed ensemble mean → validation isotonic calibration → calibrated P(BBB+).
        - **Primary threshold:** {PRIMARY_THRESHOLD} on the calibrated scale (revision MCC-optimal). Historical GUI thr 0.35 is not used.
        - **No unsupported CI/AD claims:** this build does not ship ensemble ±2×SE confidence intervals or train_fps applicability-domain fingerprints.
        """
    )

    st.divider()

    st.markdown("## Quick start")

    st.info(
        "**Ready to predict!** Use the **MechBBB-ML Prediction** page in the sidebar to enter SMILES strings or upload a CSV file and get BBB permeability predictions with Stage-1 scores."
    )

    st.markdown(
        """
        ---
        ### Navigation
        - **Home:** This overview
        - **Documentation:** Setup, model details, and usage
        - **Demo Prediction Tool:** Illustrative literature-reported ligands (50 compounds)
        - **MechBBB-ML Prediction:** Run predictions (SMILES, structure files, or Batch CSV)
        """
    )

    st.divider()

    st.markdown(
        """
        ### Publication and Contact

        **MechBBB: A Two-Stage Mechanism-Informed Machine Learning Tool for Blood-Brain Barrier Permeability Prediction.**  
        Yu Shin, Sahith Mada, Sivanesan Dakshanamurthy\\*  
        *Pharmaceuticals* (Submitted).

        **For Contact:**  
        Dr. Sivanesan Dakshanamurthy, PhD, MBA  
        [sivanesan@innsciteai.com](mailto:sivanesan@innsciteai.com) · [sd233@georgetown.edu](mailto:sd233@georgetown.edu)
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material for the MechBBB-ML revision Model C classifier.")

    st.markdown(
        """
        ## Purpose
        This application provides a Streamlit interface for the MechBBB-ML two-stage mechanistically augmented
        BBB permeability classifier (revision Model C). It supports single SMILES input, structure file upload (SDF, MOL, PDB, PDBQT, MOL2), and batch CSV processing.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py       # Main application (historical UI)
        ├── app.py                 # Thin entry that calls streamlit_app.main()
        ├── requirements.txt      # Dependencies
        ├── src/mechbbb/          # Frozen revision prediction module
        │   ├── predict.py        # predict_single, predict_batch, load_predictor
        │   ├── chem.py
        │   └── features.py
        ├── similarity_module.py  # Present for compatibility; AD not enabled
        ├── demo_ligands.py       # 50 illustrative demo ligands
        └── artifacts/            # Revision Model C artifacts
            ├── stage1/           # efflux.joblib, influx.joblib, pampa.joblib
            ├── stage2_modelC/    # seed0.txt … seed4.txt
            ├── calibration/      # isotonic_model_C.joblib, locked_threshold_model_C.json
            ├── threshold.json
            └── stage1_train_medians.json
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (Python 3.10 recommended).
        2. Install dependencies: `pip install -r requirements.txt`.
        3. Launch the app: `streamlit run streamlit_app.py` (or `streamlit run app.py`).
        4. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        f"""
        ## Model overview (revision Model C)
        - **Stage-1:** LightGBM models on PhysChem + ECFP4 yield uncalibrated p_efflux, p_influx, p_pampa.
        - **Stage-2:** 5-model ensemble on PhysChem + ECFP4 + Stage-1 scores → ensemble mean → isotonic → calibrated P(BBB+).
        - **Primary threshold:** {PRIMARY_THRESHOLD} (MCC-optimal on BBBP validation, calibrated scale).
        - **Secondary locked points:** {HIGH_SENS_THRESHOLD} (high sensitivity), {HIGH_SPEC_THRESHOLD} (high specificity).
        - **Features:** 10 physicochemical descriptors + 2048-bit ECFP4 + 3 Stage-1 scores = 2061 total.
        - Historical thr **0.35 is not** the revision MCC-optimal threshold.
        """
    )

    st.markdown(
        """
        ## What this build does **not** claim
        - **No ±2×SE confidence intervals** from ensemble seed variance (not a validated uncertainty method in this deployment).
        - **No applicability-domain / train_fps similarity warnings** (`train_fps.npz` is not shipped; `get_train_fps()` returns None).
        - Stage-1 scores are **uncalibrated model outputs**, not experimental transporter or PAMPA measurements.
        """
    )

    st.markdown(
        """
        ## Batch CSV outputs
        Columns include: is_valid, canonical_smiles, inchikey, prob_calibrated, prob_raw, BBB_class, threshold,
        p_efflux, p_influx, p_pampa, error. No CI columns.
        """
    )

    st.success("Questions? Contact: Dr. Sivanesan Dakshanamurthy (sd233@georgetown.edu)")


def render_mechbbb_prediction_page():
    """Render the MechBBB-ML prediction page."""
    st.title("BBB Permeability Prediction")
    st.markdown(
        """
        Predict BBB permeability using MechBBB-ML (revision Model C). Enter a SMILES string, upload a structure file (SDF, MOL, PDB, PDBQT, MOL2), or upload a CSV file for batch processing.
        The model outputs calibrated P(BBB+), raw ensemble mean, Stage-1 scores (uncalibrated), and classification at the selected threshold.
        
        **Input modes:** Single SMILES or structure file | Batch (CSV with smiles/SMILES column)
        """
    )
    st.subheader("Ligand Structure")
    ligand_preview_slot = st.empty()
    preview_img = st.session_state.get("last_ligand_image")
    preview_smiles = st.session_state.get("last_ligand_smiles")
    if preview_img:
        with ligand_preview_slot.container():
            _, preview_col, _ = st.columns([0.25, 1, 0.25])
            with preview_col:
                st.image(io.BytesIO(preview_img), width=560)
                st.caption(
                    "Latest ligand preview"
                    + (f" · SMILES: `{preview_smiles}`" if preview_smiles else "")
                )
    else:
        ligand_preview_slot.info("Ligand preview will appear here after a valid single-molecule prediction.")

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info(
            "Ensure the **artifacts/** folder contains revision Model C files:\n"
            "- artifacts/stage1/efflux.joblib, influx.joblib, pampa.joblib\n"
            "- artifacts/stage2_modelC/seed0.txt … seed4.txt\n"
            "- artifacts/calibration/isotonic_model_C.joblib\n"
            "- artifacts/threshold.json\n"
            "(Historical names such as stage1_efflux.joblib or model_seed*.pkl are not used.)"
        )
        return

    threshold = _select_threshold("pred")

    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Single SMILES or structure file", "Batch (CSV)"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Single SMILES or structure file":
        smiles_input = st.text_input(
            "SMILES (or upload a structure file below)",
            placeholder="e.g. CCO, c1ccccc1",
            key="smiles_input",
        )
        st.markdown("**Or upload a structure file:**")
        structure_file = st.file_uploader(
            "Upload structure file",
            type=["sdf", "mol", "pdb", "pdbqt", "mol2"],
            key="structure_upload",
            help="Supported: SDF, MOL, PDB, PDBQT, MOL2. First molecule will be used.",
        )
        smiles_to_use = None
        if structure_file:
            content = structure_file.read()
            ext = os.path.splitext(structure_file.name)[1]
            extracted = extract_smiles_from_file(content, ext)
            if extracted:
                smiles_to_use = extracted
                st.session_state.structure_file_content = content
                st.session_state.structure_file_ext = ext
                st.success(f"Extracted SMILES from {structure_file.name}")
            else:
                st.session_state.structure_file_content = None
                st.session_state.structure_file_ext = None
                st.error(f"Could not extract SMILES from {ext.upper()} file. Try SMILES input instead.")
        else:
            st.session_state.structure_file_content = None
            st.session_state.structure_file_ext = None
            if smiles_input and smiles_input.strip():
                smiles_to_use = smiles_input.strip()
        if st.button("Predict", type="primary", key="btn_single"):
            if smiles_to_use:
                result = predict_single(
                    smiles_to_use,
                    threshold=threshold,
                    predictor=predictor,
                )
                if result.is_valid:
                    st.success("Valid SMILES")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="result-panel">
                                <div class="result-title">Calibrated P(BBB+)</div>
                                <div class="result-value">{result.prob:.4f}</div>
                                <div class="result-subtext">Isotonic on ensemble mean</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div class="result-panel">
                                <div class="result-title">Raw ensemble mean</div>
                                <div class="result-value">{result.prob_raw:.4f}</div>
                                <div class="result-subtext">Mean of 5 Stage-2 seeds</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.markdown(
                            f"""
                            <div class="result-panel">
                                <div class="result-title">Prediction</div>
                                <div class="result-value">{result.bbb_class}</div>
                                <div class="result-subtext">Class at selected threshold</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col4:
                        st.markdown(
                            f"""
                            <div class="result-panel">
                                <div class="result-title">Threshold</div>
                                <div class="result-value">{result.threshold:.2f}</div>
                                <div class="result-subtext">Active decision cutoff</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.progress(float(result.prob), text=f"Calibrated P(BBB+): {result.prob:.1%}")
                    st.caption(
                        f"InChIKey: `{result.inchikey}` · Canonical SMILES: `{result.canonical_smiles}`"
                    )

                    st.markdown(
                        '<div class="section-heading-compact">Stage-1 scores (uncalibrated model outputs)</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "These are uncalibrated Stage-1 model scores (not experimental efflux/influx/PAMPA measurements)."
                    )
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("p_efflux", f"{result.p_efflux:.4f}")
                        st.progress(float(result.p_efflux))
                    with mcol2:
                        st.metric("p_influx", f"{result.p_influx:.4f}")
                        st.progress(float(result.p_influx))
                    with mcol3:
                        st.metric("p_pampa", f"{result.p_pampa:.4f}")
                        st.progress(float(result.p_pampa))

                    # Similarity/AD intentionally disabled (no train_fps.npz).
                    _ = get_train_fps()

                    # 2D ligand preview always from canonical SMILES; drawing failure must not affect prediction.
                    smiles_for_lookup = (result.canonical_smiles or result.smiles or "").strip()
                    img_bytes = resolve_structure_preview(smiles_for_lookup, size=400, width=560, height=440)
                    if img_bytes:
                        st.session_state.last_ligand_image = img_bytes
                        st.session_state.last_ligand_smiles = result.canonical_smiles
                        with ligand_preview_slot.container():
                            _, preview_col, _ = st.columns([0.25, 1, 0.25])
                            with preview_col:
                                st.image(io.BytesIO(img_bytes), width=560)
                                st.caption(
                                    "Latest ligand preview"
                                    + (
                                        f" · SMILES: `{result.canonical_smiles}`"
                                        if result.canonical_smiles
                                        else ""
                                    )
                                )
                    else:
                        st.info(
                            "Structure preview unavailable"
                            + (f" (SMILES: `{result.canonical_smiles}`)" if result.canonical_smiles else "")
                            + ". Prediction results above are unaffected."
                        )
                else:
                    st.error(result.error)
            else:
                st.warning("Please enter a SMILES string or upload a structure file (SDF, MOL, PDB, PDBQT, MOL2).")

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in ("smiles", "canonical_smiles", "smi") or c == "SMILES"
                ),
                None,
            )
            if col is None:
                st.error("CSV must have a SMILES column (smiles, SMILES, canonical_smiles, or smi).")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                if st.button("Predict batch", type="primary", key="btn_batch"):
                    smiles_list = df[col].astype(str).tolist()
                    results = predict_batch(
                        smiles_list,
                        threshold=threshold,
                        predictor=predictor,
                    )
                    df_out = df.copy()
                    df_out["is_valid"] = [r.is_valid for r in results]
                    df_out["canonical_smiles"] = [r.canonical_smiles for r in results]
                    df_out["inchikey"] = [r.inchikey for r in results]
                    df_out["prob_calibrated"] = [r.prob if r.is_valid else None for r in results]
                    df_out["prob_raw"] = [r.prob_raw if r.is_valid else None for r in results]
                    df_out["BBB_class"] = [r.bbb_class if r.is_valid else None for r in results]
                    df_out["threshold"] = [r.threshold for r in results]
                    df_out["p_efflux"] = [r.p_efflux for r in results]
                    df_out["p_influx"] = [r.p_influx for r in results]
                    df_out["p_pampa"] = [r.p_pampa for r in results]
                    df_out["error"] = [r.error for r in results]

                    st.subheader("Results")
                    st.dataframe(df_out, use_container_width=True)
                    st.caption(
                        "Stage-1 columns (p_efflux/p_influx/p_pampa) are uncalibrated model outputs, not experimental values. "
                        "No CI / similarity columns in this revision build."
                    )

                    st.subheader("Download results")
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False),
                        "mechbbb_predictions.csv",
                        "text/csv",
                        key="download_csv",
                    )
        else:
            st.info("Upload a CSV file with a SMILES column to run batch predictions.")

    st.divider()


def render_demo_prediction_page():
    """Render the Demo Prediction Tool page with illustrative CNS+/CNS− ligands."""
    st.title("Demo Prediction Tool")
    st.markdown(
        """
        Run predictions on **25 illustrative CNS-penetrating (CNS+)** and **25 illustrative non-CNS-penetrating (CNS−)** ligands.
        Labels are literature-reported / illustrative and **not independently verified** for this release.
        Predictions use **revision Model C** (calibrated P(BBB+) at the selected operating point).
        """
    )

    bad_demo = validate_demo_ligands()
    if bad_demo:
        names = ", ".join(n for n, _ in bad_demo)
        st.error(
            f"Demo ligand SMILES failed RDKit validation ({len(bad_demo)}): {names}. "
            "Fix demo_ligands.py before trusting structures on this page."
        )

    st.session_state.structure_file_content = None
    st.session_state.structure_file_ext = None

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info(
            "Ensure **artifacts/** contains revision Model C files "
            "(stage1/*.joblib, stage2_modelC/seed*.txt, calibration/isotonic_model_C.joblib)."
        )
        return

    threshold = _select_threshold("demo")

    st.subheader("CNS-penetrating ligands (CNS+)")
    cns_plus_labels = [f"{name}" for name, _ in CNS_PENETRATING_LIGANDS]
    cns_plus_map = {name: smi for name, smi in CNS_PENETRATING_LIGANDS}
    selected_cns_plus = st.selectbox(
        "Select an illustrative CNS-penetrating ligand",
        options=cns_plus_labels,
        key="demo_cns_plus",
    )
    smiles_cns_plus = cns_plus_map.get(selected_cns_plus, "")

    st.subheader("Non-CNS-penetrating ligands (CNS−)")
    cns_minus_labels = [f"{name}" for name, _ in NON_CNS_PENETRATING_LIGANDS]
    cns_minus_map = {name: smi for name, smi in NON_CNS_PENETRATING_LIGANDS}
    selected_cns_minus = st.selectbox(
        "Select an illustrative non-CNS-penetrating ligand",
        options=cns_minus_labels,
        key="demo_cns_minus",
    )
    smiles_cns_minus = cns_minus_map.get(selected_cns_minus, "")

    st.markdown(
        '<div class="section-heading-compact">Ligand structure (selected)</div>',
        unsafe_allow_html=True,
    )
    pv1, pv2 = st.columns(2)
    for col, title, smi in (
        (pv1, selected_cns_plus, smiles_cns_plus),
        (pv2, selected_cns_minus, smiles_cns_minus),
    ):
        with col:
            st.caption(f"{title}")
            img_sel = resolve_structure_preview(smi, size=320, width=320, height=250)
            if img_sel:
                st.image(io.BytesIO(img_sel), use_container_width=True)
            elif smi:
                st.caption("Structure preview unavailable")
            else:
                st.caption("—")

    st.divider()
    st.subheader("Run prediction")

    run_for = st.radio(
        "Predict for",
        ["CNS-penetrating ligand only", "Non-CNS-penetrating ligand only", "Both"],
        horizontal=True,
        key="demo_which",
    )

    if st.button("Predict", type="primary", key="btn_demo"):
        to_run = []
        if run_for in ("CNS-penetrating ligand only", "Both"):
            to_run.append((selected_cns_plus, smiles_cns_plus, "CNS+ (illustrative label)"))
        if run_for in ("Non-CNS-penetrating ligand only", "Both"):
            to_run.append((selected_cns_minus, smiles_cns_minus, "CNS− (illustrative label)"))

        for label, smiles, expected in to_run:
            st.markdown(f"#### {label} — {expected}")
            result = predict_single(smiles, threshold=threshold, predictor=predictor)
            if result.is_valid:
                res_left, res_right = st.columns([1.1, 1])
                with res_left:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Calibrated P(BBB+)", f"{result.prob:.4f}")
                    with col2:
                        st.metric("Raw ensemble mean", f"{result.prob_raw:.4f}")
                    with col3:
                        st.metric("Prediction", result.bbb_class)
                    with col4:
                        st.metric("Threshold", f"{result.threshold:.2f}")
                    st.caption(f"SMILES: `{smiles}`")
                    st.caption(
                        f"Stage-1 (uncalibrated): p_efflux={result.p_efflux:.4f}, "
                        f"p_influx={result.p_influx:.4f}, p_pampa={result.p_pampa:.4f}"
                    )
                smiles_for_demo = (result.canonical_smiles or result.smiles or smiles or "").strip()
                img_demo = resolve_structure_preview(smiles_for_demo, size=400)
                with res_right:
                    if img_demo:
                        st.image(
                            io.BytesIO(img_demo),
                            caption="Ligand structure (2D)",
                            use_container_width=True,
                        )
                    else:
                        st.caption("Structure preview unavailable (prediction unaffected).")
            else:
                st.error(result.error)
            st.divider()

    st.caption(
        "Demo ligands are illustrative / literature-reported labels and are not independently verified. "
        "Predictions use revision Model C. Similarity / AD warnings are disabled in this build."
    )


# ============================================================================
# MAIN - NAVIGATION
# ============================================================================

_MAIN_RENDERED = False


def main():
    """Main app entry point with navigation."""
    global _MAIN_RENDERED
    if _MAIN_RENDERED:
        return
    _MAIN_RENDERED = True

    st.set_page_config(
        page_title="MechBBB-ML - BBB Permeability Studio",
        page_icon=None,
        layout="wide",
        menu_items={
            "Report a bug": "https://github.com/your-org/mechbbb-gui/issues",
            "About": "Two-stage mechanistically augmented BBB permeability classifier (revision Model C).",
        },
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")

    if st.sidebar.button("Home", use_container_width=True, key="mechbbb_sidebar_nav_home"):
        st.session_state.current_page = "Home"

    if st.sidebar.button("Documentation", use_container_width=True, key="mechbbb_sidebar_nav_docs"):
        st.session_state.current_page = "Documentation"

    if st.sidebar.button("Demo Prediction Tool", use_container_width=True, key="mechbbb_sidebar_nav_demo"):
        st.session_state.current_page = "Demo Prediction Tool"

    if st.sidebar.button("MechBBB-ML Prediction", use_container_width=True, key="mechbbb_sidebar_nav_prediction"):
        st.session_state.current_page = "MechBBB-ML Prediction"

    st.sidebar.markdown("---")

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "MechBBB-ML Prediction":
        render_mechbbb_prediction_page()
    elif st.session_state.current_page == "Demo Prediction Tool":
        render_demo_prediction_page()


if __name__ == "__main__":
    main()
