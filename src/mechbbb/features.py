"""PhysChem10 + ECFP4 feature engineering (revision-identical order)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


PHYSCHEM_ORDER = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "HeavyAtomCount",
    "FractionCSP3", "NumAromaticRings",
]


def compute_physchem10(smiles_list):
    results = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append([np.nan] * 10)
                continue
            results.append([
                Descriptors.MolWt(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol),
                Descriptors.HeavyAtomCount(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
            ])
        except Exception:
            results.append([np.nan] * 10)
    return pd.DataFrame(results, columns=PHYSCHEM_ORDER)


def compute_morgan(smiles_list, radius=2, n_bits=2048):
    n = len(smiles_list)
    fp_array = np.zeros((n, n_bits), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            for j in range(n_bits):
                if fp.GetBit(j):
                    fp_array[i, j] = 1
        except Exception:
            pass
    return fp_array
