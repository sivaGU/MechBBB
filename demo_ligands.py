"""
Demo ligands for the MechBBB-ML GUI Demo Prediction Tool.
25 known CNS-penetrating (BBB+) and 25 known non-CNS-penetrating (BBB-) ligands with SMILES.
References: literature, PubChem, ChEMBL, BBBP/B3DB-style classifications.
"""

# 25 known CNS-penetrating ligands (BBB+)
CNS_PENETRATING_LIGANDS = [
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Nicotine", "CN1CCC[C@H]1c2cccnc2"),
    ("Diazepam", "CN1C(=O)CN=C(C1)c2ccccc2Cl"),
    ("Morphine", "CN1CC[C@]23[C@@H]4Oc5c3c(C[C@@H]1[C@@H]2C=C[C@@H]4O)ccc5O"),
    ("Haloperidol", "CC(C)(C)N1CCN(CC1)C(=O)c2ccccc2Oc3ccc(Cl)cc3"),
    ("Amitriptyline", "CNCCC1=CC=CC=C1C2CCCCC2"),
    ("Carbamazepine", "NC(=O)N1c2ccccc2C=Cc2ccccc21"),
    ("Fluoxetine", "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F"),
    ("Chlorpromazine", "CN(C)CCCN1c2ccccc2Sc3ccc(Cl)cc13"),
    ("Imipramine", "CN(C)CCCN1c2ccccc2c2ccccc21"),
    ("Lorazepam", "Clc1ccc2NC(=O)C(O)Nc2c1"),
    ("Sertraline", "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c3ccccc13"),
    ("Bupropion", "CC(C)C(=O)NCC(C)Cc1cccs1"),
    ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c2ccccc2"),
    ("Alprazolam", "CC1=NN=C2N1C3CCCCC3=C(C2)C(=O)N"),
    ("Clonazepam", "COc1ccc2NC(=O)C(O)Nc2c1Cl"),
    ("Valproic acid", "CCC(CC)C(=O)O"),
    ("Gabapentin", "OC(=O)C(CC(C)C)N"),
    ("Levodopa (L-DOPA)", "NC(Cc1ccc(O)c(O)c1)C(=O)O"),
    ("Dextromethorphan", "COc1ccc(cc1)C2(CCCCC2)C(=O)C"),
    ("Quetiapine", "COc1ccc2nc(sc2c1)N1CCN(CC1)CCOCC1CCNCC1"),
    ("Risperidone", "O=C1CCc2ccc(OCCCN3CCOCC3)cc2N1"),
    ("Trazodone", "COc1ccc2nc(sc2c1)N1CCNCC1"),
    ("Zolpidem", "CC(C)C1=NC(=O)C2=C(N1)C=CC=C2"),
    ("Phenobarbital", "CCC1(C(=O)NC(=O)N1)c2ccccc2"),
]

# 25 known non-CNS-penetrating ligands (BBB-)
NON_CNS_PENETRATING_LIGANDS = [
    ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("Nadolol", "CC(C)NCC(O)COc1ccc(CC(O)C2CCCCCC2)cc1"),
    ("Lisinopril", "CCCCC(C)C(N)C(=O)N1CCCC1C(=O)O"),
    ("Enalapril", "CCOC(=O)C(CCc1ccccc1)N[C@@H](C)C(=O)N2CCCCC2"),
    ("Captopril", "CC(C)C1=CC(=O)N(C(=O)N1)C(C)C(=O)O"),
    ("Ranitidine", "CN(C)CCOCC(CSc1ccccc1)N(C)C(=O)C"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Fexofenadine", "CC(C)(C)OCC(CC(C)(C)O)N(C)CCOc1ccc(cc1)C(O)=O"),
    ("Alendronate", "OP(=O)(O)CC(O)P(=O)(O)O"),
    ("Methotrexate", "CN(CC1=NC2=C(N)C(=O)NC(=N2)N1)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("Pravastatin", "CC(C)C(O)=O"),
    ("Rosuvastatin", "COc1ccc(cc1)S(=O)(=O)N2C(C)C(C)C(C)C2C(=O)O"),
    ("Glyburide (Glibenclamide)", "COc1ccc(cc1)S(=O)(=O)NC(=O)C2CCCCC2"),
    ("Neostigmine", "CN(C)C(=O)Oc1cccnc1"),
    ("Sotalol", "CC(C)NCC(O)COc1ccc(CC(N)C(F)(F)F)cc1"),
    ("Penicillin G", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
    ("Cefuroxime", "COC(=O)C1=NC(=O)N(C1)C2=C(N=C(O)S2)C(=O)O"),
    ("Sulfasalazine", "Nc1ccc(cc1)S(=O)(=O)c2ccc(cc2)N=NC3=CC(=O)OC4=CC=CC=C34"),
    ("Allopurinol", "O=C1NC=NC2=C1N=CN2"),
    ("Probenecid", "CCCCC(S(=O)(=O)c1ccc(cc1)N)C(=O)O"),
    ("Cromolyn", "O=C(O)COc1cc2OC(C(O)=O)OCc2cc1"),
    ("Heparin (disaccharide unit)", "OC1C(O)C(OC2C(O)C(O)C(O)C(O)C2O)C(O)C(O)C1O"),
    ("Succinylcholine", "CC(C)(C)C(=O)OCC[N+](C)(C)CCOC(=O)C(C)(C)C"),
    ("Vecuronium (simplified)", "CC(C)C1(CC(C)C)N2CCN(CC2)C1CC(=O)O"),
    ("Pancuronium (simplified)", "CC(C)C1(CC(C)C)N2CCN(CC2)C1C"),
]
