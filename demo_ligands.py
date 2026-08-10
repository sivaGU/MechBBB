ligands · PY
"""
Demo ligands for the MechBBB-ML GUI Demo Prediction Tool.
25 known CNS-penetrating (BBB+) and 25 known non-CNS-penetrating (BBB-) ligands with SMILES.
All SMILES validated with RDKit and checked against the named compound's molecular weight.
References: PubChem, ChEMBL, DrugBank; BBBP/B3DB-style classifications.
"""
 
# 25 known CNS-penetrating ligands (BBB+)
CNS_PENETRATING_LIGANDS = [
    ('Caffeine', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'),
    ('Nicotine', 'CN1CCC[C@H]1c1cccnc1'),
    ('Diazepam', 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21'),
    ('Morphine', 'CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5'),
    ('Haloperidol', 'O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1'),
    ('Amitriptyline', 'CN(C)CCC=C1c2ccccc2CCc2ccccc21'),
    ('Carbamazepine', 'NC(=O)N1c2ccccc2C=Cc2ccccc21'),
    ('Fluoxetine', 'CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1'),
    ('Chlorpromazine', 'CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21'),
    ('Imipramine', 'CN(C)CCCN1c2ccccc2CCc2ccccc21'),
    ('Lorazepam', 'O=C1Nc2ccc(Cl)cc2C(c2ccccc2Cl)=NC1O'),
    ('Sertraline', 'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21'),
    ('Bupropion', 'CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1'),
    ('Diphenhydramine', 'CN(C)CCOC(c1ccccc1)c1ccccc1'),
    ('Alprazolam', 'Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2'),
    ('Clonazepam', 'O=C1CN=C(c2ccccc2Cl)c2cc([N+](=O)[O-])ccc2N1'),
    ('Valproic acid', 'CCCC(CCC)C(=O)O'),
    ('Gabapentin', 'NCC1(CC(=O)O)CCCCC1'),
    ('Levodopa (L-DOPA)', 'N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O'),
    ('Dextromethorphan', 'COc1ccc2c(c1)[C@]13CCCC[C@H]1[C@@H](C2)N(C)CC3'),
    ('Quetiapine', 'OCCOCCN1CCN(C2=Nc3ccccc3Sc3ccccc32)CC1'),
    ('Risperidone', 'Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2'),
    ('Trazodone', 'O=c1n(CCCN2CCN(c3cccc(Cl)c3)CC2)nc2ccccn12'),
    ('Zolpidem', 'Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1'),
    ('Phenobarbital', 'CCC1(c2ccccc2)C(=O)NC(=O)NC1=O'),
]
 
# 25 known non-CNS-penetrating ligands (BBB-)
NON_CNS_PENETRATING_LIGANDS = [
    ('Atenolol', 'CC(C)NCC(O)COc1ccc(CC(N)=O)cc1'),
    ('Nadolol', 'CC(C)(C)NCC(O)COc1cccc2c1C[C@H](O)[C@H](O)C2'),
    ('Lisinopril', 'NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O'),
    ('Enalapril', 'CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O'),
    ('Captopril', 'CC(CS)C(=O)N1CCCC1C(=O)O'),
    ('Ranitidine', 'CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1'),
    ('Metformin', 'CN(C)C(=N)N=C(N)N'),
    ('Fexofenadine', 'CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1'),
    ('Alendronate', 'NCCCC(O)(P(=O)(O)O)P(=O)(O)O'),
    ('Methotrexate', 'CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1'),
    ('Pravastatin', 'CC[C@H](C)C(=O)O[C@H]1C[C@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@@H]21'),
    ('Rosuvastatin', 'CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O'),
    ('Glyburide (Glibenclamide)', 'COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1'),
    ('Neostigmine', 'CN(C)C(=O)Oc1cccc([N+](C)(C)C)c1'),
    ('Sotalol', 'CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1'),
    ('Penicillin G', 'CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O'),
    ('Cefuroxime', 'CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(COC(N)=O)CS[C@H]12)c1ccco1'),
    ('Sulfasalazine', 'O=C(O)c1cc(N=Nc2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O'),
    ('Allopurinol', 'O=c1[nH]cnc2[nH]ncc12'),
    ('Probenecid', 'CCCN(CCC)S(=O)(=O)c1ccc(C(=O)O)cc1'),
    ('Cromolyn', 'O=C(O)c1cc(=O)c2cc(OCC(O)COc3cccc4c(=O)cc(C(=O)O)oc34)ccc2o1'),
    ('Loperamide', 'CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1'),
    ('Domperidone', 'O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1'),
    ('Cetirizine', 'O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1'),
    ('Succinylcholine', 'C[N+](C)(C)CCOC(=O)CCC(=O)OCC[N+](C)(C)C'),
]
 
