"""
config.py  (src/)
==================
Configuração centralizada de caminhos e hiperparâmetros do projeto ToxiciTOOL.
"""

from pathlib import Path

_ESTE_ARQUIVO = Path(__file__).resolve()
RAIZ          = _ESTE_ARQUIVO.parent.parent

# ==========================================
# CAMINHOS PRINCIPAIS
# ==========================================
PATHS = {
    "data_raw":        RAIZ / "data" / "data_raw",
    "preprocessed":    RAIZ / "data" / "preprocessed",
    "splits_scaffold": RAIZ / "data" / "splits" / "scaffold",
    "splits_butina":   RAIZ / "data" / "splits" / "butina",
    "splits_random":   RAIZ / "data" / "splits" / "random",
    "tabelas":         RAIZ / "results" / "tabelas" / "multitask",
    "plots":           RAIZ / "results" / "plots" / "multitask",
    "plots_morgan":    RAIZ / "results" / "plots" / "multitask" / "png" / "regressao" / "morgan",
    "plots_analise":   RAIZ / "results" / "plots" / "multitask" / "analise",
    "modelos_morgan":  RAIZ / "results" / "modelos_salvos" / "morgan",
    "melhor_modelo":   RAIZ / "results" / "melhor_modelo",
    "log_excel":       RAIZ / "results" / "tabelas" / "multitask" / "MTL_Log_Modelos.xlsx",
}

for _path in PATHS.values():
    if _path.suffix == "":
        _path.mkdir(parents=True, exist_ok=True)

# ==========================================
# CONFIGURAÇÃO DAS VIAS
# ==========================================
VIAS   = ["mouse_vi", "mouse_vo", "mouse_ip", "rat_vi", "rat_vo", "rat_ip"]
Y_COLS = {via: f"log_dl50_{via}" for via in VIAS}

# ==========================================
# CONFIGURAÇÃO DAS FONTES DE DADOS BRUTAS
# ==========================================
CONFIG_VIAS = {
    "mouse_vi": {"arquivo": "Acute Toxicity_mouse_intravenous_LD50.csv",     "col_smiles": "Canonical SMILES", "col_ld50": "Toxicity Value"},
    "mouse_vo": {"arquivo": "Acute Toxicity_mouse_oral_LD50.csv",            "col_smiles": "Canonical SMILES", "col_ld50": "Toxicity Value"},
    "mouse_ip": {"arquivo": "Acute Toxicity_mouse_intraperitoneal_LD50.csv", "col_smiles": "Canonical SMILES", "col_ld50": "Toxicity Value"},
    "rat_vi":   {"arquivo": "Acute Toxicity_rat_intravenous_LD50.csv",       "col_smiles": "Canonical SMILES", "col_ld50": "Toxicity Value"},
    "rat_vo":   {"arquivo": "Acute Toxicity_rat_oral_LD50.csv",              "col_smiles": "Canonical SMILES", "col_ld50": "Toxicity Value"},
    "rat_ip":   {"arquivo": "Acute Toxicity_rat_intraperitoneal_LD50.csv",   "col_smiles": "Canonical SMILES", "col_ld50": "Toxicity Value"},
}

# ==========================================
# HIPERPARÂMETROS DE SPLIT
# ==========================================
SPLIT = {
    # "scaffold" | "butina" | "random"
    "method":          "butina",
    "tanimoto_cutoff": 0.4,
    "test_size":       0.2,
    "random_state":    42,
}

# ==========================================
# DESCRITORES FÍSICO-QUÍMICOS
# ==========================================
DESCRITORES = {
    # False → apenas fingerprints (comportamento legado)
    # True  → fingerprints + descritores escalonados concatenados
    "ativo": True,

    # Descritores RDKit calculados via Chem.Descriptors.
    # Escolhidos por relevância ADME/toxicocinética para LD50.
    # Relevância por grupo:
    #   Lipofilicidade  → MolLogP                 (distribuição tecidual)
    #   Tamanho/Volume  → MolWt, LabuteASA, TPSA  (absorção, penetração)
    #   Flexibilidade   → NumRotatableBonds, FractionCSP3
    #   Grupos polares  → NumHDonors, NumHAcceptors
    #   Aromaticidade   → RingCount, NumAromaticRings
    "lista": [
        "MolLogP",
        "MolWt",
        "TPSA",
        "LabuteASA",
        "NumRotatableBonds",
        "NumHDonors",
        "NumHAcceptors",
        "FractionCSP3",
        "RingCount",
        "NumAromaticRings",
    ],
}

# ==========================================
# HIPERPARÂMETROS DE PRÉ-PROCESSAMENTO
# ==========================================
PREPROC = {
    "log_cutoff": -0.5,
    "cutoff_cv":   0.2,
}