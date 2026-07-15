"""
diagnostico_quimico.py  (src/analysis/)
==========================================
Gera uma descrição estatística e química completa do banco de dados.
Utiliza Polars para performance e RDKit para detecção de grupos funcionais.

Saída:
    results/tabelas/multitask/Diagnostico_Quimico_Base.csv
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments

# Adiciona a RAIZ ao path
_SRC  = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS, VIAS, Y_COLS

# Lista de funções de fragmentos do RDKit (mapeamento amigável corrigido)
FRAG_FUNCS = {
    "Alcoois_Alifaticos": Fragments.fr_Al_OH,
    "Fenois": Fragments.fr_phenol,
    "Aldeidos": Fragments.fr_aldehyde,
    "Amidas": Fragments.fr_amide,
    "Aminas_Primarias": Fragments.fr_NH2,
    "Anilinas": Fragments.fr_aniline,
    "Acidos_Carboxilicos": Fragments.fr_COO,
    "Esteres": Fragments.fr_ester,
    "Cetonas": Fragments.fr_ketone,
    "Nitrilas": Fragments.fr_nitrile,
    "Eteres": Fragments.fr_ether,
    "Halogenios": Fragments.fr_halogen,
    "Nitro_Grupos": Fragments.fr_nitro,
    "Sulfonamidas": Fragments.fr_sulfonamd,
    "Tiols": Fragments.fr_SH,
    "Aneis_Aromaticos": Descriptors.NumAromaticRings,
    "Aneis_Alifaticos": Descriptors.NumAliphaticRings,
}

def calcular_propriedades_quimicas(smiles):
    """Calcula propriedades e fragmentos para uma molécula."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        
        props = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        }
        
        # Detecção de grupos funcionais
        for nome, func in FRAG_FUNCS.items():
            props[f"Grupo_{nome}"] = func(mol)
            
        return props
    except:
        return None

def main():
    caminho_base = PATHS["preprocessed"] / "MTL_df_base_fundida.pkl"
    if not caminho_base.exists():
        print(f"  ✗ Erro: Base '{caminho_base}' não encontrada.")
        return

    print(f"\n{'='*60}\n  DIAGNÓSTICO QUÍMICO E ESTATÍSTICO (Polars + RDKit)\n{'='*60}")
    
    # 1. Carregar via Pandas (devido ao formato .pkl) e converter para Polars
    print("  1. Carregando base de dados...")
    df_pd = pd.read_pickle(caminho_base)
    df = pl.from_pandas(df_pd)
    
    # 2. Análise Química com RDKit
    print("  2. Calculando propriedades químicas e grupos funcionais (RDKit)...")
    smiles_list = df["smiles"].to_list()
    
    quimica_data = []
    for s in smiles_list:
        res = calcular_propriedades_quimicas(s)
        if res:
            quimica_data.append(res)
    
    df_quimica = pl.from_dicts(quimica_data)
    
    # 3. Estatísticas Descritivas de Toxicidade (Polars)
    print("  3. Gerando estatísticas descritivas de toxicidade...")
    colunas_y = [Y_COLS[via] for via in VIAS if Y_COLS[via] in df.columns]
    stats_tox = df.select(colunas_y).describe()
    
    # 4. Estatísticas de Propriedades Químicas
    print("  4. Consolidando perfis químicos (Médias, Medianas, Desvios)...")
    stats_quim = df_quimica.describe()
    
    # 5. Frequência de Grupos Químicos
    print("  5. Analisando prevalência de grupos funcionais...")
    colunas_grupos = [c for c in df_quimica.columns if c.startswith("Grupo_")]
    prevalencia = []
    total_mol = len(df_quimica)
    
    for col in colunas_grupos:
        count = (df_quimica[col] > 0).sum()
        prevalencia.append({
            "Grupo": col.replace("Grupo_", ""),
            "N_Compostos": count,
            "Percentual": (count / total_mol) * 100
        })
    df_prev = pl.from_dicts(prevalencia).sort("N_Compostos", descending=True)

    # 6. Exportação consolidada para CSV
    print("  6. Exportando diagnóstico para CSV...")
    
    report_rows = []
    
    # Seção A: Geral
    report_rows.append({"Secao": "GERAL", "Variavel": "Total de Moléculas", "Metrica": "Count", "Valor": str(len(df))})
    report_rows.append({"Secao": "GERAL", "Variavel": "Moléculas Válidas (RDKit)", "Metrica": "Count", "Valor": str(total_mol)})

    # Seção B: Toxicidade
    for row in stats_tox.to_dicts():
        metrica = row.pop("statistic")
        for var, val in row.items():
            if val is not None:
                # Trata val como float antes de formatar
                try:
                    f_val = float(val)
                    report_rows.append({"Secao": "TOXICIDADE", "Variavel": var, "Metrica": metrica, "Valor": f"{f_val:.4f}"})
                except:
                    report_rows.append({"Secao": "TOXICIDADE", "Variavel": var, "Metrica": metrica, "Valor": str(val)})

    # Seção C: Propriedades Físico-Químicas
    for row in stats_quim.to_dicts():
        metrica = row.pop("statistic")
        for var, val in row.items():
            if not var.startswith("Grupo_") and val is not None:
                try:
                    f_val = float(val)
                    report_rows.append({"Secao": "PROPRIEDADES", "Variavel": var, "Metrica": metrica, "Valor": f"{f_val:.4f}"})
                except:
                    report_rows.append({"Secao": "PROPRIEDADES", "Variavel": var, "Metrica": metrica, "Valor": str(val)})

    # Seção D: Grupos Funcionais
    for row in df_prev.to_dicts():
        report_rows.append({
            "Secao": "GRUPOS_QUIMICOS", 
            "Variavel": row["Grupo"], 
            "Metrica": "Frequencia", 
            "Valor": f"{row['N_Compostos']} ({row['Percentual']:.2f}%)"
        })

    df_final = pl.from_dicts(report_rows)
    output_path = PATHS["tabelas"] / "Diagnostico_Quimico_Base.csv"
    df_final.write_csv(output_path)
    
    print(f"\n  ✓ Diagnóstico concluído e salvo em:\n    {output_path}")

if __name__ == "__main__":
    main()
