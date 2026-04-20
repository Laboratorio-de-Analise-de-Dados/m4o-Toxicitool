"""
preprocessamento_MTL.py  (src/preprocessing/)
=================================================
Pipeline de pré-processamento para Multi-Task Learning (MTL).

Fluxo:
    1. Lê os CSVs brutos de cada via
    2. Padroniza colunas → ['smiles', 'ld50', 'via']
    3. Converte LD50 para escala logarítmica (log10)
    4. Concatena TODAS as vias em um único DataFrame
    5. Canonização UMA única vez no universo completo de SMILES
    6. Deduplicação inteligente por (smiles + via) com |CV| <= cutoff
    7. Merge outer de todas as vias por 'smiles'
    8. Salva em .pkl

Configuração editável: src/config.py  (CONFIG_VIAS, PREPROC, PATHS)
"""

import gc

import numpy as np
import pandas as pd

from config import CONFIG_VIAS, PATHS, PREPROC
from preprocessing.Limpeza import Limpeza


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================


def carregar_via(cfg: dict, nome_via: str) -> pd.DataFrame:
    """Lê o arquivo bruto e padroniza para ['smiles', 'ld50', 'via']."""
    caminho = PATHS["data_raw"] / cfg["arquivo"]

    if caminho.suffix == ".csv":
        df = pd.read_csv(caminho)
    elif caminho.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(caminho)
    else:
        raise ValueError(f"Formato não suportado: {caminho.suffix}")

    df = df.rename(columns={cfg["col_smiles"]: "smiles", cfg["col_ld50"]: "ld50"})
    df = df[["smiles", "ld50"]].copy()
    df["via"] = nome_via
    return df


def aplicar_log(df: pd.DataFrame) -> pd.DataFrame:
    """Converte LD50 (mg/kg) para log10. Valores <= 0 viram NaN."""
    df = df.copy()
    df["ld50"] = pd.to_numeric(df["ld50"], errors="coerce")
    df["log_dl50"] = np.where(df["ld50"] > 0, np.log10(df["ld50"]), np.nan)
    df.drop(columns=["ld50"], inplace=True)
    return df


def canonizar_universo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canoniza os SMILES de TODO o universo de uma vez.
    """
    print("\nCanonizando universo completo de SMILES (única vez)...")
    n_antes = len(df)

    smiles_unicos = pd.DataFrame({"smiles": df["smiles"].dropna().unique()})
    limpeza_temp  = Limpeza(dataframe=smiles_unicos)
    df_canonico   = limpeza_temp.canonical_smiles(col_smiles="smiles", sanitize=True)

    mapa = dict(zip(smiles_unicos["smiles"], df_canonico["smiles"]))
    df   = df.copy()
    df["smiles"] = df["smiles"].map(mapa)
    df.dropna(subset=["smiles"], inplace=True)

    print(f"  {n_antes} → {len(df)} registros ({n_antes - len(df)} inválidos removidos)")
    return df


def deduplicar_por_via(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicação inteligente por (smiles + via).

    Regras:
        - 1 registro      → mantém como está
        - |CV| <= cutoff  → dados consistentes → retorna a média
        - |CV| >  cutoff  → dados inconsistentes → descarta o grupo
    """
    cutoff_cv  = PREPROC["cutoff_cv"]
    resultados = []

    for (smiles, via), grupo in df.groupby(["smiles", "via"]):
        if len(grupo) == 1:
            resultados.append(grupo)
            continue

        valores = pd.to_numeric(grupo["log_dl50"], errors="coerce").dropna()
        if len(valores) == 0:
            continue

        media  = float(valores.mean())
        desvio = float(valores.std())
        cv     = 0.0 if media == 0 else abs(desvio / media)

        if cv <= cutoff_cv:
            linha             = grupo.iloc[0:1].copy()
            linha["log_dl50"] = media
            resultados.append(linha)
        # cv > cutoff → grupo descartado (labs inconsistentes)

    if not resultados:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(resultados, ignore_index=True)


# ==========================================
# PIPELINE PRINCIPAL
# ==========================================


def main() -> pd.DataFrame:

    print("=" * 55)
    print("  ETAPA 1 — Carregando dados brutos")
    print("=" * 55)
    frames = []
    for nome_via, cfg in CONFIG_VIAS.items():
        df_raw  = carregar_via(cfg, nome_via)
        df_log  = aplicar_log(df_raw)
        n_antes = len(df_log)
        df_log  = df_log[df_log["log_dl50"] >= PREPROC["log_cutoff"]].dropna(
            subset=["log_dl50"]
        )
        print(f"  [{nome_via}] {n_antes} → {len(df_log)} (após cutoff log)")
        frames.append(df_log)

    df_all = pd.concat(frames, ignore_index=True)
    print(
        f"\nTotal: {len(df_all)} registros | {df_all['via'].nunique()} vias | "
        f"{df_all['smiles'].nunique()} SMILES únicos"
    )

    print("\n" + "=" * 55)
    print("  ETAPA 2 — Canonização global de SMILES")
    print("=" * 55)
    df_all = canonizar_universo(df_all)
    print(f"  SMILES únicos após canonização: {df_all['smiles'].nunique()}")

    print("\n" + "=" * 55)
    print("  ETAPA 3 — Deduplicação por (smiles + via)")
    print("=" * 55)
    n_antes = len(df_all)
    df_all  = deduplicar_por_via(df_all)
    print(f"  {n_antes} → {len(df_all)} registros após deduplicação")

    print("\n" + "=" * 55)
    print("  ETAPA 4 — Merge outer por via")
    print("=" * 55)
    dfs_por_via = []
    for nome_via in CONFIG_VIAS.keys():
        df_via = (
            df_all[df_all["via"] == nome_via][["smiles", "log_dl50"]]
            .rename(columns={"log_dl50": f"log_dl50_{nome_via}"})
            .copy()
        )
        print(f"  [{nome_via}] {len(df_via)} moléculas")
        dfs_por_via.append(df_via)

    df_final = dfs_por_via[0]
    for df_via in dfs_por_via[1:]:
        df_final = df_final.merge(df_via, on="smiles", how="outer")

    print(f"\nDataFrame final: {df_final.shape[0]} mol × {df_final.shape[1]} colunas")
    print(f"NaNs por coluna:\n{df_final.isna().sum().to_string()}")

    caminho_saida = PATHS["preprocessed"] / "MTL_df_base_fundida.pkl"
    df_final.to_pickle(caminho_saida)
    print(f"\n✓ Salvo em: {caminho_saida}")

    gc.collect()
    return df_final


if __name__ == "__main__":
    df = main()