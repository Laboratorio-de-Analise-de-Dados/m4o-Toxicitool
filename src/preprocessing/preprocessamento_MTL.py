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
    7. Merge outer de todas as vias por 'smiles' (Otimizado com Polars)
    8. Salva em .pkl

Configuração editável: src/config.py  (CONFIG_VIAS, PREPROC, PATHS)
"""

import gc
import numpy as np
import pandas as pd
import polars as pl

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
    # A classe Limpeza agora usa isomericSmiles=True por padrão
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
    """
    cutoff_cv  = PREPROC["cutoff_cv"]
    
    # Usando Polars para deduplicação rápida
    q = pl.from_pandas(df).lazy()
    
    # Agrupa e calcula estatísticas
    dedup = q.group_by(["smiles", "via"]).agg([
        pl.col("log_dl50").mean().alias("mean_val"),
        pl.col("log_dl50").std().alias("std_val"),
        pl.len().alias("count")
    ]).with_columns([
        (pl.col("std_val") / pl.col("mean_val").abs()).fill_nan(0.0).alias("cv")
    ])
    
    # Filtra por CV ou registros únicos
    validos = dedup.filter(
        (pl.col("count") == 1) | (pl.col("cv") <= cutoff_cv)
    ).select([
        pl.col("smiles"),
        pl.col("via"),
        pl.col("mean_val").alias("log_dl50")
    ])
    
    return validos.collect().to_pandas()


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
    print("  ETAPA 2 — Canonização global de SMILES (Isomérico)")
    print("=" * 55)
    df_all = canonizar_universo(df_all)
    print(f"  SMILES únicos após canonização: {df_all['smiles'].nunique()}")

    print("\n" + "=" * 55)
    print("  ETAPA 3 — Deduplicação por (smiles + via) [Polars]")
    print("=" * 55)
    n_antes = len(df_all)
    df_all  = deduplicar_por_via(df_all)
    print(f"  {n_antes} → {len(df_all)} registros após deduplicação")

    print("\n" + "=" * 55)
    print("  ETAPA 4 — Merge outer por via [Polars]")
    print("=" * 55)

    # Convertendo para Polars para pivotagem eficiente
    pl_all = pl.from_pandas(df_all)

    # Pivotagem: 'via' vira colunas, 'log_dl50' vira valores, agrupado por 'smiles'
    df_final_pl = pl_all.pivot(
        on="via",
        index="smiles",
        values="log_dl50"
    )

    # Renomeia as colunas para o padrão 'log_dl50_VIA'
    vias_detectadas = [c for c in df_final_pl.columns if c != "smiles"]
    df_final_pl = df_final_pl.rename({v: f"log_dl50_{v}" for v in vias_detectadas})

    # Garante que TODAS as vias configuradas existam no DataFrame final (mesmo que vazias)
    for via in CONFIG_VIAS.keys():
        col_target = f"log_dl50_{via}"
        if col_target not in df_final_pl.columns:
            df_final_pl = df_final_pl.with_columns(pl.lit(None).cast(pl.Float64).alias(col_target))

    # Reordenar colunas para consistência: smiles + vias na ordem do config
    colunas_ordenadas = ["smiles"] + [f"log_dl50_{v}" for v in CONFIG_VIAS.keys()]
    df_final_pl = df_final_pl.select(colunas_ordenadas)

    df_final = df_final_pl.to_pandas()

    print(f"\nDataFrame final: {df_final.shape[0]} mol × {df_final.shape[1]} colunas")
    print(f"NaNs por coluna:\n{df_final.isna().sum().to_string()}")

    caminho_saida = PATHS["preprocessed"] / "MTL_df_base_fundida.pkl"
    df_final.to_pickle(caminho_saida)
    print(f"\n✓ Salvo em: {caminho_saida}")

    gc.collect()
    return df_final


if __name__ == "__main__":
    df = main()
