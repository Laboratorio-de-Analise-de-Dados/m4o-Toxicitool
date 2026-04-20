"""
fingerprints_MTL.py  (src/representation/)
==============================================
Etapa 2 — gera DataFrames com fingerprints Morgan + descritores físico-químicos.

Grid de fingerprints (18 combinações = 2 × 3 × 3):
    quantitativo : [False, True]       → binário vs. contagem
    tamanho_fp   : [2048, 4096, 8192]
    raio         : [2, 3, 5]

Se DESCRITORES['ativo'] = True em config.py, os ~10 descritores físico-químicos
são calculados UMA vez na base fundida e concatenados a cada variante de FP.
O vetor de entrada final fica com shape = (fpSize + n_descritores,).

Nomenclatura dos arquivos:
    DESCRITORES ativo  → MTL_df_final_{tipo}_{bits}bits_raio{r}__desc.pkl
    DESCRITORES inativo → MTL_df_final_{tipo}_{bits}bits_raio{r}.pkl
"""

import gc
import itertools

import pandas as pd

from config import DESCRITORES, PATHS
from representation.Representacao import Representacao

QUANTITATIVO = [False, True]
TAMANHO_FP   = [2048, 4096, 8192]
RAIO         = [2, 3, 5]


def _sufixo_arquivo() -> str:
    """Retorna '__desc' quando descritores ativos, '' caso contrário."""
    return "__desc" if DESCRITORES["ativo"] else ""


def cria_dataframe(
    dataframe_com_romol: pd.DataFrame,
    radius: int = 2,
    fpSize: int = 2048,
    use_count: bool = False,
    col_smiles: str = "smiles",
) -> pd.DataFrame:
    """
    Recebe DataFrame com coluna 'ROMol', gera fingerprints Morgan,
    opcionalmente concatena descritores físico-químicos escalonados,
    e retorna df sem as colunas pesadas (ROMol, Fingerprint).
    """
    rep    = Representacao(dataframe=dataframe_com_romol)
    df_fps = rep.fp_Morgan(col_frames="ROMol", radius=radius, fpSize=fpSize, use_count=use_count)
    df_fps = rep.bitVect_to_array("Fingerprint")

    if DESCRITORES["ativo"]:
        df_fps = rep.calcular_descritores(
            col_smiles=col_smiles,
            lista_descritores=DESCRITORES["lista"],
        )
        df_fps = rep.concatenar_descritores()

    df_fps = df_fps.drop(columns=["ROMol", "Fingerprint"], errors="ignore")
    gc.collect()
    return df_fps


def main() -> None:
    caminho_base = PATHS["preprocessed"] / "MTL_df_base_fundida.pkl"
    print(f"Carregando base: {caminho_base}")
    df_base = pd.read_pickle(caminho_base)
    print(f"Base carregada: {df_base.shape[0]} moléculas")

    sufixo = _sufixo_arquivo()
    if DESCRITORES["ativo"]:
        print(f"\nModo: Fingerprints + {len(DESCRITORES['lista'])} descritores físico-químicos")
    else:
        print("\nModo: Fingerprints apenas (descritores desativados)")

    # Gera objetos ROMol UMA vez — evita reconversão para cada combinação
    print("\nGerando objetos RDKit (mol_to_frame)...")
    prep    = Representacao(dataframe=df_base)
    df_base = prep.mol_to_frame(col_smiles="smiles")
    df_base.dropna(subset=["ROMol"], inplace=True)
    df_base.reset_index(drop=True, inplace=True)
    print(f"Moléculas com ROMol válido: {len(df_base)}")

    total    = len(QUANTITATIVO) * len(TAMANHO_FP) * len(RAIO)
    contador = 0

    for c, t, r in itertools.product(QUANTITATIVO, TAMANHO_FP, RAIO):
        contador += 1
        tipo          = "quantitativo" if c else "binario"
        nome_arquivo  = f"MTL_df_final_{tipo}_{t}bits_raio{r}{sufixo}.pkl"
        caminho_saida = PATHS["preprocessed"] / nome_arquivo

        if caminho_saida.exists():
            print(f"[{contador}/{total}] Já existe, pulando: {nome_arquivo}")
            continue

        print(f"\n[{contador}/{total}] Gerando: {nome_arquivo}")
        print(f"  → raio={r} | fpSize={t} | use_count={c} | desc={DESCRITORES['ativo']}")

        df_resultado = cria_dataframe(df_base.copy(), radius=r, fpSize=t, use_count=c)
        df_resultado.to_pickle(caminho_saida)
        print(f"  ✓ Salvo: {caminho_saida} | shape={df_resultado.shape}")
        gc.collect()

    print(f"\n✓ {total} arquivos gerados em: {PATHS['preprocessed']}")


if __name__ == "__main__":
    main()