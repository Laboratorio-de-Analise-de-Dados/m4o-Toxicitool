"""
split_datasets.py  (src/representation/)
==========================================
Etapa 2.5 — pré-computa índices de treino/teste para todos os datasets
de fingerprints e os salva em data/splits/<método>/.

Uso:
    python run_pipeline.py --etapa split

Saída:
    data/splits/<método>/MTL_df_final_<config>__indices.npz
        → arrays 'train_idx' e 'test_idx' (int64)

Método configurável em src/config.py → SPLIT['method']:
    "scaffold"  — O(n), Murcko. Padrão recomendado.
    "butina"    — Butina clássico para n ≤ BUTINA_FULL_MAX,
                  Butina-batch (leader-follower) para n > BUTINA_FULL_MAX.
    "random"    — Baseline sem consciência estrutural.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PATHS, SPLIT
from modeling.Splitter import Splitter

# Acima deste limite, o Butina clássico (O(n²)) provavelmente OOM.
# O Butina-batch (leader-follower) é usado automaticamente.
BUTINA_FULL_MAX = 8_000

_METODO_META = {
    "scaffold": ("splits_scaffold", "Scaffold (Murcko)"),
    "butina":   ("splits_butina",   f"Butina (cutoff={SPLIT['tanimoto_cutoff']})"),
    "random":   ("splits_random",   "Random"),
}


def _path_indices(stem: str) -> Path:
    metodo   = SPLIT["method"]
    chave, _ = _METODO_META[metodo]
    return PATHS[chave] / f"{stem}__indices.npz"


def _computar_split(df: pd.DataFrame) -> tuple:
    """
    Executa o three_way_split (treino, val, teste) usando o método configurado.
    """
    splitter = Splitter(dataframe=df, smiles_col="smiles")

    return splitter.three_way_split(
        test_size=SPLIT["test_size"],
        val_size=SPLIT["val_size"],
        method=SPLIT["method"],
        tanimoto_cutoff=SPLIT.get("tanimoto_cutoff", 0.4),
        random_state=SPLIT["random_state"],
    )


def main() -> None:
    metodo = SPLIT["method"]
    if metodo not in _METODO_META:
        raise ValueError(
            f"SPLIT['method'] inválido: '{metodo}'. "
            f"Escolha: {list(_METODO_META.keys())}"
        )

    _, nome_legivel = _METODO_META[metodo]

    print("=" * 60)
    print("  ETAPA 2.5 — Pré-computação de Splits")
    print(f"  Método : {nome_legivel}")
    if metodo == "butina":
        print(f"  n ≤ {BUTINA_FULL_MAX:,} → Butina clássico (exato)")
        print(f"  n >  {BUTINA_FULL_MAX:,} → Butina-batch (leader-follower)")
    print(f"  Saída  : data/splits/{metodo}/")
    print("=" * 60)

    arquivos_pkl = sorted(PATHS["preprocessed"].glob("MTL_df_final_*.pkl"))
    if not arquivos_pkl:
        print("  Nenhum arquivo MTL_df_final_*.pkl encontrado.")
        print("  Execute antes: python run_pipeline.py --etapa fingerprints")
        return

    pulados    = 0
    calculados = 0

    for arquivo in arquivos_pkl:
        stem        = arquivo.stem
        caminho_npz = _path_indices(stem)

        # Checa se o arquivo existe e se contém o val_idx (migração 2-way -> 3-way)
        existe_completo = False
        if caminho_npz.exists():
            try:
                data = np.load(caminho_npz)
                if "val_idx" in data:
                    existe_completo = True
            except Exception:
                pass

        if existe_completo:
            print(f"  ↩  Já existe (3-way), pulando: {caminho_npz.name}")
            pulados += 1
            continue

        print(f"\n  ── {arquivo.name}")
        df = pd.read_pickle(arquivo)

        if "smiles" not in df.columns:
            print(f"  ✗  Coluna 'smiles' não encontrada — pulando")
            continue

        train_idx, val_idx, test_idx = _computar_split(df)

        np.savez(caminho_npz, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        print(f"  ✓  Salvo: {caminho_npz.name}")
        calculados += 1

    print("\n" + "=" * 60)
    print(f"  ✓ {calculados} calculado(s) | {pulados} já existiam")
    print("=" * 60)


if __name__ == "__main__":
    main()