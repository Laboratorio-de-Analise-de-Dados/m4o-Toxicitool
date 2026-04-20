#!/usr/bin/env python3
"""
test_modularization.py  (src/)
================================
Verifica que todos os módulos do projeto podem ser importados corretamente
e que os caminhos definidos em config.py existem ou podem ser criados.

Uso (a partir da raiz do projeto):
    python src/test_modularization.py
"""

import sys
from pathlib import Path

# Este arquivo está em src/ — adiciona src/ ao path para que as importações
# relativas (from config import ..., from preprocessing import ...) funcionem.
# BUG anterior: usava Path(__file__).parent / 'src', que resolvia para src/src/.
_SRC = Path(__file__).resolve().parent   # .../ToxiciTOOL/src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_config():
    """Verifica config.py e criação de diretórios."""
    print("\n── config.py ──────────────────────────────────────")
    from config import PATHS, VIAS, Y_COLS, CONFIG_VIAS, SPLIT, PREPROC

    print(f"  VIAS    : {VIAS}")
    print(f"  SPLIT   : {SPLIT}")
    print(f"  PREPROC : {PREPROC}")
    print("  Diretórios:")
    for nome, path in PATHS.items():
        if path.suffix == "":
            status = "✓" if path.exists() else "✗"
            print(f"    {status} [{nome}] {path}")
    print("  ✓ config.py OK")


def test_preprocessing():
    """Verifica importação do módulo de pré-processamento."""
    print("\n── preprocessing ──────────────────────────────────")
    import pandas as pd
    from preprocessing.Limpeza import Limpeza

    df_test = pd.DataFrame({
        "smiles":   ["CCO", "c1ccccc1"],
        "log_dl50": [-1.0, -2.0],
        "via":      ["vi", "vi"],
    })
    limpeza = Limpeza(dataframe=df_test)
    assert hasattr(limpeza, "canonical_smiles"), "Método canonical_smiles não encontrado"
    assert hasattr(limpeza, "limpa_repetidos_inteligente"), "Método limpa_repetidos_inteligente não encontrado"
    print("  ✓ Limpeza importada e instanciada com sucesso")


def test_representation():
    """Verifica importação dos módulos de representação."""
    print("\n── representation ─────────────────────────────────")
    from representation.Representacao import Representacao
    print("  ✓ Representacao importada")

    # Pipeline GNN é opcional neste projeto (ignorar se não instalado)
    try:
        from representation.RepresentacaoGrafo import RepresentacaoGrafo, DIM_ATOM, DIM_BOND
        print(f"  ✓ RepresentacaoGrafo importada (DIM_ATOM={DIM_ATOM}, DIM_BOND={DIM_BOND})")
    except ImportError:
        print("  ⚠  RepresentacaoGrafo indisponível (PyTorch/PyG não instalado — OK para pipeline Morgan)")


def test_modeling():
    """Verifica importação dos módulos de modelagem."""
    print("\n── modeling ───────────────────────────────────────")
    from modeling.Splitter import Splitter
    print("  ✓ Splitter importado")

    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__} disponível")
    except ImportError:
        print("  ⚠  TensorFlow não instalado (pipeline Morgan indisponível)")

    try:
        import torch
        import torch_geometric
        print(f"  ✓ PyTorch {torch.__version__} + PyG {torch_geometric.__version__} disponíveis")
    except ImportError:
        print("  ⚠  PyTorch/PyG não instalado (pipeline GNN indisponível — OK)")


def test_analysis():
    """Verifica importação do módulo de análise."""
    print("\n── analysis ───────────────────────────────────────")
    from analysis.analise_modelos import calcular_scores, rankear, identificar_melhor
    print("  ✓ Módulo analysis importado")


def main():
    print("=" * 55)
    print("  ToxiciTOOL — Teste de Modularização")
    print("=" * 55)

    testes = [
        ("config",         test_config),
        ("preprocessing",  test_preprocessing),
        ("representation", test_representation),
        ("modeling",       test_modeling),
        ("analysis",       test_analysis),
    ]

    falhas = []
    for nome, func in testes:
        try:
            func()
        except Exception as e:
            print(f"  ✗ FALHOU [{nome}]: {e}")
            falhas.append(nome)

    print("\n" + "=" * 55)
    if falhas:
        print(f"  ✗ {len(falhas)} teste(s) falharam: {', '.join(falhas)}")
        sys.exit(1)
    else:
        print(f"  ✓ Todos os {len(testes)} testes passaram!")
    print("=" * 55)


if __name__ == "__main__":
    main()