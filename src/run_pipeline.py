"""
run_pipeline.py  (raiz do projeto)
====================================
Ponto de entrada único para o pipeline ToxiciTOOL (pipeline Morgan).

Uso:
    # Pipeline completo
    python run_pipeline.py

    # Etapa individual
    python run_pipeline.py --etapa preprocessing
    python run_pipeline.py --etapa fingerprints
    python run_pipeline.py --etapa split
    python run_pipeline.py --etapa modelagem
    python run_pipeline.py --etapa analise

    # Via Docker
    docker-compose --profile pipeline up pipeline
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PATHS  # noqa: F401 — cria todos os diretórios como side-effect


def run_preprocessing():
    print("\n" + "=" * 60)
    print("  ETAPA 1 — Pré-processamento")
    print("=" * 60)
    from preprocessing.preprocessamento_MTL import main
    main()


def run_fingerprints():
    print("\n" + "=" * 60)
    print("  ETAPA 2 — Fingerprints Morgan")
    print("=" * 60)
    from representation.fingerprints_MTL import main
    main()


def run_split():
    print("\n" + "=" * 60)
    print("  ETAPA 2.5 — Pré-computação de Splits")
    print("=" * 60)
    from representation.split_datasets import main
    main()


def run_modelagem():
    print("\n" + "=" * 60)
    print("  ETAPA 3 — Modelagem MTL (Morgan)")
    print("=" * 60)
    from modeling.modelagem_MTL import main
    main()


def run_analise():
    print("\n" + "=" * 60)
    print("  ETAPA 4 — Análise e Ranking de Modelos")
    print("=" * 60)
    from analysis.analise_modelos import main
    main()


ETAPAS = {
    "preprocessing": run_preprocessing,
    "fingerprints":  run_fingerprints,
    "split":         run_split,
    "modelagem":     run_modelagem,
    "analise":       run_analise,
}

PIPELINE_COMPLETO = ["preprocessing", "fingerprints", "split", "modelagem", "analise"]


def main():
    parser = argparse.ArgumentParser(description="ToxiciTOOL — Pipeline Morgan MTL")
    parser.add_argument(
        "--etapa",
        choices=list(ETAPAS.keys()) + ["all"],
        default="all",
        help="Etapa a executar (padrão: all)",
    )
    args   = parser.parse_args()
    etapas = PIPELINE_COMPLETO if args.etapa == "all" else [args.etapa]

    print("=" * 60)
    print("  ToxiciTOOL — Pipeline de Predição de Toxicidade")
    print(f"  Etapas: {', '.join(etapas)}")
    print("=" * 60)

    for etapa in etapas:
        try:
            ETAPAS[etapa]()
        except Exception as e:
            print(f"\n  ✗ Erro na etapa '{etapa}': {e}")
            raise

    print("\n  ✓ Pipeline concluído!")


if __name__ == "__main__":
    main()
