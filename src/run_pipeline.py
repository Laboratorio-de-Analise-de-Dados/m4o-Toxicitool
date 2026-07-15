"""
run_pipeline.py  (raiz do projeto)
====================================
Ponto de entrada único para o pipeline ToxiciTOOL (pipeline Morgan).

Uso:
    # Pipeline completo
    python run_pipeline.py

<<<<<<< HEAD
    # Etapa individual
    python run_pipeline.py --etapa preprocessing
    python run_pipeline.py --etapa fingerprints
    python run_pipeline.py --etapa split
    python run_pipeline.py --etapa modelagem
    python run_pipeline.py --etapa analise
=======
    # Etapa individual ou encadeada
    python run_pipeline.py --etapa preprocessing fingerprints
    python run_pipeline.py --etapa split modelagem --metodo butina
>>>>>>> 746d913 (só mandando de um computador pra outro)
"""

import argparse
import os
import sys
from pathlib import Path

# Configuração de Ambiente (Replica modelagem_MTL.py para ativar GPU)
os.environ["TF_XLA_FLAGS"]  = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_SRC  = Path(__file__).resolve().parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS, SPLIT  # noqa: F401


def run_preprocessing():
    print("\n" + "=" * 60)
    print("  ETAPA 1 — Pré-processamento")
    print("=" * 60)
    from src.preprocessing.preprocessamento_MTL import main
    main()


def run_fingerprints():
    print("\n" + "=" * 60)
    print("  ETAPA 2 — Fingerprints Morgan")
    print("=" * 60)
    from src.representation.fingerprints_MTL import main
    main()


def run_split():
    print("\n" + "=" * 60)
    print(f"  ETAPA 2.5 — Pré-computação de Splits (Método: {SPLIT['method']})")
    print("=" * 60)
    from src.representation.split_datasets import main
    main()


def run_eda():
    print("\n" + "=" * 60)
    print("  ETAPA EXTRA — Análise Exploratória de Dados (EDA)")
    print("=" * 60)
    from src.analysis.eda import main
    main()


def run_modelagem():
    print("\n" + "=" * 60)
    print(f"  ETAPA 3 — Modelagem MTL (Morgan) (Método: {SPLIT['method']})")
    print("=" * 60)
    from src.modeling.modelagem_MTL import main
    main()


def run_analise():
    print("\n" + "=" * 60)
    print(f"  ETAPA 4 — Análise e Ranking de Modelos (Método: {SPLIT['method']})")
    print("=" * 60)
    from src.analysis.analise_modelos import main
    main()


def run_repeticoes():
    print("\n" + "=" * 60)
    print(f"  ETAPA EXTRA — Protocolo de N Treinamentos (Estatística) (Método: {SPLIT['method']})")
    print("=" * 60)
    from src.modeling.treinamento_repetido import main
    main()


def run_estatistica():
    print("\n" + "=" * 60)
    print(f"  ETAPA 5 — Protocolo Estatístico e Validação Completa (Método: {SPLIT['method']})")
    print("=" * 60)
    
    # 1. Validação Estatística
    print("\n>>> Iniciando Validação Estatística...")
    from src.analysis.validacao_estatistica import main as main_estat
    main_estat()

    # 2. Validação Externa
    csv_ext = PATHS["data_raw"].parent / "externos" / "compostos_literatura.csv"
    if csv_ext.exists():
        print("\n>>> Iniciando Validação Externa...")
        from src.analysis.validacao_externa import validar_externos
        try:
            validar_externos(csv_path=csv_ext)
        except Exception as e:
            print(f"  [Aviso] Falha na validação externa: {e}")
    else:
        print("\n[Aviso] Arquivo de validação externa não encontrado. Pulando etapa.")


def run_benchmark():
    print("\n" + "=" * 60)
    print(f"  ETAPA EXTRA — Modelos de Benchmark (Single-Task) (Método: {SPLIT['method']})")
    print("=" * 60)
    from src.modeling.benchmark import main
    main()


def run_crossval():
    print("\n" + "=" * 60)
    print(f"  ETAPA EXTRA — Cross-Validation Scaffold-Aware (Método: {SPLIT['method']})")
    print("=" * 60)
    from src.modeling.cross_validation import main
    main()


ETAPAS = {
    "preprocessing": run_preprocessing,
    "fingerprints":  run_fingerprints,
    "split":         run_split,
    "eda":           run_eda,
    "modelagem":     run_modelagem,
    "analise":       run_analise,
    "repeticoes":    run_repeticoes,
    "estatistica":   run_estatistica,
    "benchmark":     run_benchmark,
    "crossval":      run_crossval,
}

PIPELINE_COMPLETO = ["preprocessing", "fingerprints", "split", "eda", "modelagem", "analise", "repeticoes", "estatistica", "benchmark"]


def main():
    parser = argparse.ArgumentParser(description="ToxiciTOOL — Pipeline Morgan MTL")
    parser.add_argument(
        "--etapa",
        choices=list(ETAPAS.keys()) + ["all"],
        default=["all"],
        nargs="+",
        help="Uma ou mais etapas a executar (padrão: all)",
    )
    parser.add_argument(
        "--metodo",
        choices=["scaffold", "butina", "random"],
        default=None,
        help="Sobrescreve o método de split (scaffold, butina, random)",
    )
    
    args = parser.parse_args()

    # Sobrescreve o método no config.py se fornecido
    if args.metodo:
        SPLIT["method"] = args.metodo
        print(f"  [CONFIG] Método de split alterado para: {args.metodo}")

    # Resolve a lista de etapas
    if "all" in args.etapa:
        etapas = PIPELINE_COMPLETO
    else:
        etapas = args.etapa

    print("=" * 60)
    print("  ToxiciTOOL — Pipeline de Predição de Toxicidade")
    print(f"  Método: {SPLIT['method']}")
    print(f"  Etapas: {', '.join(etapas)}")
    print("=" * 60)

    for etapa in etapas:
        try:
            ETAPAS[etapa]()
        except Exception as e:
            print(f"\n  ✗ Erro na etapa '{etapa}': {e}")
            sys.exit(1)

    print("\n  ✓ Pipeline concluído!")


if __name__ == "__main__":
    main()
