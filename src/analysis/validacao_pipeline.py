"""
validacao_pipeline.py  (src/analysis/)
=========================================
Orquestrador principal das etapas 5-A e 5-B.

Responsabilidades:
    1. Identificar o modelo campeão de cada split (scaffold, butina, random)
       a partir dos arquivos MTL_Ranking_<split>_Modelos.xlsx
    2. Carregar modelo .keras, dados .pkl e índices .npz correspondentes
    3. Executar BenchmarkSuite (5-A) e SuiteEstatistica (5-B)
    4. Salvar tudo organizado em:
         results/tabelas/multitask/validacao/<split>/
         results/plots/multitask/validacao/<split>/

Uso standalone:
    python src/analysis/validacao_pipeline.py

Uso programático:
    from src.analysis.validacao_pipeline import ValidacaoCompleta
    val = ValidacaoCompleta("scaffold")
    val.executar_tudo()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"]         = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"]        = "0"

_SRC  = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np
import pandas as pd

from src.config import PATHS, RAIZ, VIAS, Y_COLS
from src.analysis.testes_benchmark   import BenchmarkSuite
from src.analysis.testes_estatisticos import SuiteEstatistica


# ══════════════════════════════════════════════════════════════
# CARREGADOR DE MODELO E DADOS
# ══════════════════════════════════════════════════════════════

class CarregadorDados:
    """
    Carrega o modelo .keras e os dados associados (.pkl + .npz)
    para um determinado campeão de split.

    Parâmetros
    ----------
    caminho_modelo : Path
        Caminho completo para o arquivo .keras do modelo campeão.
    """

    def __init__(self, caminho_modelo: Path) -> None:
        self.caminho_modelo = Path(caminho_modelo)
        self._validar_existencia()

    def _validar_existencia(self) -> None:
        if not self.caminho_modelo.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {self.caminho_modelo}"
            )

    def _derivar_caminhos(self) -> tuple[Path, Path]:
        """
        A partir do nome do .keras, deriva os caminhos do .pkl e do .npz.

        Convenção de nomes:
            <stem_pkl>__<metodo_split>.keras
            → data/preprocessed/<stem_pkl>.pkl
            → data/splits/<metodo_split>/<stem_pkl>__indices.npz
        """
        stem             = self.caminho_modelo.stem          # ex: MTL_df_final_binario_2048bits_raio2__desc__scaffold
        pkl_stem, metodo = stem.rsplit("__", 1)              # ex: ...__desc  +  scaffold

        pkl_path = PATHS["preprocessed"] / f"{pkl_stem}.pkl"
        npz_dir  = PATHS.get(f"splits_{metodo}", RAIZ / "data" / "splits" / metodo)
        npz_path = npz_dir / f"{pkl_stem}__indices.npz"

        return pkl_path, npz_path

    def carregar(self, incluir_scrambling: bool = False) -> dict:
        """
        Carrega e retorna o dicionário de dados no formato esperado
        por BenchmarkSuite e SuiteEstatistica.

        Retorna
        -------
        dict com chaves:
            model, X_train, X_test, y_train, y_test, predicoes, vias_disp
        """
        import tensorflow as tf
        from src.modeling.modelagem_MTL import CamadaIncerteza, get_huber_uncertainty_loss

        pkl_path, npz_path = self._derivar_caminhos()

        # Validações
        if not pkl_path.exists():
            raise FileNotFoundError(f"Dados pré-processados não encontrados: {pkl_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"Índices de split não encontrados: {npz_path}")

        print(f"  [Carregador] Lendo dados: {pkl_path.name}")
        df = pd.read_pickle(pkl_path)

        print(f"  [Carregador] Lendo índices: {npz_path.name}")
        idx_data  = np.load(npz_path)
        train_idx = idx_data["train_idx"]
        test_idx  = idx_data["test_idx"]

        # Features
        X = np.stack(df["Features"].tolist()).astype(np.float32)

        # Vias disponíveis
        vias_disp = [v for v in VIAS if Y_COLS[v] in df.columns]

        # Dicionários y
        y_train = {v: df[Y_COLS[v]].values[train_idx].astype(np.float32) for v in vias_disp}
        y_test  = {v: df[Y_COLS[v]].values[test_idx].astype(np.float32)  for v in vias_disp}

        # Carrega modelo com custom_objects
        print(f"  [Carregador] Carregando modelo: {self.caminho_modelo.name}")
        custom_objs = {"CamadaIncerteza": CamadaIncerteza}
        for i in range(len(VIAS)):
            custom_objs[f"huber_uncert_task_{i}"] = get_huber_uncertainty_loss(i, None)

        model     = tf.keras.models.load_model(self.caminho_modelo, custom_objects=custom_objs)
        preds_raw = model.predict(X[test_idx], verbose=0)

        predicoes = {via: preds_raw[i].flatten() for i, via in enumerate(VIAS) if via in vias_disp}

        return {
            "model":     model,
            "X_train":   X[train_idx],
            "X_test":    X[test_idx],
            "y_train":   y_train,
            "y_test":    y_test,
            "predicoes": predicoes,
            "vias_disp": vias_disp,
        }

    def __repr__(self) -> str:
        return f"<CarregadorDados modelo='{self.caminho_modelo.name}'>"


# ══════════════════════════════════════════════════════════════
# ORQUESTRADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════

class ValidacaoCompleta:
    """
    Orquestra as etapas 5-A (Benchmark) e 5-B (Validação Estatística)
    para o modelo campeão de um dado método de split.

    Parâmetros
    ----------
    metodo_split : str
        Um de 'scaffold', 'butina', 'random'.
    incluir_scrambling : bool
        Se True, adiciona YScrambling à SuiteEstatistica (mais lento).
    n_scrambles : int
        Número de permutações no Y-Scrambling.

    Estrutura de saída
    ------------------
    results/tabelas/multitask/validacao/<metodo_split>/
        Benchmarks_<metodo_split>.xlsx
        Tabela_Estatistica_<metodo_split>.xlsx

    results/plots/multitask/validacao/<metodo_split>/
        williams_plot.png
    """

    # Mapeamento fixo: split → arquivo de ranking
    _RANKING_MAP = {
        "scaffold": PATHS["tabelas"] / "MTL_Ranking_scaffold_Modelos.xlsx",
        "butina":   PATHS["tabelas"] / "MTL_Ranking_butina_Modelos.xlsx",
        "random":   PATHS["tabelas"] / "MTL_Ranking_random_Modelos.xlsx",
    }

    def __init__(
        self,
        metodo_split: str,
        incluir_scrambling: bool = True,
        n_scrambles: int = 50,
    ) -> None:
        if metodo_split not in self._RANKING_MAP:
            raise ValueError(
                f"metodo_split deve ser um de {list(self._RANKING_MAP)}. "
                f"Recebido: '{metodo_split}'"
            )

        self.metodo_split        = metodo_split
        self.incluir_scrambling  = incluir_scrambling
        self.n_scrambles         = n_scrambles

        # Pastas de saída organizadas por split
        self.pasta_tabelas = PATHS["tabelas_validacao"] / metodo_split
        self.pasta_plots   = PATHS["plots_validacao"]   / metodo_split
        self.pasta_tabelas.mkdir(parents=True, exist_ok=True)
        self.pasta_plots.mkdir(parents=True, exist_ok=True)

        # Instâncias das suites (serão populadas em executar_tudo)
        self._benchmark_suite:   BenchmarkSuite   | None = None
        self._suite_estatistica: SuiteEstatistica | None = None
        self._dados:             dict              | None = None

    # ── Etapa de identificação do campeão ─────────────────────

    def identificar_campeao(self) -> Path:
        """
        Lê o ranking do split e retorna o caminho do modelo #1.
        """
        ranking_path = self._RANKING_MAP[self.metodo_split]
        if not ranking_path.exists():
            raise FileNotFoundError(
                f"Ranking não encontrado: {ranking_path}\n"
                "Execute a etapa de análise primeiro."
            )

        df_ranking = pd.read_excel(ranking_path)
        campeao    = df_ranking.iloc[0]

        caminho_modelo = Path(str(campeao["caminho_modelo"]))
        r2_medio       = campeao.get("r2_medio", "?")

        print(f"\n  [Campeão — {self.metodo_split.upper()}]")
        print(f"    Arquivo : {campeao['arquivo_origem']}")
        print(f"    R² Médio: {r2_medio:.4f}" if isinstance(r2_medio, float) else f"    R² Médio: {r2_medio}")
        print(f"    Modelo  : {caminho_modelo.name}")

        return caminho_modelo

    # ── Etapa de carregamento ──────────────────────────────────

    def carregar_dados(self) -> dict:
        """Identifica o campeão, carrega modelo e dados."""
        caminho_modelo = self.identificar_campeao()
        carregador     = CarregadorDados(caminho_modelo)
        self._dados    = carregador.carregar()
        return self._dados

    # ── Etapa 5-A: Benchmark ──────────────────────────────────

    def executar_benchmarks(self) -> pd.DataFrame:
        """
        Instancia e executa BenchmarkSuite.
        Requer carregar_dados() chamado antes.
        """
        if self._dados is None:
            raise RuntimeError("Chame carregar_dados() antes de executar_benchmarks().")

        print(f"\n{'─'*60}")
        print(f"  ETAPA 5-A — BENCHMARK | Split: {self.metodo_split.upper()}")
        print(f"{'─'*60}")

        self._benchmark_suite = BenchmarkSuite(pasta_saida=self.pasta_tabelas)
        df_bench = self._benchmark_suite.executar(
            X_train      = self._dados["X_train"],
            X_test       = self._dados["X_test"],
            y_train_dict = self._dados["y_train"],
            y_test_dict  = self._dados["y_test"],
        )
        self._benchmark_suite.salvar(f"Benchmarks_{self.metodo_split}.xlsx")
        return df_bench

    # ── Etapa 5-B: Validação Estatística ──────────────────────

    def executar_estatistica(self) -> dict[str, pd.DataFrame]:
        """
        Instancia e executa SuiteEstatistica.
        Requer carregar_dados() chamado antes.
        """
        if self._dados is None:
            raise RuntimeError("Chame carregar_dados() antes de executar_estatistica().")

        print(f"\n{'─'*60}")
        print(f"  ETAPA 5-B — VALIDAÇÃO ESTATÍSTICA | Split: {self.metodo_split.upper()}")
        print(f"{'─'*60}")

        self._suite_estatistica = SuiteEstatistica(
            pasta_tabelas       = self.pasta_tabelas,
            pasta_plots         = self.pasta_plots,
            incluir_scrambling  = self.incluir_scrambling,
            n_scrambles         = self.n_scrambles,
        )
        resultados = self._suite_estatistica.executar(self._dados)
        self._suite_estatistica.salvar(prefixo=self.metodo_split)
        return resultados

    # ── Pipeline completo ──────────────────────────────────────

    def executar_tudo(self) -> None:
        """
        Executa a sequência completa:
        1. Identificar e carregar campeão
        2. 5-A: Benchmarks
        3. 5-B: Validação Estatística
        """
        print(f"\n{'='*60}")
        print(f"  VALIDAÇÃO COMPLETA — {self.metodo_split.upper()}")
        print(f"{'='*60}")

        self.carregar_dados()
        self.executar_benchmarks()
        self.executar_estatistica()

        print(f"\n  ✓ Validação concluída para: {self.metodo_split.upper()}")
        print(f"  Tabelas em: {self.pasta_tabelas}")
        print(f"  Plots em  : {self.pasta_plots}")

    def __repr__(self) -> str:
        return (
            f"<ValidacaoCompleta split='{self.metodo_split}' "
            f"scrambling={self.incluir_scrambling}>"
        )


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main(
    metodos: list[str] | None = None,
    incluir_scrambling: bool = True,
    n_scrambles: int = 50,
) -> None:
    """
    Executa ValidacaoCompleta para cada método de split listado.

    Parâmetros
    ----------
    metodos : list[str] | None
        Splits a validar. Padrão: ['scaffold', 'butina', 'random'].
    incluir_scrambling : bool
        Inclui Y-Scrambling (padrão: True — GPU ≥10GB é suficiente).
    n_scrambles : int
        Número de permutações no Y-Scrambling (padrão: 50 — qualidade publicação).
    """
    if metodos is None:
        metodos = ["scaffold", "butina", "random"]

    print("\n" + "=" * 60)
    print("  PIPELINE DE VALIDAÇÃO — ETAPAS 5-A e 5-B")
    print(f"  Splits: {metodos}")
    print(f"  Y-Scrambling: {'Sim' if incluir_scrambling else 'Não'}")
    print("=" * 60)

    for metodo in metodos:
        try:
            val = ValidacaoCompleta(
                metodo_split       = metodo,
                incluir_scrambling = incluir_scrambling,
                n_scrambles        = n_scrambles,
            )
            val.executar_tudo()
        except Exception as e:
            print(f"\n  ✗ Erro no split '{metodo}': {e}")
            import traceback
            traceback.print_exc()

    print("\n  ✓ Pipeline de validação concluído!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ToxiciTOOL — Validação Estatística (5-A + 5-B)")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["scaffold", "butina", "random"],
        default=["scaffold", "butina", "random"],
        help="Métodos de split a validar (padrão: todos)",
    )
    parser.add_argument(
        "--scrambling",
        action="store_true",
        help="Inclui Y-Scrambling (mais lento)",
    )
    parser.add_argument(
        "--n-scrambles",
        type=int,
        default=10,
        help="Número de permutações no Y-Scrambling (padrão: 10)",
    )
    args = parser.parse_args()

    main(
        metodos            = args.splits,
        incluir_scrambling = args.scrambling,
        n_scrambles        = args.n_scrambles,
    )
