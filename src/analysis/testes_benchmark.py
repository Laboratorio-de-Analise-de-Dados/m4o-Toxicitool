"""
testes_benchmark.py  (src/analysis/)
======================================
Classes OOP para benchmarks single-task (Etapa 5-A).

Hierarquia:
    BaseBenchmark          ← classe abstrata com lógica compartilhada
    ├── RandomForestBenchmark
    ├── RidgeBenchmark
    └── DummyBenchmark

    BenchmarkSuite         ← orquestra e consolida todos os benchmarks

Uso:
    suite = BenchmarkSuite(pasta_saida=Path(".../validacao/scaffold"))
    df_resultado = suite.executar(X_train, X_test, y_train_dict, y_test_dict)
    suite.salvar("Benchmarks_scaffold.xlsx")
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Garante que a raiz do projeto está no path
_SRC = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import VIAS, Y_COLS


# ══════════════════════════════════════════════════════════════
# CLASSE BASE
# ══════════════════════════════════════════════════════════════

class BaseBenchmark(ABC):
    """
    Classe base abstrata para modelos de benchmark single-task.

    Subclasses devem implementar `_criar_modelo()`, que retorna
    um estimador sklearn com interface fit/predict.

    Atributos públicos após `executar_todas_vias()`:
        resultados (list[dict]): linha por via com métricas.
    """

    def __init__(self, nome: str) -> None:
        self.nome = nome
        self.resultados: list[dict] = []

    # ── Interface obrigatória ──────────────────────────────────

    @abstractmethod
    def _criar_modelo(self) -> Any:
        """Instancia e retorna o estimador sklearn configurado."""
        ...

    # ── Lógica compartilhada ───────────────────────────────────

    @staticmethod
    def _sanitizar_X(X: np.ndarray) -> np.ndarray:
        """
        Remove Infs e NaNs das features para compatibilidade com modelos
        baseados em álgebra linear (Ridge, PCA, etc.).

        Implementação vetorizada (numpy puro — sem loop Python por coluna):
            - ±Inf → substituídos por ±1e9 (np.nan_to_num)
            - NaN  → substituídos por 0.0  (np.nan_to_num)

        Nota: Usar mediana coluna-a-coluna seria mais preciso, mas
        introduziria um loop Python sobre 2000+ colunas (~lento para 50K amostras).
        Para benchmarks comparativos, a substituição por 0/±1e9 é suficiente,
        pois os descritores já foram escalonados (RobustScaler) no pipeline.
        """
        return np.nan_to_num(
            X.astype(np.float64),
            nan=0.0,
            posinf=1e9,
            neginf=-1e9,
        )

    def executar_via(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        via: str,
    ) -> dict | None:
        """
        Treina e avalia o modelo para uma única via de exposição.

        Retorna None se dados insuficientes.
        """
        mask_train = np.isfinite(y_train)
        mask_test  = np.isfinite(y_test)

        X_tr, y_tr = X_train[mask_train], y_train[mask_train]
        X_te, y_te = X_test[mask_test],   y_test[mask_test]

        if len(y_tr) < 10 or len(y_te) < 2:
            print(f"    [!] {self.nome} | {via}: dados insuficientes "
                  f"(treino={len(y_tr)}, teste={len(y_te)}). Pulando.")
            return None

        # Sanitiza features para modelos que não toleram Inf/NaN (ex: Ridge)
        X_tr = self._sanitizar_X(X_tr)
        X_te = self._sanitizar_X(X_te)

        modelo = self._criar_modelo()
        modelo.fit(X_tr, y_tr)
        preds = modelo.predict(X_te)

        return {
            "Modelo":   self.nome,
            "Via":      via,
            "R2":       r2_score(y_te, preds),
            "MAE":      mean_absolute_error(y_te, preds),
            "RMSE":     root_mean_squared_error(y_te, preds),
            "N_Treino": int(len(y_tr)),
            "N_Teste":  int(len(y_te)),
        }

    def executar_todas_vias(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train_dict: dict[str, np.ndarray],
        y_test_dict: dict[str, np.ndarray],
    ) -> list[dict]:
        """
        Itera sobre todas as vias disponíveis e consolida resultados.
        """
        self.resultados = []
        for via in VIAS:
            col = Y_COLS[via]
            if col not in y_train_dict and via not in y_train_dict:
                continue

            # Suporta tanto dicts com chave via quanto col
            y_tr = y_train_dict.get(via, y_train_dict.get(col))
            y_te = y_test_dict.get(via, y_test_dict.get(col))

            if y_tr is None or y_te is None:
                continue

            resultado = self.executar_via(X_train, y_tr, X_test, y_te, via)
            if resultado:
                self.resultados.append(resultado)

        return self.resultados

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} nome='{self.nome}'>"


# ══════════════════════════════════════════════════════════════
# IMPLEMENTAÇÕES CONCRETAS
# ══════════════════════════════════════════════════════════════

class RandomForestBenchmark(BaseBenchmark):
    """
    Random Forest Regressor single-task.

    Parâmetros
    ----------
    n_estimators : int
        Número de árvores (padrão 100).
    n_jobs : int
        Paralelismo (-1 = todos os cores).
    random_state : int
        Semente para reprodutibilidade.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        super().__init__("Random Forest (ST)")
        self.n_estimators  = n_estimators
        self.n_jobs        = n_jobs
        self.random_state  = random_state

    def _criar_modelo(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )


class RidgeBenchmark(BaseBenchmark):
    """
    Ridge Regression (L2) single-task.

    Parâmetros
    ----------
    alpha : float
        Força de regularização L2 (padrão 10.0 — mais estável com features
        de alta dimensão e escalas variadas como fingerprints + descritores).
    solver : str
        Solver numérico. 'lsqr' é mais estável que 'cholesky' para matrizes
        mal condicionadas (alta dimensão, features correlacionadas).
    """

    def __init__(self, alpha: float = 10.0, solver: str = "lsqr") -> None:
        super().__init__("Ridge Regression")
        self.alpha  = alpha
        self.solver = solver

    def _criar_modelo(self) -> Ridge:
        return Ridge(alpha=self.alpha, solver=self.solver)


class DummyBenchmark(BaseBenchmark):
    """
    Regressor Dummy (baseline nulo).

    Parâmetros
    ----------
    strategy : str
        Estratégia de predição: 'mean', 'median', 'constant'.
    """

    def __init__(self, strategy: str = "mean") -> None:
        super().__init__(f"Dummy ({strategy})")
        self.strategy = strategy

    def _criar_modelo(self) -> DummyRegressor:
        return DummyRegressor(strategy=self.strategy)


# ══════════════════════════════════════════════════════════════
# SUITE: ORQUESTRA TODOS OS BENCHMARKS
# ══════════════════════════════════════════════════════════════

class BenchmarkSuite:
    """
    Orquestra e consolida os resultados de múltiplos benchmarks.

    Parâmetros
    ----------
    pasta_saida : Path
        Diretório onde o Excel de resultados será salvo.
    benchmarks : list[BaseBenchmark] | None
        Lista de instâncias de benchmark. Se None, usa RF + Ridge + Dummy.

    Exemplo
    -------
    suite = BenchmarkSuite(pasta_saida=Path("results/.../validacao/scaffold"))
    df = suite.executar(X_train, X_test, y_train_dict, y_test_dict)
    suite.salvar("Benchmarks_scaffold.xlsx")
    """

    def __init__(
        self,
        pasta_saida: Path,
        benchmarks: list[BaseBenchmark] | None = None,
    ) -> None:
        self.pasta_saida = Path(pasta_saida)
        self.pasta_saida.mkdir(parents=True, exist_ok=True)

        self.benchmarks: list[BaseBenchmark] = benchmarks or [
            RandomForestBenchmark(),
            RidgeBenchmark(),
            DummyBenchmark(strategy="mean"),
        ]
        self._df_resultado: pd.DataFrame | None = None

    # ── API pública ────────────────────────────────────────────

    def executar(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train_dict: dict[str, np.ndarray],
        y_test_dict: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Executa todos os benchmarks e retorna DataFrame consolidado.
        """
        print(f"\n{'='*60}")
        print(f"  BENCHMARK SUITE ({len(self.benchmarks)} modelos)")
        print(f"{'='*60}")

        todos = []
        for bench in self.benchmarks:
            print(f"\n  › Executando: {bench.nome}")
            resultados = bench.executar_todas_vias(
                X_train, X_test, y_train_dict, y_test_dict
            )
            todos.extend(resultados)

        self._df_resultado = pd.DataFrame(todos)

        if not self._df_resultado.empty:
            self._imprimir_resumo()

        return self._df_resultado

    def salvar(self, nome_arquivo: str) -> Path:
        """Salva o DataFrame de resultados em Excel."""
        if self._df_resultado is None or self._df_resultado.empty:
            raise RuntimeError("Execute suite.executar() antes de salvar.")

        caminho = self.pasta_saida / nome_arquivo
        self._df_resultado.to_excel(caminho, index=False)
        print(f"  ✓ Benchmarks salvos: {caminho}")
        return caminho

    @property
    def resultado(self) -> pd.DataFrame | None:
        """Acesso ao DataFrame de resultado após executar()."""
        return self._df_resultado

    # ── Utilitários internos ───────────────────────────────────

    def _imprimir_resumo(self) -> None:
        """Imprime tabela pivô de R² no console."""
        try:
            pivot = self._df_resultado.pivot_table(
                index="Via", columns="Modelo", values="R2"
            )
            print(f"\n  --- R² por Via (benchmark) ---\n{pivot.to_string()}\n")
        except Exception:
            pass  # Não crítico

    def __repr__(self) -> str:
        nomes = [b.nome for b in self.benchmarks]
        return f"<BenchmarkSuite benchmarks={nomes}>"
