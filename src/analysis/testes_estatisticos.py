"""
testes_estatisticos.py  (src/analysis/)
=========================================
Classes OOP para os testes estatísticos de validação (Etapa 5-B).

Hierarquia:
    BaseTesteEstatistico      ← classe abstrata com interface comum
    ├── CorrelacaoBootstrap   — Pearson, Spearman, CCC com IC 95% via bootstrap
    ├── WilcoxonTest          — Wilcoxon Signed-Rank (MTL vs Baseline Nulo) + BH
    ├── YScrambling           — Randomização de targets (10 iterações por padrão)
    └── WilliamsPlot          — Domínio de Aplicabilidade (Leverage vs Resíduo)

    SuiteEstatistica          ← orquestra todos os testes e salva o Excel unificado

Convenção de dados:
    Todos os testes recebem um dicionário `dados` com as chaves:
        X_train      (np.ndarray)
        X_test       (np.ndarray)
        y_train      (dict[str, np.ndarray])  — chave: nome da via
        y_test       (dict[str, np.ndarray])
        predicoes    (dict[str, np.ndarray])  — saídas do modelo MTL
        vias_disp    (list[str])              — vias com dados não-nulos

Uso:
    suite = SuiteEstatistica(pasta_tabelas=..., pasta_plots=...)
    suite.executar(dados)
    suite.salvar(prefixo="scaffold")
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.multitest import multipletests

# Garante que a raiz do projeto está no path
_SRC = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import VIAS

# ── Estética global dos gráficos ───────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "axes.grid":          True,
    "grid.color":         "#E0E0E0",
    "grid.linestyle":     "--",
    "axes.axisbelow":     True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ══════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES PURAS (sem estado)
# ══════════════════════════════════════════════════════════════

def _concordance_correlation_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Concordance Correlation Coefficient (Lin, 1989).
    CCC = 2·ρ·σ_t·σ_p / (σ_t² + σ_p² + (μ_t − μ_p)²)
    """
    if len(y_true) < 2:
        return np.nan
    cor     = np.corrcoef(y_true, y_pred)[0, 1]
    sd_t    = np.std(y_true)
    sd_p    = np.std(y_pred)
    num     = 2 * cor * sd_t * sd_p
    den     = np.var(y_true) + np.var(y_pred) + (np.mean(y_true) - np.mean(y_pred)) ** 2
    return float(num / den) if den != 0 else np.nan


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrica: Callable,
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Estimativa bootstrap de IC 95% para uma métrica escalar.

    Retorna: {'est': valor_original, 'low': p2.5, 'high': p97.5}
    """
    rng    = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            scores.append(metrica(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return {
        "est":  metrica(y_true, y_pred),
        "low":  float(np.percentile(scores, 2.5))  if scores else np.nan,
        "high": float(np.percentile(scores, 97.5)) if scores else np.nan,
    }


def _mascara_valida(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaNs simultâneos de y_true e y_pred."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


# ══════════════════════════════════════════════════════════════
# CLASSE BASE
# ══════════════════════════════════════════════════════════════

class BaseTesteEstatistico(ABC):
    """
    Interface comum para todos os testes estatísticos de validação.

    Subclasses devem implementar `executar(dados)` e opcionalmente
    sobrescrever `salvar()` para comportamentos customizados.

    Atributos públicos após `executar()`:
        resultado (pd.DataFrame | None): tabela de saída do teste.
    """

    def __init__(self, nome: str, pasta_tabelas: Path, pasta_plots: Path) -> None:
        self.nome         = nome
        self.pasta_tabelas = Path(pasta_tabelas)
        self.pasta_plots   = Path(pasta_plots)
        self.pasta_tabelas.mkdir(parents=True, exist_ok=True)
        self.pasta_plots.mkdir(parents=True, exist_ok=True)
        self.resultado: pd.DataFrame | None = None

    # ── Interface obrigatória ──────────────────────────────────

    @abstractmethod
    def executar(self, dados: dict) -> pd.DataFrame:
        """
        Executa o teste e armazena em self.resultado.

        Parâmetros
        ----------
        dados : dict
            Dicionário com X_train, X_test, y_train, y_test,
            predicoes e vias_disp (ver docstring do módulo).

        Retorna
        -------
        pd.DataFrame com os resultados do teste.
        """
        ...

    # ── Implementação padrão de save ──────────────────────────

    def salvar(self, nome_arquivo: str) -> Path:
        """Persiste self.resultado em Excel dentro de pasta_tabelas."""
        if self.resultado is None or self.resultado.empty:
            raise RuntimeError(
                f"[{self.nome}] Execute executar() antes de salvar()."
            )
        caminho = self.pasta_tabelas / nome_arquivo
        self.resultado.to_excel(caminho, index=True)
        print(f"  ✓ [{self.nome}] Tabela salva: {caminho.name}")
        return caminho

    def __repr__(self) -> str:
        n = len(self.resultado) if self.resultado is not None else "—"
        return f"<{self.__class__.__name__} nome='{self.nome}' linhas={n}>"


# ══════════════════════════════════════════════════════════════
# TESTE 1 — CORRELAÇÃO COM BOOTSTRAP CI 95 %
# ══════════════════════════════════════════════════════════════

class CorrelacaoBootstrap(BaseTesteEstatistico):
    """
    Calcula Pearson (r), Spearman (ρ), CCC, R² e MAE com IC 95%
    via bootstrap para cada via de exposição.

    Parâmetros
    ----------
    pasta_tabelas, pasta_plots : Path
    n_boot : int
        Número de reamostras bootstrap (padrão 1000).
    random_state : int
        Semente para reprodutibilidade.
    """

    def __init__(
        self,
        pasta_tabelas: Path,
        pasta_plots: Path,
        n_boot: int = 1000,
        random_state: int = 42,
    ) -> None:
        super().__init__("Correlação Bootstrap", pasta_tabelas, pasta_plots)
        self.n_boot       = n_boot
        self.random_state = random_state

    def executar(self, dados: dict) -> pd.DataFrame:
        """
        Pearson, Spearman, CCC, R², MAE — todos com IC 95% bootstrap.
        """
        print(f"\n  › [{self.nome}] Calculando (n_boot={self.n_boot})...")
        linhas = []

        for via in dados["vias_disp"]:
            y_t_raw = dados["y_test"].get(via)
            y_p_raw = dados["predicoes"].get(via)
            if y_t_raw is None or y_p_raw is None:
                continue

            y_t, y_p = _mascara_valida(y_t_raw, y_p_raw)
            if len(y_t) < 10:
                continue

            ci_r   = _bootstrap_ci(y_t, y_p,
                                   lambda a, b: stats.pearsonr(a, b)[0],
                                   self.n_boot, self.random_state)
            ci_rho = _bootstrap_ci(y_t, y_p,
                                   lambda a, b: stats.spearmanr(a, b)[0],
                                   self.n_boot, self.random_state)
            ci_ccc = _bootstrap_ci(y_t, y_p,
                                   _concordance_correlation_coefficient,
                                   self.n_boot, self.random_state)
            ci_r2  = _bootstrap_ci(y_t, y_p,
                                   r2_score,
                                   self.n_boot, self.random_state)
            ci_mae = _bootstrap_ci(y_t, y_p,
                                   mean_absolute_error,
                                   self.n_boot, self.random_state)

            linhas.append({
                "Via":              via,
                "n":                len(y_t),
                "Pearson_r":        ci_r["est"],
                "Pearson_IC95":     f"[{ci_r['low']:.3f}; {ci_r['high']:.3f}]",
                "Spearman_rho":     ci_rho["est"],
                "Spearman_IC95":    f"[{ci_rho['low']:.3f}; {ci_rho['high']:.3f}]",
                "CCC":              ci_ccc["est"],
                "CCC_IC95":         f"[{ci_ccc['low']:.3f}; {ci_ccc['high']:.3f}]",
                "R2":               ci_r2["est"],
                "R2_IC95":          f"[{ci_r2['low']:.3f}; {ci_r2['high']:.3f}]",
                "MAE":              ci_mae["est"],
                "MAE_IC95":         f"[{ci_mae['low']:.3f}; {ci_mae['high']:.3f}]",
            })

        self.resultado = (
            pd.DataFrame(linhas).set_index("Via") if linhas else pd.DataFrame()
        )
        return self.resultado


# ══════════════════════════════════════════════════════════════
# TESTE 2 — WILCOXON SIGNED-RANK (MTL vs BASELINE NULO)
# ══════════════════════════════════════════════════════════════

class WilcoxonTest(BaseTesteEstatistico):
    """
    Testa se o MTL é estatisticamente superior ao baseline nulo
    (predição = média do treino) usando Wilcoxon Signed-Rank.

    Correção de múltiplas comparações: Benjamini-Hochberg (FDR).

    Parâmetros
    ----------
    pasta_tabelas, pasta_plots : Path
    alpha : float
        Nível de significância (padrão 0.05).
    """

    def __init__(
        self,
        pasta_tabelas: Path,
        pasta_plots: Path,
        alpha: float = 0.05,
    ) -> None:
        super().__init__("Wilcoxon Signed-Rank", pasta_tabelas, pasta_plots)
        self.alpha = alpha

    def executar(self, dados: dict) -> pd.DataFrame:
        """
        Wilcoxon de uma cauda (MTL < Baseline) + BH para cada via.
        """
        print(f"\n  › [{self.nome}] Testando MTL vs Baseline Nulo...")
        linhas = []

        for via in dados["vias_disp"]:
            y_t_raw = dados["y_test"].get(via)
            y_p_raw = dados["predicoes"].get(via)
            y_tr    = dados["y_train"].get(via)

            if y_t_raw is None or y_p_raw is None or y_tr is None:
                continue

            y_t, y_p = _mascara_valida(y_t_raw, y_p_raw)
            if len(y_t) < 10:
                continue

            media_treino = float(np.nanmean(y_tr))
            err_mtl  = np.abs(y_t - y_p)
            err_null = np.abs(y_t - media_treino)

            stat, p = stats.wilcoxon(err_mtl, err_null, alternative="less")

            linhas.append({
                "Via":           via,
                "n":             len(y_t),
                "MAE_MTL":       float(err_mtl.mean()),
                "MAE_Baseline":  float(err_null.mean()),
                "Melhoria_%":    100 * (err_null.mean() - err_mtl.mean()) / err_null.mean(),
                "W_stat":        float(stat),
                "p_valor":       float(p),
            })

        if not linhas:
            self.resultado = pd.DataFrame()
            return self.resultado

        df = pd.DataFrame(linhas).set_index("Via")

        # Correção de múltiplas comparações (BH)
        _, p_adj, _, _ = multipletests(df["p_valor"], method="fdr_bh")
        df["p_adj_BH"]       = p_adj
        df["Significativo"]  = p_adj < self.alpha

        self.resultado = df
        return self.resultado


# ══════════════════════════════════════════════════════════════
# TESTE 3 — Y-SCRAMBLING
# ══════════════════════════════════════════════════════════════

class YScrambling(BaseTesteEstatistico):
    """
    Valida que o modelo não aprende por correlações espúrias via
    randomização repetida dos targets de treino.

    n_scrambles modelos são treinados com y_train permutado;
    o p-valor empírico mede a raridade do R² original.

    Parâmetros
    ----------
    pasta_tabelas, pasta_plots : Path
    n_scrambles : int
        Número de permutações (padrão 10 — aumentar para publicação).
    epochs_scramble : int
        Épocas de treino por permutação (padrão 20 — reduzido para velocidade).
    """

    def __init__(
        self,
        pasta_tabelas: Path,
        pasta_plots: Path,
        n_scrambles: int = 50,
        epochs_scramble: int = 30,
    ) -> None:
        super().__init__("Y-Scrambling", pasta_tabelas, pasta_plots)
        self.n_scrambles     = n_scrambles
        self.epochs_scramble = epochs_scramble

    def executar(self, dados: dict) -> pd.DataFrame:
        """
        Treina n_scrambles modelos com y_train permutado e compara R² original.
        """
        import gc
        import tensorflow as tf
        from src.modeling.modelagem_MTL import modelo_multitask

        print(f"\n  › [{self.nome}] {self.n_scrambles} permutações "
              f"({self.epochs_scramble} épocas cada)...")

        vias = dados["vias_disp"]

        # R² originais
        r2_orig = {}
        for via in vias:
            y_t, y_p = _mascara_valida(
                dados["y_test"][via], dados["predicoes"][via]
            )
            r2_orig[via] = r2_score(y_t, y_p) if len(y_t) > 1 else np.nan

        # Permutações
        scrambles = []
        rng = np.random.default_rng(42)

        for i in range(self.n_scrambles):
            print(f"    Permutação {i+1}/{self.n_scrambles}...", end="\r")
            y_tr_s = {
                via: rng.permutation(dados["y_train"][via])
                for via in vias
            }
            model = modelo_multitask(fpSize=dados["X_train"].shape[1])
            model.fit(
                dados["X_train"], y_tr_s,
                epochs=self.epochs_scramble,
                batch_size=128,
                verbose=0,
            )
            preds = model.predict(dados["X_test"], verbose=0)
            row = {"iter": i + 1}
            for j, via in enumerate(VIAS):
                if via in vias:
                    y_t, y_p = _mascara_valida(
                        dados["y_test"][via], preds[j].flatten()
                    )
                    row[f"r2_{via}"] = r2_score(y_t, y_p) if len(y_t) > 1 else np.nan
            scrambles.append(row)

            del model
            tf.keras.backend.clear_session()
            gc.collect()

        print()  # nova linha após \r

        df_s = pd.DataFrame(scrambles)
        linhas = []
        for via in vias:
            col = f"r2_{via}"
            if col not in df_s.columns:
                continue
            # p-valor empírico: P(R²_scramble ≥ R²_original)
            p_emp = (np.sum(df_s[col] >= r2_orig[via]) + 1) / (self.n_scrambles + 1)
            linhas.append({
                "Via":               via,
                "R2_Original":       r2_orig[via],
                "R2_Scramble_Media": float(df_s[col].mean()),
                "R2_Scramble_DP":    float(df_s[col].std()),
                "p_valor_empirico":  float(p_emp),
            })

        df = pd.DataFrame(linhas).set_index("Via") if linhas else pd.DataFrame()

        if not df.empty:
            _, p_adj, _, _ = multipletests(df["p_valor_empirico"], method="fdr_bh")
            df["p_adj_BH"]      = p_adj
            df["Significativo"] = p_adj < 0.05

        self.resultado = df
        return self.resultado


# ══════════════════════════════════════════════════════════════
# TESTE 4 — WILLIAMS PLOT (DOMÍNIO DE APLICABILIDADE)
# ══════════════════════════════════════════════════════════════

class WilliamsPlot(BaseTesteEstatistico):
    """
    Domínio de Aplicabilidade via Williams Plot:
    Leverage (h) vs Resíduo Padronizado.

    Moléculas com h > h* ou |resíduo| > 3σ são consideradas fora do domínio.

    Parâmetros
    ----------
    pasta_tabelas, pasta_plots : Path
    n_componentes_pca : int
        Componentes PCA usados para calcular a matriz H⁻¹ (padrão 50).
    """

    def __init__(
        self,
        pasta_tabelas: Path,
        pasta_plots: Path,
        n_componentes_pca: int = 50,
    ) -> None:
        super().__init__("Williams Plot", pasta_tabelas, pasta_plots)
        self.n_componentes_pca = n_componentes_pca

    def executar(self, dados: dict) -> pd.DataFrame:
        """
        Calcula leverage e resíduos padronizados para cada molécula de teste.
        Gera o gráfico e retorna DataFrame com métricas de AD por molécula.
        """
        print(f"\n  › [{self.nome}] Calculando Domínio de Aplicabilidade (PCA)...")
        from sklearn.preprocessing import StandardScaler

        # Sanitiza Inf/NaN e normaliza antes do PCA para evitar overflow
        # (features = fingerprints binários + descritores em escalas muito diferentes)
        X_tr_raw = np.nan_to_num(dados["X_train"].astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        X_te_raw = np.nan_to_num(dados["X_test"].astype(np.float64),  nan=0.0, posinf=1e6, neginf=-1e6)

        scaler   = StandardScaler(with_std=False)  # centraliza sem dividir por DP (mais estável para binários)
        X_tr_raw = scaler.fit_transform(X_tr_raw)
        X_te_raw = scaler.transform(X_te_raw)

        n_comp = min(
            self.n_componentes_pca,
            dados["X_train"].shape[0],
            dados["X_train"].shape[1],
        )
        pca = PCA(n_components=n_comp)
        X_tr_pca = pca.fit_transform(X_tr_raw)
        X_te_pca = pca.transform(X_te_raw)

        # H = X(X'X)⁻¹X' → leverage diagonal
        XtX_inv     = np.linalg.pinv(X_tr_pca.T @ X_tr_pca)
        leverages   = np.array([x @ XtX_inv @ x for x in X_te_pca])
        h_star      = 3 * n_comp / X_tr_pca.shape[0]  # Limite crítico de leverage

        # Resumo de AD por via
        ad_rows = []
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        for i, via in enumerate(VIAS):
            ax = axes.flatten()[i]
            if via not in dados["vias_disp"]:
                ax.set_visible(False)
                continue

            y_t, y_p = _mascara_valida(
                dados["y_test"][via], dados["predicoes"][via]
            )
            if len(y_t) < 2:
                ax.set_visible(False)
                continue

            residuos   = y_p - y_t
            std_res    = residuos.std() or 1.0
            res_pad    = (residuos - residuos.mean()) / std_res

            dentro_ad  = (leverages < h_star) & (np.abs(res_pad) <= 3)
            pct_dentro = 100 * dentro_ad.mean()

            ad_rows.append({
                "Via":          via,
                "h_star":       round(h_star, 4),
                "n_total":      len(y_t),
                "n_dentro_AD":  int(dentro_ad.sum()),
                "pct_dentro_AD": round(pct_dentro, 2),
                "outliers_leverage":  int((leverages >= h_star).sum()),
                "outliers_residuo":   int((np.abs(res_pad) > 3).sum()),
            })

            # Plot
            cores = np.where(dentro_ad, "#4C72B0", "#C44E52")
            ax.scatter(leverages, res_pad, c=cores, alpha=0.6, s=15, edgecolors="none")
            ax.axhline(3,      color="#C44E52", linestyle="--", linewidth=1, label="|z|=3")
            ax.axhline(-3,     color="#C44E52", linestyle="--", linewidth=1)
            ax.axvline(h_star, color="#DD8452", linestyle="--", linewidth=1, label=f"h*={h_star:.3f}")
            ax.set_title(f"{via}\n{pct_dentro:.1f}% dentro do AD", fontsize=10)
            ax.set_xlabel("Leverage (h)", fontsize=9)
            ax.set_ylabel("Resíduo Padronizado (z)", fontsize=9)
            if i == 0:
                ax.legend(fontsize=8)

        plt.suptitle("Williams Plot — Domínio de Aplicabilidade", fontsize=13, fontweight="bold")
        plt.tight_layout()

        plot_path = self.pasta_plots / "williams_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  ✓ [{self.nome}] Plot salvo: {plot_path.name}")

        self.resultado = (
            pd.DataFrame(ad_rows).set_index("Via") if ad_rows else pd.DataFrame()
        )
        return self.resultado


# ══════════════════════════════════════════════════════════════
# SUITE: ORQUESTRA TODOS OS TESTES ESTATÍSTICOS
# ══════════════════════════════════════════════════════════════

class SuiteEstatistica:
    """
    Orquestra a execução de todos os testes estatísticos e consolida
    as saídas em um único Excel multi-aba formatado para publicação.

    Parâmetros
    ----------
    pasta_tabelas : Path
        Onde salvar os arquivos .xlsx.
    pasta_plots : Path
        Onde salvar os gráficos (Williams Plot, etc.).
    testes : list[BaseTesteEstatistico] | None
        Testes a executar. Se None, usa o conjunto padrão.

    Exemplo
    -------
    suite = SuiteEstatistica(
        pasta_tabelas=Path(".../validacao/scaffold"),
        pasta_plots=Path(".../plots/validacao/scaffold"),
    )
    suite.executar(dados)
    suite.salvar(prefixo="scaffold")
    """

    TESTES_PADRAO = (CorrelacaoBootstrap, WilcoxonTest, WilliamsPlot, YScrambling)
    # YScrambling incluído no padrão — GPU moderna (>=8GB VRAM) é suficiente.
    # Para desativar: incluir_scrambling=False

    def __init__(
        self,
        pasta_tabelas: Path,
        pasta_plots: Path,
        testes: list[BaseTesteEstatistico] | None = None,
        incluir_scrambling: bool = True,
        n_scrambles: int = 50,
    ) -> None:
        self.pasta_tabelas      = Path(pasta_tabelas)
        self.pasta_plots        = Path(pasta_plots)
        self.pasta_tabelas.mkdir(parents=True, exist_ok=True)
        self.pasta_plots.mkdir(parents=True, exist_ok=True)

        if testes is not None:
            self.testes = testes
        else:
            self.testes: list[BaseTesteEstatistico] = [
                CorrelacaoBootstrap(self.pasta_tabelas, self.pasta_plots),
                WilcoxonTest(self.pasta_tabelas, self.pasta_plots),
                WilliamsPlot(self.pasta_tabelas, self.pasta_plots),
            ]
            if incluir_scrambling:
                self.testes.append(
                    YScrambling(
                        self.pasta_tabelas,
                        self.pasta_plots,
                        n_scrambles=n_scrambles,
                    )
                )

        self._resultados: dict[str, pd.DataFrame] = {}

    # ── API pública ────────────────────────────────────────────

    def executar(self, dados: dict) -> dict[str, pd.DataFrame]:
        """
        Executa todos os testes e armazena resultados internamente.

        Retorna dict {nome_teste: DataFrame}.
        """
        print(f"\n{'='*60}")
        print(f"  SUITE ESTATÍSTICA ({len(self.testes)} testes)")
        print(f"{'='*60}")

        for teste in self.testes:
            try:
                df = teste.executar(dados)
                self._resultados[teste.nome] = df
            except Exception as e:
                print(f"  [!] Erro em [{teste.nome}]: {e}")
                self._resultados[teste.nome] = pd.DataFrame()

        return self._resultados

    def salvar(self, prefixo: str) -> Path:
        """
        Salva todos os resultados em um único Excel multi-aba.

        Aba 'Resumo_Publicacao' contém métricas principais de Correlação
        e Wilcoxon lado a lado — formatado para tabela de publicação ABNT.
        """
        nome_arquivo = f"Tabela_Estatistica_{prefixo}.xlsx"
        caminho      = self.pasta_tabelas / nome_arquivo

        with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
            # Aba de resumo para publicação
            self._escrever_resumo_publicacao(writer, prefixo)

            # Uma aba por teste
            for nome, df in self._resultados.items():
                if df is not None and not df.empty:
                    nome_aba = nome[:31]  # Excel limita a 31 caracteres
                    df.to_excel(writer, sheet_name=nome_aba)

        print(f"\n  ✓ Excel unificado salvo: {caminho}")
        return caminho

    @property
    def resultados(self) -> dict[str, pd.DataFrame]:
        """Acesso ao dicionário de resultados após executar()."""
        return self._resultados

    # ── Utilitários internos ───────────────────────────────────

    def _escrever_resumo_publicacao(
        self, writer: pd.ExcelWriter, prefixo: str
    ) -> None:
        """Consolida as métricas principais em uma aba única."""
        try:
            df_corr   = self._resultados.get("Correlação Bootstrap", pd.DataFrame())
            df_wilcox = self._resultados.get("Wilcoxon Signed-Rank", pd.DataFrame())
            df_scram  = self._resultados.get("Y-Scrambling", pd.DataFrame())

            partes = {}
            if not df_corr.empty:
                partes.update({
                    "R²":  df_corr["R2"],
                    "MAE": df_corr["MAE"],
                    "CCC": df_corr["CCC"],
                    "IC95_R2": df_corr["R2_IC95"],
                })
            if not df_wilcox.empty:
                partes.update({
                    "p_Wilcoxon (adj. BH)": df_wilcox["p_adj_BH"],
                    "Significativo (α=0.05)": df_wilcox["Significativo"],
                    "Melhoria_%": df_wilcox["Melhoria_%"],
                })
            if not df_scram.empty:
                partes["p_Scrambling (adj. BH)"] = df_scram["p_adj_BH"]

            if partes:
                df_pub = pd.DataFrame(partes)
                df_pub.index.name = "Via de Exposição"
                df_pub.to_excel(writer, sheet_name="Resumo_Publicacao_ABNT")

        except Exception as e:
            print(f"  [Aviso] Não foi possível gerar aba de resumo: {e}")

    def __repr__(self) -> str:
        nomes = [t.nome for t in self.testes]
        return f"<SuiteEstatistica testes={nomes}>"
