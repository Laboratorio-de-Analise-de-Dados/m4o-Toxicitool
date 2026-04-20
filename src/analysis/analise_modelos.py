"""
analise_modelos.py  (src/analysis/)
=====================================
Lê o log de métricas, gera rankings e gráficos comparativos.

Saídas globais (todos os métodos juntos):
    - MTL_Ranking_Modelos.xlsx
    - panorama_r2_vs_mae.png

Saídas por método de split:
    - MTL_Ranking_<método>.xlsx
    - comparativo_r2_por_via_<método>.png
    - comparativo_mae_por_via_<método>.png

Método de split registrado na coluna 'metodo_split' do log.
"""

import shutil
from pathlib import Path
from adjustText import adjust_text

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PATHS, VIAS


# ==========================================
# ANÁLISE DE SCORES
# ==========================================

def carregar_log() -> pd.DataFrame:
    if not PATHS["log_excel"].exists():
        raise FileNotFoundError(f"Log não encontrado: {PATHS['log_excel']}")
    df = pd.read_excel(PATHS["log_excel"])
    print(f"Log carregado: {len(df)} experimentos\n")
    return df


def calcular_scores(df: pd.DataFrame) -> pd.DataFrame:
    df       = df.copy()
    cols_r2  = [f"r2_{v}"  for v in VIAS if f"r2_{v}"  in df.columns]
    cols_mae = [f"mae_{v}" for v in VIAS if f"mae_{v}" in df.columns]

    df["r2_medio"]  = df[cols_r2].mean(axis=1, skipna=True)
    df["mae_medio"] = df[cols_mae].mean(axis=1, skipna=True)
    df["n_vias"]    = df[cols_r2].notna().sum(axis=1)

    r2_norm  = (df["r2_medio"]  - df["r2_medio"].min())  / (df["r2_medio"].max()  - df["r2_medio"].min()  + 1e-9)
    mae_norm = (df["mae_medio"] - df["mae_medio"].min()) / (df["mae_medio"].max() - df["mae_medio"].min() + 1e-9)
    df["score_composto"] = r2_norm - mae_norm
    return df


def rankear(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        by=["r2_medio", "mae_medio", "loss_global"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    df.index     += 1
    df.index.name = "ranking"
    return df


def identificar_melhor(df_ranked: pd.DataFrame) -> pd.Series:
    melhor = df_ranked.iloc[0]
    print("=" * 60)
    print("  MELHOR MODELO (global)")
    print("=" * 60)
    print(f"  Arquivo    : {melhor['arquivo_origem']}")
    print(f"  Split      : {melhor.get('metodo_split', '?')}")
    print(f"  R² médio   : {melhor['r2_medio']:.4f}")
    print(f"  MAE médio  : {melhor['mae_medio']:.4f}")
    print(f"  Loss       : {melhor['loss_global']:.4f}")
    print(f"  Nº vias    : {int(melhor['n_vias'])}")
    for via in VIAS:
        if f"r2_{via}" in melhor and not pd.isna(melhor[f"r2_{via}"]):
            print(f"    {via:12s} → R²={melhor[f'r2_{via}']:.3f} | MAE={melhor[f'mae_{via}']:.3f}")
    print("=" * 60)
    return melhor


# ==========================================
# CÓPIA DO MELHOR MODELO
# ==========================================

def copiar_melhor_modelo(melhor: pd.Series) -> None:
    """
    Copia o .keras do melhor modelo para results/melhor_modelo/.
    """
    destino = PATHS["melhor_modelo"]

    caminho_str = melhor.get("caminho_modelo", "")
    if caminho_str:
        origem = Path(caminho_str)
        if origem.exists():
            shutil.copy2(origem, destino / origem.name)
            print(f"✓ Modelo copiado: {destino / origem.name}")
        else:
            print(f"⚠  Modelo não encontrado: {origem}")
    else:
        print("⚠  Coluna 'caminho_modelo' não encontrada no log.")

    melhor.to_frame(name="valor").to_excel(destino / "resumo_melhor_modelo.xlsx")
    print(f"✓ Resumo salvo: {destino / 'resumo_melhor_modelo.xlsx'}")


# ==========================================
# VISUALIZAÇÕES E ESTÉTICA ACADÊMICA
# ==========================================

# Estética Global (Fundo branco opaco, grade sutil, sem bordas pesadas)
plt.rcParams.update({
    "font.family": "serif",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def _labels_top(top: pd.DataFrame) -> list[str]:
    """Encurta os nomes dos arquivos iterando pelas linhas."""
    labels = []
    for i, (_, row) in enumerate(top.iterrows()):
        nome = row['arquivo_origem'].replace('MTL_df_final_', '').replace('.pkl', '')
        nome = nome.replace('quantitativo', 'Quant').replace('binario', 'Bin')
        nome = nome.replace('bits', 'b').replace('raio', 'R')
        nome = nome.replace('__desc', '\n(+Desc)')
        labels.append(f"#{i+1}\n{nome}")
    return labels

def _ajustar_limite_y_barras(ax, vals, folga_percentual=0.15):
    """Garante que há espaço acima das barras para os rótulos numéricos."""
    ymax = max([v for v in vals if not np.isnan(v)] + [0])
    ax.set_ylim(0, ymax * (1 + folga_percentual))

def plot_r2_por_via(df_ranked: pd.DataFrame, top_n: int = 10, sufixo: str = "") -> None:
    vias_disp = [v for v in VIAS if f"r2_{v}" in df_ranked.columns]
    top       = df_ranked.head(top_n).copy()
    labels    = _labels_top(top)
    x, width  = np.arange(len(top)), 0.8 / len(vias_disp)
    fig, ax   = plt.subplots(figsize=(max(12, top_n * 1.2), 6))

    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    all_vals = []
    for j, via in enumerate(vias_disp):
        offset = (j - len(vias_disp) / 2 + 0.5) * width
        vals   = top[f"r2_{via}"].fillna(0).to_numpy(dtype=float)
        all_vals.extend(vals)
        bars   = ax.bar(x + offset, vals, width, label=via, color=cores[j % len(cores)], edgecolor="black", linewidth=0.5)
        
        for bar, val in zip(bars, vals):
            if val > 0.05:
                # Distanciamento vertical ajustado para evitar sobreposição 
                # (1% da altura máxima do eixo)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.axhline(0.5, color="#333333", linestyle=":", linewidth=1.2, label="Baseline R²=0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Coeficiente de Determinação ($R^2$)", fontsize=11)
    
    _ajustar_limite_y_barras(ax, all_vals)
    # Fixar o limite superior do R2 caso o maior valor seja alto demais.
    if ax.get_ylim()[1] < 1.1:
         ax.set_ylim(0, 1.1)

    titulo = f"Desempenho Preditivo por Via de Exposição ($R^2$) — Top {top_n}" + (f" ({sufixo})" if sufixo else "")
    ax.set_title(titulo, fontsize=12, pad=15, fontweight="bold")
    
    # Legenda fora da área de plotagem
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(vias_disp)+1, frameon=False)
    
    plt.tight_layout()
    nome = f"comparativo_r2_por_via{'_' + sufixo if sufixo else ''}.png"
    plt.savefig(PATHS["plots_analise"] / nome, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot R² salvo: {nome}")


def plot_mae_por_via(df_ranked: pd.DataFrame, top_n: int = 10, sufixo: str = "") -> None:
    vias_disp = [v for v in VIAS if f"mae_{v}" in df_ranked.columns]
    top       = df_ranked.head(top_n).copy()
    labels    = _labels_top(top)
    x, width  = np.arange(len(top)), 0.8 / len(vias_disp)
    fig, ax   = plt.subplots(figsize=(max(12, top_n * 1.2), 6))

    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    all_vals = []
    for j, via in enumerate(vias_disp):
        offset = (j - len(vias_disp) / 2 + 0.5) * width
        vals   = top[f"mae_{via}"].fillna(0).to_numpy(dtype=float)
        all_vals.extend(vals)
        bars   = ax.bar(x + offset, vals, width, label=via, color=cores[j % len(cores)], edgecolor="black", linewidth=0.5)
        
        for bar, val in zip(bars, vals):
            if val > 0.01:
                 # Rotação em 90 graus e folga vertical para caber sem esbarrar
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Erro Absoluto Médio (MAE log-escala)", fontsize=11)
    
    _ajustar_limite_y_barras(ax, all_vals)

    titulo = f"Erro Médio por Via de Exposição (MAE) — Top {top_n}" + (f" ({sufixo})" if sufixo else "")
    ax.set_title(titulo, fontsize=12, pad=15, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(vias_disp), frameon=False)
    
    plt.tight_layout()
    nome = f"comparativo_mae_por_via{'_' + sufixo if sufixo else ''}.png"
    plt.savefig(PATHS["plots_analise"] / nome, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot MAE salvo: {nome}")


def plot_panorama(df_ranked: pd.DataFrame) -> None:
    """Scatter global com repulsão de texto básica e símbolo acadêmico para o melhor.""" 
    
    fig, ax = plt.subplots(figsize=(10, 7))

    metodos_unicos = df_ranked["metodo_split"].unique() if "metodo_split" in df_ranked.columns else ["todos"]
    paleta = {"scaffold": "#4C72B0", "butina_cutoff0.4": "#DD8452", "random": "#55A868", "butina_batch": "#C44E52", "todos": "#8172B3"}

    for metodo in metodos_unicos:
        sub = df_ranked[df_ranked["metodo_split"] == metodo] if "metodo_split" in df_ranked.columns else df_ranked
        cor = paleta.get(metodo, "#7f7f7f")
        ax.scatter(sub["mae_medio"], sub["r2_medio"],
                   c=cor, s=90, alpha=0.8, edgecolors="white", linewidths=0.8,
                   label=metodo.replace("_cutoff0.4", ""))

    melhor = df_ranked.iloc[0]
    
    ax.scatter(melhor["mae_medio"], melhor["r2_medio"],
               facecolor="none", edgecolor="black", s=400, linewidth=2, zorder=4)
    ax.scatter(melhor["mae_medio"], melhor["r2_medio"],
               color="black", marker="+", s=200, linewidth=2, zorder=5, label="Melhor Modelo")

    texts = []

    for i, (_, row) in enumerate(df_ranked.head(3).iterrows()):
        nome_curto = row["arquivo_origem"].replace("MTL_df_final_", "").replace(".pkl", "")
        nome_curto = nome_curto.replace("quantitativo", "Q").replace("binario", "B").replace("__desc", " +D")
        
        t = ax.text(row["mae_medio"], row["r2_medio"], f"#{i+1} {nome_curto}",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        texts.append(t)

    try:

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    except NameError:
        print("  [Aviso] adjustText não instalado. Repulsão de texto desativada no Panorama.")
        print("  Dica: pip install adjustText")


    ax.set_xlabel("Erro Absoluto Médio Global (MAE) $\\rightarrow$ Menor é melhor", fontsize=11)
    ax.set_ylabel("Coeficiente de Determinação Global ($R^2$) $\\rightarrow$ Maior é melhor", fontsize=11)
    ax.set_title("Panorama de Desempenho: $R^2$ vs MAE por Técnica de Split", fontsize=12, pad=15, fontweight="bold")
    
    ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0", fontsize=9)
    plt.tight_layout()
    plt.savefig(PATHS["plots_analise"] / "panorama_r2_vs_mae.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot panorama salvo")


def salvar_ranking(df_ranked: pd.DataFrame, sufixo: str = "") -> None:
    nome    = f"MTL_Ranking{'_' + sufixo if sufixo else ''}_Modelos.xlsx"
    caminho = PATHS["tabelas"] / nome
    df_ranked.to_excel(caminho, index=False)
    print(f"✓ Ranking salvo: {caminho}")


# ==========================================
# ANÁLISE POR MÉTODO DE SPLIT
# ==========================================

def analisar_por_metodo(df: pd.DataFrame) -> None:
    if "metodo_split" not in df.columns:
        print("  Coluna 'metodo_split' não encontrada — pulando análise por método.")
        return

    metodos = sorted(df["metodo_split"].dropna().unique())
    print(f"\nMétodos de split encontrados: {metodos}")

    for metodo in metodos:
        sub = df[df["metodo_split"] == metodo].copy()
        if len(sub) == 0:
            continue

        print(f"\n{'─'*50}")
        print(f"  Método: {metodo} ({len(sub)} experimentos)")

        df_ranked = rankear(sub)
        melhor    = df_ranked.iloc[0]
        print(f"  Melhor → R²={melhor['r2_medio']:.4f} | MAE={melhor['mae_medio']:.4f} | {melhor['arquivo_origem']}")

        top_n = min(10, len(df_ranked))
        plot_r2_por_via(df_ranked,  top_n=top_n, sufixo=metodo)
        plot_mae_por_via(df_ranked, top_n=top_n, sufixo=metodo)
        salvar_ranking(df_ranked, sufixo=metodo)


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("  ANÁLISE DE MODELOS MTL")
    print("=" * 60)

    df        = carregar_log()
    df        = calcular_scores(df)
    df_ranked = rankear(df)
    melhor    = identificar_melhor(df_ranked)

    copiar_melhor_modelo(melhor)

    # ── Gráficos e ranking globais ─────────────────────────────────
    top_n = min(10, len(df_ranked))
    plot_r2_por_via(df_ranked,  top_n=top_n)
    plot_mae_por_via(df_ranked, top_n=top_n)
    plot_panorama(df_ranked)
    salvar_ranking(df_ranked)

    # ── Análise separada por método de split ───────────────────────
    analisar_por_metodo(df)

    print("\n✓ Análise concluída!")
    return df_ranked, melhor


if __name__ == "__main__":
    df_ranked, melhor = main()