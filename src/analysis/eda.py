import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Adiciona a RAIZ ao path
_SRC  = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS, VIAS, Y_COLS

def plot_distribuicao_por_via(df: pd.DataFrame) -> None:
    """Gera histogramas 2x3 com estatísticas descritivas de log10(LD50)."""
    print("  [EDA] Gerando histogramas de distribuição...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, via in enumerate(VIAS):
        col = Y_COLS.get(via)
        if col in df.columns:
            valores = df[col].dropna()
            if len(valores) > 0:
                sns.histplot(valores, bins=30, kde=True, ax=axes[i], color="#5DADE2")
                
                # Estatísticas
                stats = {
                    "Média": valores.mean(),
                    "Mediana": valores.median(),
                    "DP": valores.std(),
                    "Skew": valores.skew(),
                    "N": len(valores)
                }
                
                texto_stats = "\n".join([f"{k}: {v:.2f}" if k != "N" else f"{k}: {v}" for k, v in stats.items()])
                axes[i].text(0.95, 0.95, texto_stats, transform=axes[i].transAxes, 
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5), fontsize=9)
                
                axes[i].set_title(f"Distribuição: {via}", fontweight='bold')
                axes[i].set_xlabel("log10(LD50)")
                axes[i].set_ylabel("Frequência")
            else:
                axes[i].set_title(f"{via} (Sem dados)")

    plt.tight_layout()
    output_dir = PATHS["plots"] / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "histogramas_distribuicao.png", dpi=300)
    plt.close()

def plot_cobertura_matriz(df: pd.DataFrame) -> None:
    """Gera heatmap de esparsidade e salva tabela de cobertura."""
    print("  [EDA] Analisando esparsidade e cobertura...")
    
    cols_y = [Y_COLS[v] for v in VIAS if Y_COLS[v] in df.columns]
    df_y = df[cols_y]
    
    # Heatmap visual (amostra)
    plt.figure(figsize=(12, 8))
    amostra = df_y.sample(min(2000, len(df_y)), random_state=42)
    # Ordenar por preenchimento para visualização clara
    amostra = amostra.assign(n_present=amostra.notna().sum(axis=1)).sort_values('n_present', ascending=False).drop('n_present', axis=1)
    
    sns.heatmap(amostra.notna(), cmap="YlGnBu", cbar=False, yticklabels=False)
    plt.title("Mapa de Cobertura de Dados (Amostra N=2000)\nAzul = Medido | Amarelo = Ausente", fontsize=14)
    plt.xlabel("Vias de Administração")
    plt.ylabel("Compostos")
    plt.tight_layout()
    
    plt.savefig(PATHS["plots"] / "eda" / "heatmap_cobertura.png", dpi=300)
    plt.close()

    # Tabela de estatísticas de esparsidade
    n_vias = df_y.notna().sum(axis=1)
    resumo = {
        "Total de Moléculas": len(df),
        "Pelo menos 1 via": (n_vias >= 1).sum(),
        "Pelo menos 2 vias": (n_vias >= 2).sum(),
        "Pelo menos 3 vias": (n_vias >= 3).sum(),
        "Todas as 6 vias": (n_vias == 6).sum(),
    }
    
    for via, col in Y_COLS.items():
        if col in df.columns:
            n = df[col].notna().sum()
            resumo[f"N em {via}"] = n
            resumo[f"% preenchido {via}"] = f"{(n/len(df)*100):.2f}%"

    df_resumo = pd.DataFrame(list(resumo.items()), columns=["Métrica", "Valor"])
    df_resumo.to_excel(PATHS["tabelas"] / "EDA_Esparsidade_Matriz.xlsx", index=False)

def log_dl50_para_classe_ghs(log_dl50):
    """
    Categoria 1: ≤ 5 mg/kg  (log ≤ 0.699)
    Categoria 2: 5 < LD50 ≤ 50 (0.699 < log ≤ 1.699)
    Categoria 3: 50 < LD50 ≤ 300 (1.699 < log ≤ 2.477)
    Categoria 4: 300 < LD50 ≤ 2000 (2.477 < log ≤ 3.301)
    Categoria 5: > 2000 (log > 3.301)
    """
    dl50 = 10 ** log_dl50
    if dl50 <= 5: return 1
    if dl50 <= 50: return 2
    if dl50 <= 300: return 3
    if dl50 <= 2000: return 4
    return 5

def plot_classes_ghs(df: pd.DataFrame) -> None:
    """Gera barplot de frequências por categoria GHS e via."""
    print("  [EDA] Gerando distribuição de classes GHS (OECD)...")
    
    dados_ghs = []
    for via in VIAS:
        col = Y_COLS.get(via)
        if col in df.columns:
            valores = df[col].dropna()
            for v in valores:
                cat = log_dl50_para_classe_ghs(v)
                dados_ghs.append({"Via": via, "GHS_Category": cat})
    
    df_ghs = pd.DataFrame(dados_ghs)
    
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df_ghs, x="Via", hue="GHS_Category", palette="RdYlGn_r")
    plt.title("Distribuição de Compostos por Categorias GHS", fontsize=15, fontweight='bold')
    plt.xlabel("Via de Administração")
    plt.ylabel("Contagem de Compostos")
    plt.legend(title="Categoria GHS", loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(PATHS["plots"] / "eda" / "distribuicao_classes_ghs.png", dpi=300)
    plt.close()

def plot_sobreposicao_vias(df: pd.DataFrame) -> None:
    """Calcula interseção de compostos entre cada par de vias."""
    print("  [EDA] Gerando matriz de sobreposição...")
    
    cols = [Y_COLS[v] for v in VIAS if Y_COLS[v] in df.columns]
    n = len(cols)
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            mask = df[cols[i]].notna() & df[cols[j]].notna()
            matrix[i, j] = mask.sum()
            
    df_over = pd.DataFrame(matrix, index=VIAS, columns=VIAS)
    
    plt.figure(figsize=(10, 8))
    # Máscara para o triângulo superior
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(df_over, annot=True, fmt="d", cmap="Greens", mask=mask, cbar_kws={'label': 'N Compostos em Comum'})
    plt.title("Sobreposição Química (N Compostos Compartilhados)", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(PATHS["plots"] / "eda" / "sobreposicao_vias.png", dpi=300)
    plt.close()

def exportar_tabela_splits(df_base: pd.DataFrame) -> None:
    """Lê os índices de split e reporta composição por via."""
    print("  [EDA] Consolidando estatísticas de Split...")
    
    # Procurar o primeiro arquivo de split disponível (Scaffold é prioridade)
    metodos = ["scaffold", "butina", "random"]
    registros = []
    
    for met in metodos:
        pasta = PATHS.get(f"splits_{met}")
        if not pasta: continue
        
        arquivos = list(pasta.glob("*.npz"))
        if not arquivos: continue
        
        # Usamos o primeiro para amostragem do comportamento do split químico
        data = np.load(arquivos[0])
        train_idx = data["train_idx"]
        test_idx  = data["test_idx"]
        val_idx   = data.get("val_idx", np.array([]))
        
        for via in VIAS:
            col = Y_COLS.get(via)
            if col in df_base.columns:
                # Nativos (não-NaN) em cada set
                val_raw = df_base[col].values
                n_train = np.count_nonzero(~np.isnan(val_raw[train_idx]))
                n_test  = np.count_nonzero(~np.isnan(val_raw[test_idx]))
                n_val   = np.count_nonzero(~np.isnan(val_raw[val_idx])) if len(val_idx)>0 else 0
                
                registros.append({
                    "Método": met.capitalize(),
                    "Via": via,
                    "N_Treino": n_train,
                    "N_Validação": n_val,
                    "N_Teste": n_test,
                    "Total_Via": n_train + n_val + n_test
                })
    
    if registros:
        df_splits = pd.DataFrame(registros)
        df_splits.to_excel(PATHS["tabelas"] / "EDA_composicao_splits.xlsx", index=False)
        print(f"  ✓ Tabela de splits salva em {PATHS['tabelas']}")

def main():
    caminho_base = PATHS["preprocessed"] / "MTL_df_base_fundida.pkl"
    if not caminho_base.exists():
        print(f"  ✗ Erro: Base '{caminho_base}' não encontrada.")
        return

    print(f"\n{'='*60}\n  INICIANDO EDA - ToxiciTOOL 2.0\n{'='*60}")
    df = pd.read_pickle(caminho_base)
    
    plot_distribuicao_por_via(df)
    plot_cobertura_matriz(df)
    plot_classes_ghs(df)
    plot_sobreposicao_vias(df)
    exportar_tabela_splits(df)
    
    print("\n✓ Análise Exploratória concluída!")

if __name__ == "__main__":
    main()
