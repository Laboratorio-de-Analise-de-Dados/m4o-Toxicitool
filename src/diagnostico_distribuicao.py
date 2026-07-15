
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Adiciona src ao path para importar config
sys.path.append(os.path.join(os.getcwd(), "src"))
from config import PATHS, VIAS

def plot_distributions():
    df = pd.read_pickle(PATHS['preprocessed'] / 'MTL_df_base_fundida.pkl')
    
    plt.figure(figsize=(18, 12))
    for i, via in enumerate(VIAS):
        col = f'log_dl50_{via}'
        data = df[col].dropna()
        
        plt.subplot(2, 3, i+1)
        sns.histplot(data, kde=True, color='skyblue')
        
        # Estatísticas básicas
        mean = data.mean()
        std = data.std()
        n = len(data)
        
        plt.axvline(mean, color='red', linestyle='--', label=f'Média: {mean:.2f}')
        plt.title(f'{via}\nn={n} | std={std:.2f}')
        plt.legend()

    plt.tight_layout()
    out_path = "results/plots/multitask/diagnostico_distribuicao.png"
    plt.savefig(out_path)
    print(f"Gráfico de diagnóstico salvo em: {out_path}")

    # Correlação entre tasks
    plt.figure(figsize=(10, 8))
    cols = [f'log_dl50_{v}' for v in VIAS]
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlação entre Log DL50 de diferentes Vias")
    plt.tight_layout()
    plt.savefig("results/plots/multitask/diagnostico_correlacao.png")
    print(f"Heatmap de correlação salvo em: results/plots/multitask/diagnostico_correlacao.png")

if __name__ == "__main__":
    plot_distributions()
