"""
treinamento_repetido.py  (src/modeling/)
==========================================
TODO 3 — Protocolo de N Treinamentos para Validação Estatística.

Executa o treinamento MTL múltiplas vezes com sementes diferentes para a
inicialização dos pesos, mantendo o split químico FIXO. Isso isola a
variância do modelo.

Saídas:
    - Excel com N execuções individuais
    - Excel com estatísticas agregadas (Média, DP, IC 95%)
    - Boxplots de desempenho por via
"""

import gc
import os
import sys
from pathlib import Path

# Configuração de Ambiente (Replica modelagem_MTL.py)
os.environ["TF_XLA_FLAGS"]  = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

_SRC = Path(__file__).resolve().parent.parent # (Pois os arquivos estão dentro de src/modeling/)
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import ESTATISTICA, PATHS, SPLIT, VIAS, Y_COLS
from src.modeling.modelagem_MTL import carregar_split, preparar_dados, modelo_multitask


def reset_seeds(seed: int):
    """Garante reprodutibilidade da inicialização dos pesos."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Para determinismo total em GPU (pode ser lento):
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'


def treinar_repetido(arquivo_pkl: Path, metodo_split: str = None, n_repeticoes: int = None):
    if n_repeticoes is None:
        n_repeticoes = ESTATISTICA.get("n_repeticoes", 30)

    print(f"\n{'='*60}")
    print(f"  INICIANDO PROTOCOLO ESTATÍSTICO: {n_repeticoes} execuções")
    print(f"  Arquivo: {arquivo_pkl.name}")
    print(f"  Split:   {metodo_split or 'padrão'}")
    print(f"{'='*60}")

    df_base = pd.read_pickle(arquivo_pkl)
    train_idx, val_idx, test_idx, metodo_split_final = carregar_split(arquivo_pkl, metodo_override=metodo_split)
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados(df_base, train_idx, val_idx, test_idx)
    
    # Economia de RAM: remove o DataFrame original após extrair as matrizes
    del df_base
    gc.collect()

    todas_metricas = []
    import time
    import psutil

    print(f"\n[START] Iniciando {n_repeticoes} iterações para o split {metodo_split_final}...")
    start_total = time.time()

    for i in range(n_repeticoes):
        seed = 100 + i
        iter_start = time.time()
        
        # Checkpoint de Memória RAM antes da iteração
        process = psutil.Process(os.getpid())
        mem_uso = process.memory_info().rss / (1024 ** 2) # MB
        
        print(f"\n{'─'*40}")
        print(f"  >>> ITERAÇÃO {i+1}/{n_repeticoes} | Seed: {seed}")
        print(f"  >>> RAM em uso: {mem_uso:.2f} MB")
        
        reset_seeds(seed)
        model = modelo_multitask(fpSize=X_train.shape[1])
        
        # Treinamento
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=64,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            ],
            verbose=0
        )

        # Avaliação
        predicoes = model.predict(X_test, verbose=0)
        
        run_res = {"iteracao": i + 1, "seed": seed}
        for j, (via, col) in enumerate(Y_COLS.items()):
            if via not in y_test: continue
            y_t, y_p = y_test[via], predicoes[j].flatten()
            mask = ~np.isnan(y_t)
            if mask.sum() > 1:
                y_t_l, y_p_l = y_t[mask], y_p[mask]
                run_res[f"r2_{via}"]  = r2_score(y_t_l, y_p_l)
                run_res[f"mae_{via}"] = mean_absolute_error(y_t_l, y_p_l)
        
        todas_metricas.append(run_res)
        
        # Log de performance da iteração
        r2_m = np.nanmean([v for k, v in run_res.items() if k.startswith("r2_")])
        iter_end = time.time()
        print(f"      ✓ Concluída em {iter_end - iter_start:.1f}s")
        print(f"      ✓ R² Médio: {r2_m:.3f}")

        # Limpeza de memória agressiva
        del model
        del predicoes
        tf.keras.backend.clear_session()
        gc.collect()

    end_total = time.time()
    print(f"\n[FINISH] Tempo total para {metodo_split_final}: {(end_total - start_total)/60:.2f} min")

    # Limpeza final dos dados de treino
    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    df_results = pd.DataFrame(todas_metricas)
    return df_results, metodo_split_final


def calcular_estatisticas(df_repeticoes: pd.DataFrame) -> pd.DataFrame:
    """Calcula Média, DP e Intervalo de Confiança 95%."""
    print("\n  [Stats] Agregando resultados...")
    
    metricas = [c for c in df_repeticoes.columns if c.startswith("r2_") or c.startswith("mae_")]
    stats_list = []

    for met in metricas:
        valores = df_repeticoes[met].dropna()
        if len(valores) == 0: continue
        
        media = valores.mean()
        std   = valores.std()
        n     = len(valores)
        # Erro padrão da média e IC 95% (t-Student)
        sem   = std / np.sqrt(n)
        ic95  = sem * stats.t.ppf((1 + 0.95) / 2., n-1)
        
        stats_list.append({
            "Métrica": met,
            "Média": media,
            "Desvio Padrão": std,
            "Erro Padrão": sem,
            "IC_95_Margem": ic95,
            "Mín": valores.min(),
            "Máx": valores.max(),
            "N": n
        })
        
    return pd.DataFrame(stats_list)


def plotar_boxplots(df_repeticoes: pd.DataFrame, stem: str, metodo: str):
    """Gera boxplots de R2 por via para visualizar a estabilidade."""
    print(f"  [Plot] Gerando boxplots de estabilidade para {metodo}...")
    
    # Prepara dados para o seaborn
    cols_r2 = [c for c in df_repeticoes.columns if c.startswith("r2_")]
    df_melt = df_repeticoes.melt(id_vars=["iteracao"], value_vars=cols_r2, 
                                 var_name="Via", value_name="R2")
    df_melt["Via"] = df_melt["Via"].str.replace("r2_", "")

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Boxplot + Stripplot para ver a distribuição individual
    ax = sns.boxplot(data=df_melt, x="Via", y="R2", palette="viridis", showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})
    sns.stripplot(data=df_melt, x="Via", y="R2", color=".3", size=4, alpha=0.3, jitter=True)
    
    # Adiciona anotações de mediana acima de cada box
    medians = df_melt.groupby(['Via'])['R2'].median()
    vias_sorted = [v.get_text() for v in ax.get_xticklabels()]
    for i, via in enumerate(vias_sorted):
        ax.text(i, medians[via] + 0.02, f'md={medians[via]:.3f}', 
                horizontalalignment='center', size='x-small', color='black', weight='semibold')

    plt.title(f"Estabilidade do Modelo MTL (N={len(df_repeticoes)})\nConfig: {stem} | Split: {metodo}", fontsize=14, pad=15)
    plt.xlabel("Via de Exposição / Modelo", fontsize=12)
    plt.ylabel("Coeficiente de Determinação ($R^2$)", fontsize=12)
    plt.ylim(0, 1) # Foco em performance positiva
    
    # Legenda customizada para os símbolos (fora do gráfico)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Média', markerfacecolor='white', markeredgecolor='black', markersize=8),
        Line2D([0], [0], color='gray', lw=2, label='Mediana'),
        Line2D([0], [0], marker='o', color='w', label='Execução Individual', markerfacecolor='gray', alpha=0.5, markersize=5)
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=True, fontsize='small')
    
    plt.tight_layout()
    
    out_path = PATHS["estatistica"] / f"boxplot_r2_repeticoes_{stem}__{metodo}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ Gráfico salvo: {out_path.name}")


def main():
    # Carrega dinamicamente o modelo campeão a partir do log gerado pela análise
    resumo_path = PATHS["melhor_modelo"] / "resumo_melhores_modelos.xlsx"
    if not resumo_path.exists():
        print(f"Erro: Arquivo de resumo não encontrado: {resumo_path}")
        print("Certifique-se de rodar a etapa de 'analise' primeiro.")
        return
        
    df_resumo = pd.read_excel(resumo_path)
    # Pega o primeiro modelo (Geral) do resumo e usa o arquivo de origem diretamente
    linha_campeao = df_resumo.iloc[0]
    nome_campeao = linha_campeao['arquivo_origem']
    
    arquivo = PATHS["preprocessed"] / nome_campeao
    
    if not arquivo.exists():
        print(f"Erro: Arquivo do campeão não encontrado: {nome_campeao}")
        return

    for metodo_alvo in ["scaffold", "butina"]:
        stem = arquivo.stem
        
        df_raw, metodo_final = treinar_repetido(arquivo, metodo_split=metodo_alvo)
        df_stats = calcular_estatisticas(df_raw)
        
        # Salvar Tabelas (usando a nova pasta de estatística/tabelas)
        path_raw   = PATHS["tabelas"] / f"MTL_Repeticoes_{stem}__{metodo_final}.xlsx"
        path_stats = PATHS["tabelas"] / f"MTL_Estatisticas_{stem}__{metodo_final}.xlsx"
        
        df_raw.to_excel(path_raw, index=False)
        df_stats.to_excel(path_stats, index=False)
        
        # Plots (usando a nova pasta de estatística)
        plotar_boxplots(df_raw, stem, metodo_final)
        
        print(f"\n{'─'*60}")
        print(f"  ✓ Protocolo Concluído para {metodo_final}!")
        print(f"  ✓ Dados brutos: {path_raw.name}")
        print(f"  ✓ Resumo estatístico: {path_stats.name}")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
