"""
validacao_estatistica.py  (src/analysis/)
==========================================
Testes estatísticos e de hipótese para validação do modelo MTL vencedor.

Testes implementados
────────────────────
1. Correlação  — Pearson (r), Spearman (ρ) e CCC com bootstrap CI 95 %
2. Wilcoxon    — MTL vs Baseline Nulo com correção Benjamini-Hochberg (BH)
3. Y-Scrambling — Randomização de targets para validação de correlação ao acaso
4. Williams Plot — Domínio de Aplicabilidade (AD) via Leverage vs Resíduos
5. Bland-Altman & Resíduos — Análise de viés e normalidade

Uso standalone
──────────────
    python src/analysis/validacao_estatistica.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Adiciona a RAIZ ao path
_SRC  = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.multitest import multipletests

from src.config import PATHS, VIAS, Y_COLS, RAIZ

# Diretórios de saída
_PLOTS_VAL   = PATHS["plots_validacao"]
_TABELAS_VAL = PATHS["tabelas_validacao"]
_PREPROCESSED = PATHS.get('preprocessed') or (RAIZ / 'data' / 'preprocessed')

# ============================================================
# UTILS E CARREGAMENTO
# ============================================================

def _masked_mse_local(y_true, y_pred):
    import tensorflow as tf
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)
    y_true_safe = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    sq_diff = tf.square(y_true_safe - y_pred)
    return tf.reduce_sum(sq_diff * mask, axis=-1) / (tf.reduce_sum(mask, axis=-1) + tf.keras.backend.epsilon())

def _derivar_caminhos(caminho_modelo: Path) -> dict:
    stem = caminho_modelo.stem
    pkl_stem, metodo_split = stem.rsplit('__', 1)
    pkl_path = _PREPROCESSED / f"{pkl_stem}.pkl"
    path_key = f"splits_{metodo_split.split('_')[0]}"
    idx_path = PATHS.get(path_key, RAIZ / "data" / "splits" / metodo_split) / f"{pkl_stem}__indices.npz"
    return {'pkl': pkl_path, 'idx': idx_path}

def carregar_modelo_e_dados(caminho_modelo: Path) -> dict:
    import tensorflow as tf
    caminhos = _derivar_caminhos(caminho_modelo)
    caminho_pkl, caminho_idx = caminhos['pkl'], caminhos['idx']

    df = pd.read_pickle(caminho_pkl)
    dados_idx = np.load(caminho_idx)
    train_idx, test_idx = dados_idx['train_idx'], dados_idx['test_idx']

    X = np.stack(df['Features'].tolist()).astype(np.float32)
    vias_disp = [v for v in VIAS if Y_COLS[v] in df.columns]
    
    from src.modeling.modelagem_MTL import CamadaIncerteza, get_huber_uncertainty_loss
    custom_objs = {'masked_mse': _masked_mse_local, 'CamadaIncerteza': CamadaIncerteza}
    for i in range(len(VIAS)):
        custom_objs[f"huber_uncert_task_{i}"] = get_huber_uncertainty_loss(i, None)

    model = tf.keras.models.load_model(caminho_modelo, custom_objects=custom_objs)
    preds_raw = model.predict(X[test_idx], verbose=0)
    
    return {
        'model': model, 'X_train': X[train_idx], 'X_test': X[test_idx],
        'y_train': {v: df[Y_COLS[v]].values[train_idx] for v in vias_disp},
        'y_test': {v: df[Y_COLS[v]].values[test_idx] for v in vias_disp},
        'predicoes': {via: preds_raw[i].flatten() for i, via in enumerate(VIAS)},
        'vias_disp': vias_disp
    }

# ============================================================
# ESTATÍSTICAS E TESTES
# ============================================================

def concordance_correlation_coefficient(y_true, y_pred):
    if len(y_true) < 2: return np.nan
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    return (2 * cor * sd_true * sd_pred) / (np.var(y_true) + np.var(y_pred) + (np.mean(y_true) - np.mean(y_pred))**2)

def bootstrap_ci(y_true, y_pred, metrica, n_boot=1000):
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        try: scores.append(metrica(y_true[idx], y_pred[idx]))
        except: pass
    return {'est': metrica(y_true, y_pred), 'low': np.percentile(scores, 2.5), 'high': np.percentile(scores, 97.5)}

def teste_correlacao_todas_vias(dados, n_boot=1000):
    res = []
    for via in dados['vias_disp']:
        yt, yp = dados['y_test'][via], dados['predicoes'][via]
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        yt, yp = yt[mask], yp[mask]
        
        c_r = bootstrap_ci(yt, yp, lambda a, b: stats.pearsonr(a, b)[0], n_boot)
        c_rho = bootstrap_ci(yt, yp, lambda a, b: stats.spearmanr(a, b)[0], n_boot)
        c_ccc = bootstrap_ci(yt, yp, concordance_correlation_coefficient, n_boot)
        c_r2 = bootstrap_ci(yt, yp, r2_score, n_boot)
        c_mae = bootstrap_ci(yt, yp, mean_absolute_error, n_boot)
        
        res.append({
            'Via de Exposição': via, 
            'Amostras (n)': len(yt), 
            'Pearson (r)': c_r['est'], 
            'IC 95% (Pearson)': f"[{c_r['low']:.3f}; {c_r['high']:.3f}]",
            'Spearman (ρ)': c_rho['est'], 
            'CCC': c_ccc['est'], 
            'R²': c_r2['est'],
            'MAE': c_mae['est']
        })
    return pd.DataFrame(res).set_index('Via de Exposição')

def teste_wilcoxon_todas_vias(dados):
    res = []
    for via in dados['vias_disp']:
        yt, yp = dados['y_test'][via], dados['predicoes'][via]
        mask = ~np.isnan(yt); yt, yp = yt[mask], yp[mask]
        media_tr = np.nanmean(dados['y_train'][via])
        err_mtl, err_null = np.abs(yt - yp), np.abs(yt - media_tr)
        stat, p = stats.wilcoxon(err_mtl, err_null, alternative='less')
        res.append({
            'Via de Exposição': via, 
            'MAE (MTL)': np.mean(err_mtl), 
            'MAE (Baseline)': np.mean(err_null), 
            'Melhoria (%)': 100 * (np.mean(err_null) - np.mean(err_mtl)) / np.mean(err_null),
            'p-valor': p
        })
    df = pd.DataFrame(res).set_index('Via de Exposição')
    _, p_adj, _, _ = multipletests(df['p-valor'], method='fdr_bh')
    df['p-valor (Adj. BH)'] = p_adj
    df['Significativo (α=0,05)'] = p_adj < 0.05
    return df

def teste_y_scrambling(dados, n_scrambles=10):
    from modeling.modelagem_MTL import modelo_multitask
    import tensorflow as tf
    vias = dados['vias_disp']
    r2_orig = {v: r2_score(dados['y_test'][v][~np.isnan(dados['y_test'][v])], dados['predicoes'][v][~np.isnan(dados['y_test'][v])]) for v in vias}
    
    scrambles = []
    for i in range(n_scrambles):
        y_tr_s = {v: np.random.permutation(dados['y_train'][v]) for v in vias}
        model = modelo_multitask(fpSize=dados['X_train'].shape[1])
        model.fit(dados['X_train'], y_tr_s, epochs=10, batch_size=128, verbose=0)
        preds = model.predict(dados['X_test'], verbose=0)
        row = {'iter': i}
        for j, v in enumerate(VIAS):
            if v in vias: row[f'r2_{v}'] = r2_score(dados['y_test'][v][~np.isnan(dados['y_test'][v])], preds[j].flatten()[~np.isnan(dados['y_test'][v])])
        scrambles.append(row)
        tf.keras.backend.clear_session()
    
    df_s = pd.DataFrame(scrambles)
    res = []
    for v in vias:
        p = (np.sum(df_s[f'r2_{v}'] >= r2_orig[v]) + 1) / (n_scrambles + 1)
        res.append({'Via de Exposição': v, 'R² Original': r2_orig[v], 'p-valor (Randomização)': p})
    df = pd.DataFrame(res).set_index('Via de Exposição')
    _, p_adj, _, _ = multipletests(df['p-valor (Randomização)'], method='fdr_bh')
    df['p-valor (Adj. BH)'] = p_adj
    return df

# ============================================================
# PLOTS
# ============================================================

def plot_williams(dados, split, salvar=True):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(dados['X_train'].shape[0], 50))
    xtr_pca = pca.fit_transform(dados['X_train'])
    xte_pca = pca.transform(dados['X_test'])
    xtx_inv = np.linalg.pinv(np.dot(xtr_pca.T, xtr_pca))
    levs = np.array([np.dot(np.dot(x, xtx_inv), x.T) for x in xte_pca])
    h_star = (3 * xtr_pca.shape[1]) / xtr_pca.shape[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, via in enumerate(VIAS):
        ax = axes.flatten()[i]
        if via not in dados['vias_disp']: continue
        yt, yp = dados['y_test'][via], dados['predicoes'][via]
        mask = ~np.isnan(yt); res = yp[mask]-yt[mask]; res_s = (res-np.mean(res))/np.std(res)
        ax.scatter(levs[mask], res_s, alpha=0.5, s=10)
        ax.axhline(3, color='r', ls='--'); ax.axhline(-3, color='r', ls='--')
        ax.axvline(h_star, color='b', ls='--')
        ax.set_title(f"Williams: {via}")
        ax.set_xlabel("Leverage (h)")
        ax.set_ylabel("Resíduo Padronizado")
    plt.tight_layout()
    if salvar: plt.savefig(_PLOTS_VAL / f"williams_{split}.png")
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    resumo_path = PATHS['melhor_modelo'] / 'resumo_melhores_modelos.xlsx'
    if not resumo_path.exists(): return print("Execute analise primeiro.")
    df_resumo = pd.read_excel(resumo_path)
    
    for metodo in ['scaffold', 'butina']:
        mask = df_resumo['metodo_split'] == metodo
        if not mask.any(): continue
        
        print(f"\n>>> VALIDANDO E FORMATANDO PARA PUBLICAÇÃO: {metodo.upper()}")
        caminho_mod = Path(str(df_resumo[mask].iloc[0]['caminho_modelo']))
        dados = carregar_modelo_e_dados(caminho_mod)
        
        df_corr = teste_correlacao_todas_vias(dados)
        df_wilcox = teste_wilcoxon_todas_vias(dados)
        df_scramble = teste_y_scrambling(dados)
        
        # Cria Tabela Resumo para Publicação (ABNT)
        df_pub = pd.DataFrame({
            'R²': df_corr['R²'],
            'MAE': df_corr['MAE'],
            'CCC': df_corr['CCC'],
            'p-valor (Wilcoxon)': df_wilcox['p-valor (Adj. BH)'],
            'p-valor (Scrambling)': df_scramble['p-valor (Adj. BH)'],
            'Significativo': df_wilcox['Significativo (α=0,05)']
        })
        
        path_xlsx = _TABELAS_VAL / f"Tabela_Estatistica_Publicacao_{metodo}.xlsx"
        with pd.ExcelWriter(path_xlsx) as writer:
            df_pub.to_excel(writer, sheet_name='Resumo_Publicacao_ABNT')
            df_corr.to_excel(writer, sheet_name='Correlacao_Detalhada')
            df_wilcox.to_excel(writer, sheet_name='Wilcoxon_Detalhado')
            df_scramble.to_excel(writer, sheet_name='Y_Scramble_Detalhado')
            
        plot_williams(dados, metodo)
        print(f"✓ Tabela ABNT salva: {path_xlsx.name}")

if __name__ == '__main__':
    main()
