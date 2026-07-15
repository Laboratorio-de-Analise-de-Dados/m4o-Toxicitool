"""
validacao_externa.py  (src/analysis/)
=======================================
Validação do modelo MTL em compostos externos (literatura / bases públicas).

Fluxo
─────
1. Lê um CSV externo com colunas SMILES e LD50 (mg/kg) por via
2. Aplica o mesmo pipeline de canonização + fingerprints da etapa 2
3. Verifica sobreposição com o conjunto de treino via Tanimoto
   (compostos com similaridade máxima ≥ 0.6 são marcados como "dentro do domínio")
4. Prediz com o modelo carregado
5. Calcula métricas e gera plots

Formato esperado do CSV externo
────────────────────────────────
Colunas obrigatórias:
    smiles      — SMILES canônico ou bruto da molécula
    ld50_<via>  — LD50 em mg/kg para cada via disponível
                  (ex: ld50_rat_vo, ld50_mouse_ip)
    nome        — (opcional) identificador textual

Exemplo de uso
──────────────
    from analysis.validacao_externa import validar_externos
    resultado = validar_externos(
        csv_path='data/externos/compostos_literatura.csv',
        caminho_modelo=None,   # usa melhor_modelo automaticamente
    )

Configuração: src/config.py  (PATHS, VIAS, Y_COLS, RAIZ)
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

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.config import PATHS, VIAS, Y_COLS, RAIZ
from src.preprocessing.Limpeza import Limpeza
from src.representation.Representacao import Representacao
from src.analysis.validacao_estatistica import (
    _masked_mse_local,
    _PLOTS_VAL,
    _TABELAS_VAL,
    bootstrap_ci,
)

# ── Caminho para pkl de fingerprints (para extrair parâmetros de FP) ───
_PREPROCESSED: Path = PATHS.get('preprocessed') or (RAIZ / 'data' / 'preprocessed')

# ── Tanimoto similarity threshold para domínio de aplicabilidade ───────
TANIMOTO_DOMINIO = 0.6   # compostos com sim >= 0.6 são "no domínio"


# ============================================================
# LEITURA E PADRONIZAÇÃO
# ============================================================

def ler_csv_externo(csv_path: Path, sep: str = ',') -> pd.DataFrame:
    """
    Lê o CSV externo e valida as colunas mínimas.

    A coluna 'smiles' é obrigatória. Colunas 'ld50_<via>' são detectadas
    automaticamente — ao menos uma deve existir.

    Retorna DataFrame com colunas: smiles, nome (se existir), ld50_<via>...
    """
    df = pd.read_csv(csv_path, sep=sep)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    if 'smiles' not in df.columns:
        raise ValueError(
            "CSV externo deve ter coluna 'smiles'.\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    # Detecta colunas de LD50
    cols_ld50 = [c for c in df.columns if c.startswith('ld50_')]
    if not cols_ld50:
        raise ValueError(
            "Nenhuma coluna 'ld50_<via>' encontrada no CSV.\n"
            "Esperado: ld50_rat_vo, ld50_mouse_ip, etc.\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    print(f"  CSV externo: {len(df)} compostos | vias: {cols_ld50}")
    return df


def converter_para_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas ld50_<via> de mg/kg para log10.
    Cria colunas log_dl50_<via> e remove ld50 bruto.
    """
    df = df.copy()
    for col in [c for c in df.columns if c.startswith('ld50_')]:
        via_nome = col.replace('ld50_', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[f'log_dl50_{via_nome}'] = np.where(
            df[col] > 0, np.log10(df[col]), np.nan
        )
    return df


def canonizar_externos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o mesmo pipeline de canonização da Limpeza ao CSV externo.
    SMILES inválidos viram NaN e são removidos.
    """
    print("  Canonizando SMILES externos...")
    n_antes = len(df)

    limpeza = Limpeza(dataframe=df[['smiles']].copy())
    df_canon = limpeza.canonical_smiles(col_smiles='smiles', sanitize=True)
    df['smiles'] = df_canon['smiles'].values

    df.dropna(subset=['smiles'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  {n_antes} → {len(df)} compostos após canonização")
    return df


# ============================================================
# FINGERPRINTS
# ============================================================

def gerar_fingerprints_externos(
    df:      pd.DataFrame,
    radius:  int = 2,
    fpSize:  int = 2048,
) -> pd.DataFrame:
    """
    Gera Morgan fingerprints para os compostos externos.

    Os parâmetros radius e fpSize devem corresponder aos do modelo carregado.
    Eles podem ser lidos do nome do pkl:
        MTL_df_final_binario_2048bits_raio2.pkl → fpSize=2048, radius=2
    """
    print(f"  Gerando fingerprints (radius={radius}, fpSize={fpSize})...")
    rep = Representacao(dataframe=df)
    df  = rep.fingerprint(col_smiles='smiles', radius=radius, fpSize=fpSize)
    df.dropna(subset=['Features'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df)} compostos com fingerprint válido")
    return df


def extrair_params_do_pkl(caminho_pkl: Path) -> dict:
    """
    Infere radius e fpSize a partir do nome do pkl.
    Padrão: MTL_df_final_{tipo}_{fpSize}bits_raio{radius}.pkl
    """
    stem = caminho_pkl.stem  # ex: MTL_df_final_binario_2048bits_raio2
    try:
        # fpSize
        bits_part = [p for p in stem.split('_') if 'bits' in p][0]
        fpSize = int(bits_part.replace('bits', ''))
        # radius
        raio_part = [p for p in stem.split('_') if p.startswith('raio')][0]
        radius = int(raio_part.replace('raio', ''))
        return {'radius': radius, 'fpSize': fpSize}
    except Exception:
        print(f"  ⚠️  Não foi possível inferir parâmetros de '{stem}'. Usando radius=2, fpSize=2048.")
        return {'radius': 2, 'fpSize': 2048}


# ============================================================
# DOMÍNIO DE APLICABILIDADE
# ============================================================

def verificar_dominio_aplicabilidade(
    df_externos:  pd.DataFrame,
    smiles_treino: np.ndarray,
    radius:        int = 2,
    fpSize:        int = 2048,
    threshold:     float = TANIMOTO_DOMINIO,
) -> pd.DataFrame:
    """
    Para cada composto externo, calcula a similaridade Tanimoto máxima
    com o conjunto de treino.

    Compostos com sim_max >= threshold estão "no domínio" — predições
    são mais confiáveis.
    Compostos com sim_max < threshold estão "fora do domínio" — predições
    devem ser interpretadas com cautela.

    Adiciona colunas:
        sim_max_treino — similaridade máxima com qualquer composto de treino
        no_dominio     — True / False
    """
    print(f"  Calculando domínio de aplicabilidade (threshold={threshold})...")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

    def to_fp(smi):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            return gen.GetFingerprint(mol) if mol else None
        except Exception:
            return None

    fps_ext   = [to_fp(s) for s in df_externos['smiles']]
    fps_train = [fp for fp in [to_fp(s) for s in smiles_treino] if fp is not None]

    print(f"  Comparando {len(fps_ext)} compostos externos vs {len(fps_train)} de treino...")
    sim_max = []
    for fp in fps_ext:
        if fp is None:
            sim_max.append(0.0)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
            sim_max.append(float(max(sims)) if sims else 0.0)

    df_ext = df_externos.copy()
    df_ext['sim_max_treino'] = sim_max
    df_ext['no_dominio']     = df_ext['sim_max_treino'] >= threshold

    n_dentro = df_ext['no_dominio'].sum()
    pct = 100 * n_dentro / len(df_ext)
    print(f"  No domínio (sim ≥ {threshold}): {n_dentro}/{len(df_ext)} ({pct:.1f}%)")
    return df_ext


# ============================================================
# PREDIÇÃO E AVALIAÇÃO
# ============================================================

def predizer_externos(
    model,
    df_externos: pd.DataFrame,
) -> dict:
    """
    Gera predições com o modelo Keras para os compostos externos.
    ``df_externos`` deve ter coluna 'Features'.

    Retorna dict {via: array de predições} — mesmo formato de validacao_estatistica.
    """
    X = np.stack(df_externos['Features'].tolist()).astype(np.float32)
    preds_raw = model.predict(X, verbose=0)
    predicoes = {via: preds_raw[i].flatten() for i, via in enumerate(VIAS)}
    return predicoes


def avaliar_externos(
    df_externos: pd.DataFrame,
    predicoes:   dict,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """
    Compara predições vs valores reais (log_dl50_<via>) dos compostos externos.

    Retorna DataFrame de métricas com R², MAE e bootstrap CI 95 %.
    Opcionalmente filtra por domínio de aplicabilidade se coluna existir.
    """
    resultados = []
    no_dominio = df_externos.get('no_dominio', pd.Series([True] * len(df_externos)))

    for via in VIAS:
        col_log = f'log_dl50_{via}'
        if col_log not in df_externos.columns:
            continue

        y_true = df_externos[col_log].values
        y_pred = predicoes[via]
        mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)

        if mask.sum() < 3:
            continue

        yt_all, yp_all  = y_true[mask], y_pred[mask]
        dom_mask         = mask & no_dominio.values.astype(bool)

        res = dict(via=via, n_total=int(mask.sum()), n_no_dominio=int(dom_mask.sum()))

        for sufixo, yt, yp in [
            ('_todos',     yt_all, yp_all),
            ('_dominio',   y_true[dom_mask], y_pred[dom_mask])
        ]:
            n = len(yt)
            if n < 3:
                continue
            r2  = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            ci_r2  = bootstrap_ci(yt, yp, r2_score,             n_bootstrap)
            ci_mae = bootstrap_ci(yt, yp, mean_absolute_error,  n_bootstrap)
            res.update({
                f'r2{sufixo}'          : round(r2,  4),
                f'r2_ci_low{sufixo}'   : round(ci_r2['ci_low'],  4),
                f'r2_ci_high{sufixo}'  : round(ci_r2['ci_high'], 4),
                f'mae{sufixo}'         : round(mae, 4),
                f'mae_ci_low{sufixo}'  : round(ci_mae['ci_low'],  4),
                f'mae_ci_high{sufixo}' : round(ci_mae['ci_high'], 4),
            })

        resultados.append(res)

    df_metr = pd.DataFrame(resultados).set_index('via')
    print("\n── Métricas Validação Externa ──────────────────────────────")
    cols_show = [c for c in df_metr.columns if 'r2_todos' in c or 'mae_todos' in c or 'n_' in c]
    print(df_metr[cols_show].to_string())
    return df_metr


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plot_scatter_externo(
    df_externos: pd.DataFrame,
    predicoes:   dict,
    salvar:      bool = True,
) -> plt.Figure:
    """
    Grid 2×3 de scatter real vs predito para os compostos externos.
    Compostos no domínio aparecem em azul; fora do domínio em laranja.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    no_dom = df_externos.get('no_dominio', pd.Series([True] * len(df_externos))).values

    for i, via in enumerate(VIAS):
        ax     = axes[i]
        col    = f'log_dl50_{via}'
        if col not in df_externos.columns:
            ax.set_title(f"{via}\nNão disponível"); ax.axis('off'); continue

        y_true = df_externos[col].values
        y_pred = predicoes[via]
        mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)

        if mask.sum() < 2:
            ax.set_title(f"{via}\nDados insuficientes"); continue

        yt, yp = y_true[mask], y_pred[mask]
        dom    = no_dom[mask]

        ax.scatter(yt[dom],  yp[dom],  alpha=0.7, s=40, color='steelblue', label='No domínio',    edgecolors='k', linewidths=0.3)
        ax.scatter(yt[~dom], yp[~dom], alpha=0.5, s=40, color='orange',    label='Fora do domínio',edgecolors='k', linewidths=0.3, marker='^')

        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], 'gray', ls='--', lw=1)

        if len(yt) > 1:
            r2  = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            ax.set_title(f'{via}  (n={mask.sum()})\nR²={r2:.3f} | MAE={mae:.3f}')
        ax.set_xlabel('log₁₀(LD50) real (literatura)')
        ax.set_ylabel('log₁₀(LD50) predito')
        ax.legend(fontsize=7)

    fig.suptitle('Validação Externa — Compostos da Literatura', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if salvar:
        caminho = _PLOTS_VAL / 'validacao_externa_scatter.png'
        plt.savefig(caminho, dpi=300)
        print(f"✓ Scatter externo salvo: {caminho}")

    return fig


def plot_dominio_distribuicao(df_externos: pd.DataFrame, salvar: bool = True) -> plt.Figure:
    """
    Histograma da similaridade Tanimoto máxima com o treino,
    com linha vertical no threshold de domínio.
    """
    if 'sim_max_treino' not in df_externos.columns:
        print("⚠️  Coluna 'sim_max_treino' não encontrada. Execute verificar_dominio_aplicabilidade() primeiro.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_externos['sim_max_treino'], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(TANIMOTO_DOMINIO, color='red', lw=2, ls='--', label=f'Threshold = {TANIMOTO_DOMINIO}')
    n_dom = df_externos['no_dominio'].sum()
    ax.set_xlabel('Similaridade Tanimoto máxima (vs. treino)')
    ax.set_ylabel('Frequência')
    ax.set_title(
        f'Domínio de Aplicabilidade\n'
        f'{n_dom}/{len(df_externos)} compostos no domínio ({100*n_dom/len(df_externos):.1f}%)'
    )
    ax.legend()
    plt.tight_layout()

    if salvar:
        caminho = _PLOTS_VAL / 'dominio_aplicabilidade.png'
        plt.savefig(caminho, dpi=300)
        print(f"✓ Domínio salvo: {caminho}")

    return fig


# ============================================================
# FUNÇÃO PRINCIPAL INTEGRADA
# ============================================================

def validar_externos(
    csv_path:        str | Path,
    caminho_modelo:  Optional[Path] = None,
    smiles_treino:   Optional[np.ndarray] = None,
    n_bootstrap:     int = 500,
) -> dict:
    """
    Pipeline completo de validação externa.

    Parâmetros
    ──────────
    csv_path       — caminho para o CSV com compostos externos
    caminho_modelo — .keras do modelo (None = usa melhor_modelo)
    smiles_treino  — array de SMILES do treino para o AD; se None,
                     tenta recuperar do pkl correspondente ao modelo
    n_bootstrap    — re-amostragens para CI

    Retorna
    ───────
    dict com: df_externos, predicoes, metricas, fig_scatter, fig_dominio
    """
    import tensorflow as tf
    from src.analysis.validacao_estatistica import _derivar_caminhos

    print("=" * 60)
    print("  ToxiciTOOL — Validação Externa")
    print("=" * 60)

    # ── Localizar modelo ────────────────────────────────────────────
    if caminho_modelo is None:
        resumo_path = PATHS['melhor_modelo'] / 'resumo_melhores_modelos.xlsx'
        resumo      = pd.read_excel(resumo_path, index_col=0)
        caminho_modelo = Path(str(resumo.loc['caminho_modelo', 'valor']))
    print(f"  Modelo: {caminho_modelo.name}")

    # ── Parâmetros de fingerprint ───────────────────────────────────
    caminhos   = _derivar_caminhos(caminho_modelo)
    params_fp  = extrair_params_do_pkl(caminhos['pkl'])
    radius, fpSize = params_fp['radius'], params_fp['fpSize']

    # ── Carregar modelo ─────────────────────────────────────────────
    model = tf.keras.models.load_model(
        caminho_modelo,
        custom_objects={'masked_mse': _masked_mse_local},
    )

    # ── Recuperar SMILES de treino para AD ──────────────────────────
    if smiles_treino is None and caminhos['pkl'].exists():
        df_train_full = pd.read_pickle(caminhos['pkl'])
        dados_idx     = np.load(caminhos['idx'])
        smiles_treino = df_train_full['smiles'].values[dados_idx['train_idx']]
        print(f"  SMILES de treino carregados: {len(smiles_treino)}")

    # ── Processar compostos externos ────────────────────────────────
    csv_path = Path(csv_path)
    df_ext   = ler_csv_externo(csv_path)
    df_ext   = converter_para_log(df_ext)
    df_ext   = canonizar_externos(df_ext)
    df_ext   = gerar_fingerprints_externos(df_ext, radius=radius, fpSize=fpSize)

    # ── Domínio de aplicabilidade ───────────────────────────────────
    if smiles_treino is not None:
        df_ext = verificar_dominio_aplicabilidade(
            df_ext, smiles_treino, radius=radius, fpSize=fpSize
        )

    # ── Predição e métricas ─────────────────────────────────────────
    predicoes = predizer_externos(model, df_ext)
    metricas  = avaliar_externos(df_ext, predicoes, n_bootstrap=n_bootstrap)

    # ── Salvar métricas ─────────────────────────────────────────────
    caminho_excel = _TABELAS_VAL / 'MTL_Validacao_Externa.xlsx'
    metricas.to_excel(caminho_excel)
    print(f"\n✓ Métricas externas salvas: {caminho_excel}")

    # ── Plots ───────────────────────────────────────────────────────
    fig_scatter = plot_scatter_externo(df_ext, predicoes)
    fig_dominio = plot_dominio_distribuicao(df_ext) if 'sim_max_treino' in df_ext.columns else None

    # ── Salvar CSV com predições ────────────────────────────────────
    for via in VIAS:
        df_ext[f'pred_log_dl50_{via}'] = predicoes[via]
    saida_csv = _TABELAS_VAL / f'{csv_path.stem}_predicoes.csv'
    df_ext.drop(columns=['Features'], errors='ignore').to_csv(saida_csv, index=False)
    print(f"✓ Predições salvas: {saida_csv}")

    print("\n✅ Validação externa concluída!")

    return dict(
        df_externos  = df_ext,
        predicoes    = predicoes,
        metricas     = metricas,
        fig_scatter  = fig_scatter,
        fig_dominio  = fig_dominio,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Exemplo de uso com arquivo CSV externo.
    Edite o caminho abaixo para apontar para seus dados externos.
    """
    csv_exemplo = RAIZ / 'data' / 'externos' / 'compostos_literatura.csv'

    if not csv_exemplo.exists():
        print(f"⚠️  CSV externo não encontrado: {csv_exemplo}")
        print("   Crie o arquivo com colunas: smiles, ld50_<via>, nome (opcional)")
        print("   Exemplo de linha:")
        print("     smiles,ld50_rat_vo,nome")
        print("     CCO,7060,Etanol")
        return

    resultado = validar_externos(csv_path=csv_exemplo)
    print("\nMétricas externas:")
    print(resultado['metricas'])


if __name__ == '__main__':
    main()
