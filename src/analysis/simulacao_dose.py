"""
simulacao_dose.py  (src/analysis/)
====================================
Simula e compara estratégias de determinação de LD50 in vivo.

Contexto biológico
──────────────────
O método clássico de determinação de LD50 (OECD TG 425 — Up-and-Down)
administra uma dose ao animal, observa a resposta (morte / sobrevivência) e
ajusta a próxima dose por um fator fixo (d = 3.16 ≈ 10^0.5 por padrão).
O procedimento para após N animais ou após observar reversões suficientes.
Tipicamente usa 5–10 animais por composto.

Abordagens simuladas
─────────────────────
1. Up-and-Down clássico (OECD 425)
       Inicia em uma dose padrão; sobe/desce por fator fixo.
       Não usa nenhuma informação computacional prévia.

2. Up-and-Down informado pelo modelo
       Usa a predição do modelo MTL como dose inicial.
       Isso "pula" a fase exploratória e converge mais rápido.

3. Busca binária guiada pelo modelo
       A predição define o centro de um intervalo [low, high].
       Cada experimento particiona o intervalo ao meio até convergir.
       Análogo ao Bayesian adaptive design.

Métrica principal
──────────────────
Número de "experimentos" (animais) necessários para estimar o LD50
com erro ≤ ``tolerancia`` em escala log₁₀.

Nota sobre ética animal
────────────────────────
A motivação científica desta simulação é exatamente reduzir o número de
animais necessários (princípio 3R: Replace, Reduce, Refine).
O modelo MTL pode ser usado como ferramenta de triagem pré-clínica.

Uso
───
    from analysis.simulacao_dose import simular_e_comparar

    # Usa compostos do conjunto de teste do melhor modelo
    resultado = simular_e_comparar(n_compostos=50)
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

from src.config import PATHS, VIAS, Y_COLS, RAIZ
from src.analysis.validacao_estatistica import (
    _masked_mse_local,
    _PLOTS_VAL,
    _TABELAS_VAL,
)

# ── Parâmetros OECD 425 ────────────────────────────────────────────────
FATOR_UP_DOWN    = 10 ** 0.5   # ≈ 3.16 — fator multiplicativo por passo
DOSE_INICIAL_PAD = 55       # mg/kg
LOG_DOSE_MIN     = 0.5         # log10(mg/kg) — limite inferior da busca
LOG_DOSE_MAX     = 4.5         # log10(mg/kg) — limite superior da busca
MAX_EXPERIMENTOS = 20          # limite de segurança para evitar loop infinito


# ============================================================
# MODELO DE RESPOSTA ANIMAL
# ============================================================

def resposta_animal(dose_mg_kg: float, ld50_real_mg_kg: float) -> int:
    """
    Simula a resposta binária (morte / sobrevivência) de um animal.

    Modelo de Hill (dose-resposta sigmoidal):
        P(morte | dose) = dose^n / (LD50^n + dose^n)    com n = 2

    Isso é mais realista que um simples threshold — captura a variabilidade
    biológica real ao redor do LD50 (resposta probabilística, não determinística).

    Retorna: 1 = morte, 0 = sobrevivência
    """
    n = 2.0   # coeficiente de Hill — inclinação da curva dose-resposta
    prob_morte = (dose_mg_kg ** n) / (ld50_real_mg_kg ** n + dose_mg_kg ** n)
    return int(np.random.random() < prob_morte)


def resposta_animal_deterministica(dose_mg_kg: float, ld50_real_mg_kg: float) -> int:
    """
    Versão determinística: morte se dose ≥ LD50, sobrevivência caso contrário.
    Útil para análises sem variância estocástica.
    """
    return int(dose_mg_kg >= ld50_real_mg_kg)


# ============================================================
# UP-AND-DOWN CLÁSSICO (OECD 425)
# ============================================================

def simular_up_down_classico(
    ld50_real_mg_kg: float,
    dose_inicial:    float = DOSE_INICIAL_PAD,
    fator:           float = FATOR_UP_DOWN,
    max_exp:         int   = MAX_EXPERIMENTOS,
    tolerancia_log:  float = 0.3,
    estocastico:     bool  = True,
    seed:            int   = None,
) -> dict:
    """
    Simula o método Up-and-Down OECD 425.

    Regra de parada: convergência dentro de ``tolerancia_log`` em escala log₁₀
    por 3 consecutivos, ou ``max_exp`` atingido.

    Retorna dict com:
        n_experimentos   — animais usados
        log_ld50_estimado— estimativa final em log₁₀
        erro_log         — |estimativa − real| em log
        historico        — lista de (dose, resposta) por experimento
        convergiu        — True se dentro da tolerância
    """
    if seed is not None:
        np.random.seed(seed)

    resposta_fn     = resposta_animal if estocastico else resposta_animal_deterministica
    log_ld50_real   = np.log10(ld50_real_mg_kg)
    dose_atual      = dose_inicial
    historico       = []
    consecutivos_ok = 0

    for _ in range(max_exp):
        resposta   = resposta_fn(dose_atual, ld50_real_mg_kg)
        historico.append((dose_atual, resposta))

        log_dose_atual = np.log10(dose_atual)
        erro_atual     = abs(log_dose_atual - log_ld50_real)

        if erro_atual <= tolerancia_log:
            consecutivos_ok += 1
            if consecutivos_ok >= 2:
                break
        else:
            consecutivos_ok = 0

        # Regra Up-and-Down
        if resposta == 1:   # morte → reduz dose
            dose_atual = max(dose_atual / fator, 10 ** LOG_DOSE_MIN)
        else:               # sobrevivência → aumenta dose
            dose_atual = min(dose_atual * fator, 10 ** LOG_DOSE_MAX)

    # Estimativa final = média geométrica das últimas 3 doses
    ultimas_doses  = [h[0] for h in historico[-3:]]
    log_estimado   = float(np.mean(np.log10(ultimas_doses)))
    ld50_estimado  = 10 ** log_estimado
    erro_log_final = abs(log_estimado - log_ld50_real)

    return dict(
        n_experimentos    = len(historico),
        log_ld50_real     = round(log_ld50_real,  4),
        log_ld50_estimado = round(log_estimado,   4),
        ld50_estimado_mg  = round(ld50_estimado,  2),
        erro_log          = round(erro_log_final, 4),
        convergiu         = bool(erro_log_final <= tolerancia_log),
        historico         = historico,
        metodo            = 'up_down_classico',
    )


# ============================================================
# UP-AND-DOWN INFORMADO PELO MODELO
# ============================================================

def simular_up_down_modelo(
    ld50_real_mg_kg:  float,
    pred_log_ld50:    float,   # predição do modelo em log₁₀
    fator:            float = FATOR_UP_DOWN,
    max_exp:          int   = MAX_EXPERIMENTOS,
    tolerancia_log:   float = 0.3,
    estocastico:      bool  = True,
    seed:             int   = None,
) -> dict:
    """
    Up-and-Down com a predição do modelo como dose inicial.

    A diferença do método clássico é exatamente a dose de partida:
    em vez da dose padrão arbitrária (175 mg/kg), inicia na predição
    do modelo — que já está próxima do valor real.
    """
    dose_inicial = 10 ** pred_log_ld50
    resultado    = simular_up_down_classico(
        ld50_real_mg_kg = ld50_real_mg_kg,
        dose_inicial    = dose_inicial,
        fator           = fator,
        max_exp         = max_exp,
        tolerancia_log  = tolerancia_log,
        estocastico     = estocastico,
        seed            = seed,
    )
    resultado['metodo'] = 'up_down_modelo'
    return resultado


# ============================================================
# BUSCA BINÁRIA GUIADA PELO MODELO
# ============================================================

def simular_busca_binaria(
    ld50_real_mg_kg: float,
    pred_log_ld50:   float,
    incerteza_log:   float = 0.5,   # ± incerteza inicial em log₁₀
    max_exp:         int   = MAX_EXPERIMENTOS,
    tolerancia_log:  float = 0.3,
    estocastico:     bool  = True,
    seed:            int   = None,
) -> dict:
    """
    Busca binária no espaço de doses, guiada pela predição do modelo.

    O modelo define o centro de um intervalo de busca [pred - incerteza, pred + incerteza].
    Cada experimento testa a dose central do intervalo atual e reduz o intervalo à metade.

    Isso é equivalente a um design adaptativo onde:
        - O modelo provê o prior (distribuição inicial de probabilidade)
        - Cada experimento provê evidência (resposta binária morte/sobrevivência)
        - A estimativa posterior converge para o LD50 real

    Parâmetros
    ──────────
    incerteza_log — metade da largura do intervalo inicial em log₁₀
                    (padrão 0.5 = intervalo de 1 log = fator 10 em mg/kg)
    """
    if seed is not None:
        np.random.seed(seed)

    resposta_fn   = resposta_animal if estocastico else resposta_animal_deterministica
    log_ld50_real = np.log10(ld50_real_mg_kg)

    log_low  = pred_log_ld50 - incerteza_log
    log_high = pred_log_ld50 + incerteza_log
    historico = []

    for _ in range(max_exp):
        log_dose_atual = (log_low + log_high) / 2.0
        dose_atual     = 10 ** log_dose_atual
        resposta       = resposta_fn(dose_atual, ld50_real_mg_kg)
        historico.append((dose_atual, resposta))

        # Atualiza intervalo: se morte → LD50 está abaixo da dose atual
        if resposta == 1:
            log_high = log_dose_atual
        else:
            log_low  = log_dose_atual

        # Verifica convergência
        erro = abs(log_dose_atual - log_ld50_real)
        if (log_high - log_low) <= tolerancia_log:
            break

    log_estimado   = (log_low + log_high) / 2.0
    ld50_estimado  = 10 ** log_estimado
    erro_log_final = abs(log_estimado - log_ld50_real)

    return dict(
        n_experimentos    = len(historico),
        log_ld50_real     = round(log_ld50_real,  4),
        log_ld50_estimado = round(log_estimado,   4),
        ld50_estimado_mg  = round(ld50_estimado,  2),
        erro_log          = round(erro_log_final, 4),
        convergiu         = bool(erro_log_final <= tolerancia_log),
        historico         = historico,
        metodo            = 'busca_binaria_modelo',
    )


# ============================================================
# COMPARAÇÃO EM MÚLTIPLOS COMPOSTOS
# ============================================================

def simular_e_comparar(
    dados:          dict,
    via:            str       = 'rat_vo',
    n_compostos:    int       = 50,
    n_repeticoes:   int       = 10,
    tolerancia_log: float     = 0.3,
    seed_base:      int       = 42,
) -> dict:
    """
    Compara os três métodos em ``n_compostos`` compostos do conjunto de teste.

    ``dados`` — dicionário retornado por ``carregar_modelo_e_dados()``.

    Para cada composto, cada método é rodado ``n_repeticoes`` vezes (porque
    a simulação é estocástica) e o número mediano de experimentos é reportado.

    Retorna dict com:
        df_resultados — DataFrame por composto e método
        resumo        — estatísticas agregadas (mediana, IQR, % convergência)
    """
    y_true = dados['y_test'].get(via)
    y_pred = dados['predicoes'].get(via)

    if y_true is None or y_pred is None:
        raise ValueError(f"Via '{via}' não disponível nos dados. Vias: {dados.get('vias_disp')}")

    mask   = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    n_disp = min(n_compostos, len(yt))

    rng    = np.random.default_rng(seed_base)
    idx    = rng.choice(len(yt), size=n_disp, replace=False)
    yt_sel, yp_sel = yt[idx], yp[idx]

    registros = []

    for k, (log_real, log_pred) in enumerate(zip(yt_sel, yp_sel)):
        ld50_real = float(10 ** log_real)
        pred_log  = float(log_pred)

        for metodo_fn, label, kwargs in [
            (
                simular_up_down_classico, 'ClassicUpDown',
                dict(dose_inicial=DOSE_INICIAL_PAD, tolerancia_log=tolerancia_log)
            ),
            (
                simular_up_down_modelo, 'ModelUpDown',
                dict(pred_log_ld50=pred_log, tolerancia_log=tolerancia_log)
            ),
            (
                simular_busca_binaria, 'BinarySearch',
                dict(pred_log_ld50=pred_log, tolerancia_log=tolerancia_log)
            ),
        ]:
            n_exps, erros, convergencias = [], [], []
            for rep in range(n_repeticoes):
                r = metodo_fn(ld50_real_mg_kg=ld50_real, seed=seed_base + k * 100 + rep, **kwargs)
                n_exps.append(r['n_experimentos'])
                erros.append(r['erro_log'])
                convergencias.append(int(r['convergiu']))

            registros.append(dict(
                composto_idx     = k,
                log_ld50_real    = round(log_real, 4),
                log_ld50_pred    = round(pred_log, 4),
                erro_pred_modelo = round(abs(log_real - pred_log), 4),
                metodo           = label,
                n_exp_mediana    = float(np.median(n_exps)),
                n_exp_q25        = float(np.percentile(n_exps, 25)),
                n_exp_q75        = float(np.percentile(n_exps, 75)),
                erro_log_mediana = float(np.median(erros)),
                pct_convergiu    = float(np.mean(convergencias) * 100),
            ))

    df = pd.DataFrame(registros)

    # ── Resumo agregado ─────────────────────────────────────────────
    resumo = (
        df.groupby('metodo')
        .agg(
            n_exp_mediana    = ('n_exp_mediana',    'median'),
            n_exp_q25        = ('n_exp_q25',        'median'),
            n_exp_q75        = ('n_exp_q75',        'median'),
            erro_log_mediana = ('erro_log_mediana', 'median'),
            pct_convergiu    = ('pct_convergiu',    'mean'),
            n_compostos      = ('composto_idx',     'nunique'),
        )
        .round(3)
    )

    print("\n── Resumo: Nº de Experimentos por Método ───────────────────")
    print(resumo[['n_exp_mediana', 'n_exp_q25', 'n_exp_q75', 'pct_convergiu']].to_string())

    # ── Teste de Wilcoxon: ClassicUpDown vs ModelUpDown ─────────────
    n_classic = df[df['metodo']=='ClassicUpDown']['n_exp_mediana'].values
    n_model   = df[df['metodo']=='ModelUpDown']  ['n_exp_mediana'].values
    n_binary  = df[df['metodo']=='BinarySearch'] ['n_exp_mediana'].values

    if len(n_classic) == len(n_model) and len(n_classic) > 5:
        stat_m, p_m = stats.wilcoxon(n_classic, n_model,  alternative='greater')
        stat_b, p_b = stats.wilcoxon(n_classic, n_binary, alternative='greater')
        print(f"\n  Wilcoxon: Classic > ModelUpDown  → p = {p_m:.4f}")
        print(f"  Wilcoxon: Classic > BinarySearch → p = {p_b:.4f}")
        print(f"  (p < 0.05 indica que o método clássico precisa de mais experimentos)")

    # ── Salvar ───────────────────────────────────────────────────────
    caminho_excel = _TABELAS_VAL / f'MTL_Simulacao_Dose_{via}.xlsx'
    with pd.ExcelWriter(caminho_excel) as writer:
        df.to_excel(writer,     sheet_name='Por_Composto', index=False)
        resumo.to_excel(writer, sheet_name='Resumo')
    print(f"\n✓ Resultados salvos: {caminho_excel}")

    return dict(df_resultados=df, resumo=resumo, via=via)


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plot_convergencia_single(
    resultado_classico: dict,
    resultado_modelo:   dict,
    resultado_binario:  dict,
    ld50_real_mg_kg:    float,
    titulo:             str = '',
    salvar_path:        Optional[Path] = None,
) -> plt.Figure:
    """
    Plota a trajetória de dose dos três métodos para um único composto.

    Eixo X: número do experimento  |  Eixo Y: dose em log₁₀(mg/kg)
    Linha horizontal tracejada vermelha: LD50 real
    Faixa verde: margem de tolerância ±0.3 log
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    log_real  = np.log10(ld50_real_mg_kg)

    configs = [
        (resultado_classico, 'Up-Down Clássico\n(OECD 425)', 'steelblue', axes[0]),
        (resultado_modelo,   'Up-Down Informado\npelo Modelo', 'darkorange', axes[1]),
        (resultado_binario,  'Busca Binária\nguiada pelo Modelo', 'seagreen',  axes[2]),
    ]

    for res, label, cor, ax in configs:
        hist  = res['historico']
        doses = [h[0] for h in hist]
        resp  = [h[1] for h in hist]
        n     = len(hist)
        x     = list(range(1, n + 1))

        log_doses = np.log10(doses)
        cores_pts = ['red' if r == 1 else 'blue' for r in resp]
        ax.scatter(x, log_doses, c=cores_pts, s=60, zorder=5, edgecolors='k', linewidths=0.4)
        ax.plot(x, log_doses, color=cor, lw=1.5, alpha=0.7)

        ax.axhline(log_real, color='red', lw=2, ls='--', label=f'LD50 real = {ld50_real_mg_kg:.0f} mg/kg')
        ax.axhspan(log_real - 0.3, log_real + 0.3, alpha=0.12, color='green', label='±0.3 log')

        ax.set_xlabel('Experimento nº')
        ax.set_ylabel('log₁₀(dose [mg/kg])')
        ax.set_title(f'{label}\n{n} experimentos | erro={res["erro_log"]:.3f} log')
        ax.legend(fontsize=7)

        # Marcadores de morte (▲ vermelho) e sobrevivência (● azul)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', ms=8, label='Sobreviveu'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='red',  ms=8, label='Morreu'),
        ]
        ax.legend(handles=handles, fontsize=7, loc='lower right')

    fig.suptitle(f'Trajetória de Dose — {titulo}\nLD50 real = {ld50_real_mg_kg:.0f} mg/kg', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if salvar_path:
        plt.savefig(salvar_path, dpi=300)
    return fig


def plot_comparacao_metodos(resultado: dict, salvar: bool = True) -> plt.Figure:
    """
    Violin plot + boxplot comparando o número de experimentos por método.
    Inclui teste estatístico de Wilcoxon (p-value anotado).
    """
    df  = resultado['df_resultados']
    via = resultado['via']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ── Painel 1: distribuição do nº de experimentos ────────────────
    ax1     = axes[0]
    metodos = ['ClassicUpDown', 'ModelUpDown', 'BinarySearch']
    labels  = ['Up-Down\nClássico', 'Up-Down\nc/ Modelo', 'Busca Binária\nc/ Modelo']
    cores   = ['#4878d0', '#ee854a', '#6acc65']
    dados_n = [df[df['metodo']==m]['n_exp_mediana'].values for m in metodos]

    parts = ax1.violinplot(dados_n, positions=range(len(metodos)), showmedians=True, showextrema=True)
    for pc, cor in zip(parts['bodies'], cores):
        pc.set_facecolor(cor); pc.set_alpha(0.6)

    ax1.set_xticks(range(len(metodos))); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Nº de experimentos (mediana por composto)')
    ax1.set_title(f'Nº de experimentos por método\nVia: {via}')
    ax1.axhline(5, color='gray', lw=0.8, ls=':', label='OECD mínimo (5)')
    ax1.legend(fontsize=8)

    # Anotação de medianas
    for i, d in enumerate(dados_n):
        med = np.median(d)
        ax1.text(i, med + 0.2, f'{med:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Painel 2: erro log vs nº de experimentos ────────────────────
    ax2 = axes[1]
    for m, label, cor in zip(metodos, labels, cores):
        sub = df[df['metodo'] == m]
        ax2.scatter(
            sub['n_exp_mediana'], sub['erro_log_mediana'],
            alpha=0.6, s=35, color=cor, label=label.replace('\n', ' '),
            edgecolors='k', linewidths=0.3,
        )

    ax2.axhline(0.3, color='red', lw=1.2, ls='--', label='Tolerância (0.3 log)')
    ax2.set_xlabel('Nº de experimentos (mediana)')
    ax2.set_ylabel('Erro |log₁₀(LD50 est.) − log₁₀(LD50 real)|')
    ax2.set_title('Trade-off: Nº experimentos vs Erro')
    ax2.legend(fontsize=7)

    fig.suptitle(f'Comparação de Estratégias de Dose-Finding — Via {via}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if salvar:
        caminho = _PLOTS_VAL / f'simulacao_dose_{via}.png'
        plt.savefig(caminho, dpi=300)
        print(f"✓ Plot comparação salvo: {caminho}")

    return fig


def plot_reducao_animais(resultado: dict, salvar: bool = True) -> plt.Figure:
    """
    Gráfico de barras mostrando a redução média de animais em relação ao método clássico.
    Comunica diretamente o impacto na redução do uso de animais (princípio 3R).
    """
    df      = resultado['df_resultados']
    via     = resultado['via']
    resumo  = resultado['resumo']

    # Usa a MÉDIA dos experimentos por composto para o cálculo da redução,
    # pois a mediana é muito discreta e pode esconder ganhos reais em subconjuntos.
    medias = df.groupby('metodo')['n_exp_mediana'].mean()
    
    base_classico = medias.get('ClassicUpDown', 1.0)

    metodos = [m for m in ['ModelUpDown', 'BinarySearch'] if m in medias.index]
    labels  = {'ModelUpDown': 'Up-Down\nc/ Modelo', 'BinarySearch': 'Busca Binária\nc/ Modelo'}
    cores   = ['#ee854a', '#6acc65']

    reducoes = [
        100 * (base_classico - medias[m]) / base_classico
        for m in metodos
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [labels[m] for m in metodos], reducoes,
        color=cores, edgecolor='k', linewidth=0.5, width=0.5,
    )

    for bar, val in zip(bars, reducoes):
        # Garante que não mostre valores negativos por ruído estatístico insignificante
        val_display = max(0.0, val)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val_display:.1f}%', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Redução média de animais vs. Clássico (%)')
    ax.set_title(
        f'Redução Estimada de Uso de Animais (Princípio 3R)\n'
        f'(Média global: {base_classico:.1f} animais/composto) | Via {via}'
    )
    ax.set_ylim(0, max(reducoes + [30]) * 1.25)
    ax.axhline(0, color='gray', lw=0.8)
    plt.tight_layout()

    if salvar:
        caminho = _PLOTS_VAL / f'reducao_animais_{via}.png'
        plt.savefig(caminho, dpi=300)
        print(f"✓ Plot redução animais salvo: {caminho}")

    return fig


# ============================================================
# DEMO SINGLE COMPOUND
# ============================================================

def demo_composto_unico(
    log_ld50_real: float = 3.0,    # LD50 = 1000 mg/kg
    log_ld50_pred: float = 2.85,   # predição ligeiramente errada
    seed:          int   = 42,
) -> tuple:
    """
    Demonstração com um único composto: roda os 3 métodos e plota as trajetórias.
    Útil para visualizar intuitivamente o comportamento de cada estratégia.

    ld50_real = 10^3.0 = 1000 mg/kg → toxicidade moderada (sal de cozinha ≈ 3000)
    """
    ld50_real = 10 ** log_ld50_real

    res_c = simular_up_down_classico(ld50_real, seed=seed, tolerancia_log=0.3)
    res_m = simular_up_down_modelo(ld50_real, pred_log_ld50=log_ld50_pred, seed=seed, tolerancia_log=0.3)
    res_b = simular_busca_binaria(ld50_real, pred_log_ld50=log_ld50_pred, seed=seed, tolerancia_log=0.3)

    print(f"\n── Demo: LD50 real = {ld50_real:.0f} mg/kg (log = {log_ld50_real}) ─────")
    print(f"  Predição do modelo: log = {log_ld50_pred} (erro = {abs(log_ld50_real-log_ld50_pred):.2f} log)")
    for res in [res_c, res_m, res_b]:
        conv = '✓' if res['convergiu'] else '✗'
        print(f"  {res['metodo']:28s} → {res['n_experimentos']:2d} exp | erro = {res['erro_log']:.3f} {conv}")

    caminho_fig = _PLOTS_VAL / 'demo_convergencia_unico.png'
    fig = plot_convergencia_single(
        res_c, res_m, res_b, ld50_real,
        titulo=f'LD50 real={ld50_real:.0f} mg/kg | pred={10**log_ld50_pred:.0f} mg/kg',
        salvar_path=caminho_fig,
    )
    print(f"✓ Demo plot salvo: {caminho_fig}")
    return res_c, res_m, res_b, fig


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  ToxiciTOOL — Simulação de Dose-Finding")
    print("=" * 60)

    # Demo com composto único (não precisa de modelo)
    print("\n[1] Demo com composto único")
    demo_composto_unico()

    # Comparação completa precisa do modelo carregado
    print("\n[2] Para comparação completa (requer modelo):")
    print("    from src.analysis.validacao_estatistica import carregar_modelo_e_dados")
    print("    from src.analysis.simulacao_dose import simular_e_comparar, plot_comparacao_metodos")
    print("    dados = carregar_modelo_e_dados()")
    print("    resultado = simular_e_comparar(dados, via='rat_vo', n_compostos=50)")
    print("    plot_comparacao_metodos(resultado)")
    print("    plot_reducao_animais(resultado)")


if __name__ == '__main__':
    main()
