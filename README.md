# ToxiciTOOL

> **Pipeline de Multi-Task Learning para predição da toxicidade aguda (LD50) de compostos químicos através de múltiplas vias de exposição.**

Desenvolvido durante o Programa Institucional de Bolsas de Iniciação em Desenvolvimento Tecnológico e Inovação (PIBITI) no laboratório LIMC-IA sob orientação do Dr. Guilherme Ferreira Silveira. O Toxicitool implementa uma abordagem Multi-Task Learning para predição de LD50 a fim de se reduzir os custos monetários e animais em etapas pré-clínicas do processo de P&D

---

## Índice

- [Visão Geral](#visão-geral)
- [Fundamentos Técnicos](#fundamentos-técnicos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pipeline de Execução](#pipeline-de-execução)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Configuração](#configuração)
- [Resultados e Artefatos](#resultados-e-artefatos)
- [Reprodutibilidade](#reprodutibilidade)
- [Referências](#referências)

---

## Visão Geral

A predição da dose letal mediana (LD50) representa um dos desafios mais complexos na toxicologia computacional pela natureza multifatorial dos processos ADME e pela multiplicidade de mecanismos de ação biológica [[1]](#ref1). O ToxiciTOOL aborda o problema com um **modelo Multi-Task Learning (MTL)**: uma rede neural de tronco compartilhado com seis cabeças de regressão independentes, predizendo o log₁₀(LD50) para cada combinação espécie/via simultaneamente [[2]](#ref2).

### Vias modeladas

| ID | Espécie | Via |
|----|---------|-----|
| `mouse_vi` | Camundongo | Intravenosa |
| `mouse_vo` | Camundongo | Oral |
| `mouse_ip` | Camundongo | Intraperitoneal |
| `rat_vi` | Rato | Intravenosa |
| `rat_vo` | Rato | Oral |
| `rat_ip` | Rato | Intraperitoneal |

---

## Fundamentos Técnicos

### Representação molecular

A qualidade da representação molecular é o fator determinante no desempenho de modelos QSAR [[3]](#ref3). O pipeline utiliza **Morgan Count Fingerprints** (ECFP com contagem de subestruturas), cuja superioridade sobre fingerprints binários em tarefas de regressão biológica é documentada na literatura [[4]](#ref4): a contagem captura a *densidade* de grupos funcionais, não apenas sua presença.

#### Grid de fingerprints (18 combinações)

| Parâmetro | Valores | Raciocínio |
|-----------|---------|-----------|
| Raio | 2, 3, 5 | ECFP4 (local), ECFP6 (médio), ECFP10 (global) |
| Tamanho | 2048, 4096, 8192 bits | Cobertura vs. esparsidade |
| Tipo | binário, contagem | Presença vs. frequência |

#### Descritores físico-químicos

Complementando os fingerprints, 10 descritores RDKit relacionados a propriedades ADME são calculados e concatenados ao vetor de entrada [[5]](#ref5):

| Descritor | Relevância |
|-----------|-----------|
| `MolLogP` | Lipofilicidade — distribuição tecidual |
| `MolWt` | Tamanho — absorção e excreção |
| `TPSA` | Superfície polar — permeabilidade de membrana |
| `LabuteASA` | Área acessível ao solvente |
| `NumRotatableBonds` | Flexibilidade conformacional |
| `NumHDonors` / `NumHAcceptors` | Pontes de hidrogênio |
| `FractionCSP3` | Complexidade 3D |
| `RingCount` / `NumAromaticRings` | Aromaticidade — metabolismo hepático |

Os descritores são escalonados com **RobustScaler** (mediana/IQR) antes da concatenação, garantindo robustez frente a outliers biológicos extremos [[6]](#ref6). O vetor final tem shape `(fpSize + 10,)`.

### Particionamento consciente de similaridade

O split aleatório em datasets moleculares introduz **data leakage químico**: membros da mesma série estrutural aparecem em treino e teste simultaneamente, inflando artificialmente as métricas [[7]](#ref7). O ToxiciTOOL implementa três estratégias:

**Scaffold Split** — baseado no algoritmo de Bemis-Murcko [[8]](#ref8), agrupa moléculas pelo núcleo estrutural central. O(n). **Padrão recomendado.**

**Butina Clássico** — clustering por similaridade de Tanimoto [[9]](#ref9), constrói a matriz triangular completa de distâncias. O(n²) — restrito a n ≤ 8.000.

**Butina-batch (Leader-Follower)** — implementação escalável desenvolvida neste projeto. Mantém em RAM apenas os fingerprints dos líderes de cluster (k << n), com complexidade O(n×k). Produz resultados quimicamente equivalentes ao Butina clássico para o objetivo de evitar data leakage.

Os índices de split são **pré-computados e salvos em disco** (`data/splits/<método>/`), garantindo que nenhum cálculo de similaridade ocorra durante a fase de GPU.

### Arquitetura da rede neural

```
Input (fpSize + 10 descritores)
    │
    ├─ Dense(512, L2) → BatchNorm → ReLU → Dropout(0.35)
    ├─ Dense(256, L2) → BatchNorm → ReLU → Dropout(0.35)
    ├─ Dense(128, L2) → BatchNorm → ReLU → Dropout(0.45)
    │
    ├─ CamadaIncerteza [log_vars treinável, shape=(6,)]
    │
    ├── Dense(1) → mouse_vi
    ├── Dense(1) → mouse_vo
    ├── Dense(1) → mouse_ip
    ├── Dense(1) → rat_vi
    ├── Dense(1) → rat_vo
    └── Dense(1) → rat_ip
```

### Função de perda: Masked Huber + Incerteza Homocedástica

**Masked Huber Loss** — ignora entradas NaN e isola outliers biológicos com δ = 1,0 (uma ordem de magnitude em log-scale):

$$\mathcal{L}_{Huber}(e) = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta|e| - \frac{1}{2}\delta^2 & |e| > \delta \end{cases}$$

**Ponderação por Incerteza Homocedástica** (Kendall & Gal, 2017 [[10]](#ref10)) — cada tarefa possui um parâmetro treinável σ² que pondera automaticamente a contribuição da via na loss total:

$$\mathcal{L}_{total} = \sum_{i=1}^{6} \left( \frac{1}{\sigma_i^2} \mathcal{L}_i + \log \sigma_i \right)$$

Vias com escassez de dados convergem para σ maior → recebem menor peso → balanceamento automático sem heurísticas manuais.

---

## Estrutura do Projeto

```
ToxiciTOOL/
├── src/
│   ├── config.py                    # Caminhos e hiperparâmetros centralizados
│   ├── run_pipeline.py              # Ponto de entrada único
│   ├── test_modularization.py       # Smoke-tests
│   ├── validate_requirements.py     # Validação de dependências
│   ├── preprocessing/
│   │   ├── Limpeza.py               # Canonização SMILES + deduplicação
│   │   └── preprocessamento_MTL.py  # Etapa 1
│   ├── representation/
│   │   ├── Representacao.py         # FP Morgan + descritores + RobustScaler
│   │   ├── fingerprints_MTL.py      # Etapa 2 — grid 18 combinações
│   │   └── split_datasets.py        # Etapa 2.5 — pré-computa splits
│   ├── modeling/
│   │   ├── Splitter.py              # random / scaffold / butina / butina-batch
│   │   └── modelagem_MTL.py         # Etapa 3 — MTL + treinamento
│   └── analysis/
│       └── analise_modelos.py       # Etapa 4 — ranking + plots por método
│
├── data/                            # Não versionado (ver .gitignore)
│   ├── data_raw/                    # CSVs brutos de entrada
│   ├── preprocessed/                # PKLs de fingerprints + descritores
│   └── splits/scaffold|butina|random/  # Índices .npz pré-computados
│
├── results/                         # Parcialmente não versionado
│   ├── melhor_modelo/               # Melhor .keras + resumo Excel
│   ├── modelos_salvos/morgan/       # Todos os .keras (não versionado)
│   ├── plots/                       # Figuras (não versionado)
│   └── tabelas/multitask/           # Logs e rankings Excel (versionado)
│
├── notebooks/
│   ├── reproduzir_campeoes.ipynb    # Reprodutibilidade auditável ← principal
│   └── analysis/ visualization/
│
├── environment.yml                  # Dependências Conda
├── .gitignore
└── README.md
```

---

## Pipeline de Execução

```
data/data_raw/*.csv
       │  Etapa 1 — preprocessamento
       │  Canoniza SMILES (RDKit), converte LD50→log₁₀,
       │  deduplica por (smiles+via) com |CV| ≤ 20%,
       │  merge outer de todas as vias.
       ▼
data/preprocessed/MTL_df_base_fundida.pkl
       │  Etapa 2 — fingerprints
       │  Grid 18 combinações de FP Morgan + 10 descritores
       │  físico-químicos escalonados (RobustScaler).
       ▼
data/preprocessed/MTL_df_final_<tipo>_<bits>bits_raio<r>__desc.pkl  (×18)
       │  Etapa 2.5 — split
       │  Pré-computa índices treino/teste via scaffold/butina/random.
       │  Nenhum cálculo de similaridade durante o treinamento.
       ▼
data/splits/<método>/<stem>__indices.npz  (×18)
       │  Etapa 3 — modelagem
       │  Carrega índices → treina MTL com Huber+Incerteza →
       │  salva .keras → registra métricas no log Excel.
       ▼
results/modelos_salvos/morgan/*.keras  +  MTL_Log_Modelos.xlsx
       │  Etapa 4 — análise
       │  Ranking global + por método de split + plots comparativos.
       ▼
results/plots/  +  MTL_Ranking_*.xlsx
```

---

## Instalação

### Ambiente Conda (bare metal — recomendado)

```bash
# 1. Clonar o repositório
git clone https://github.com/C0rvito/m4o-Toxicitool.git
cd ToxiciTOOL

# 2. Criar e ativar o ambiente
conda env create -f environment.yml
conda activate toxicitool

# 3. (GPU NVIDIA) Instalar TensorFlow com CUDA 12.x
pip install "tensorflow[and-cuda]"

# 4. Configurar linker dinâmico (Arch Linux / qualquer distro bare metal)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDA_CACHE_MAXSIZE=2147483648
EOF
conda deactivate && conda activate toxicitool

# 5. Verificar GPU
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

> **RDKit** deve ser instalado via `conda-forge`. A distribuição pip não inclui `PandasTools` nem `rdMolStandardize`.

### Validação

```bash
python src/test_modularization.py    # smoke-tests de importação por módulo
python src/validate_requirements.py  # verifica dependências do environment.yml
```

---

## Uso Rápido

```bash
# Pipeline completo
python src/run_pipeline.py

# Etapas individuais
python src/run_pipeline.py --etapa preprocessing  # Etapa 1
python src/run_pipeline.py --etapa fingerprints   # Etapa 2
python src/run_pipeline.py --etapa split          # Etapa 2.5
python src/run_pipeline.py --etapa modelagem      # Etapa 3
python src/run_pipeline.py --etapa analise        # Etapa 4
```

---

## Configuração

Todos os parâmetros são controlados em **`src/config.py`**. Nenhum outro arquivo precisa ser editado.

### Método de split

```python
SPLIT = {
    "method":          "scaffold",  # "scaffold" | "butina" | "random"
    "tanimoto_cutoff": 0.4,         # só para method="butina"
    "test_size":       0.2,
    "random_state":    42,
}
```

Para `method="butina"`, a seleção automática aplica:
- n ≤ 8.000 → Butina clássico (O(n²), exato)
- n > 8.000 → Butina-batch leader-follower (O(n×k), escalável)

### Descritores físico-químicos

```python
DESCRITORES = {
    "ativo": True,    # False → apenas fingerprints (comportamento legado)
    "lista": ["MolLogP", "MolWt", "TPSA", ...],
}
```

---

## Resultados e Artefatos

### Rankings por método de split

```
results/tabelas/multitask/
├── MTL_Log_Modelos.xlsx                         # Log completo de todos os experimentos
├── MTL_Ranking_Modelos.xlsx                     # Ranking global
├── MTL_Ranking_scaffold_Modelos.xlsx
├── MTL_Ranking_butina_cutoff0.4_Modelos.xlsx
└── MTL_Ranking_random_Modelos.xlsx
```

### Carregamento do modelo campeão

```python
import tensorflow as tf
import numpy as np
from src.modeling.modelagem_MTL import CamadaIncerteza

# compile=False é obrigatório — as losses são funções dinâmicas (closures)
model = tf.keras.models.load_model(
    "results/melhor_modelo/<modelo>.keras",
    custom_objects={"CamadaIncerteza": CamadaIncerteza},
    compile=False,
)

# Inferência em novas moléculas (X com shape (n, fpSize+10))
predicoes = model.predict(X_novo)
# predicoes[i] → array de log₁₀(LD50) para a i-ésima via

# Incerteza aprendida por via
log_vars = model.get_layer("incerteza_layer").get_weights()[0]
sigmas   = np.exp(log_vars / 2)   # σ — quanto maior, menos dados, menos peso
```

---

## Reprodutibilidade

O notebook **`notebooks/reproduzir_campeoes.ipynb`** permite auditar qualquer experimento sem re-executar o pipeline de pré-processamento ou split.

### Lógica do notebook

| Célula | Ação |
|--------|------|
| 1 | Trava global: `np.random.seed(42)`, `tf.random.set_seed(42)`, `TF_DETERMINISTIC_OPS=1` |
| 2–3 | Imports e seleção do experimento (`PKL_CAMPEAO`, `METODO_SPLIT`) |
| 4 | Carrega PKL + índices `.npz` pré-computados |
| 5 | Monta tensores X_train, X_test, y_train, y_test |
| 6–7 | Instancia `modelo_multitask`, executa `.fit()`, plota curva de aprendizado |
| 8 | Avalia R² e MAE por via com tabela formatada |
| 9 | Gráficos de dispersão predito vs. real (6 subplots, padrão de publicação) |
| 10 | Plota perfil de incerteza σ por via (barplot) |
| 11 | Scatter R² vs MAE por via |
| 12 | Resumo completo de reprodutibilidade com timestamp |

### Executar

```bash
jupyter notebook notebooks/reproduzir_campeoes.ipynb
```

---

## Referências

<a name="ref1">[1]</a> Zhu, H. *et al.* "Quantitative Structure-Activity Relationship Modeling of Rat Acute Toxicity by Oral Exposure." *Chemical Research in Toxicology*, 22(12), 1913–1921, 2009. https://doi.org/10.1021/tx900189p

<a name="ref2">[2]</a> Caruana, R. "Multitask Learning." *Machine Learning*, 28(1), 41–75, 1997. https://doi.org/10.1023/A:1007379606734

<a name="ref3">[3]</a> Muratov, E.N. *et al.* "QSAR without borders." *Chemical Society Reviews*, 49(11), 3525–3564, 2020. https://doi.org/10.1039/D0CS00098A

<a name="ref4">[4]</a> Rogers, D.; Hahn, M. "Extended-Connectivity Fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742–754, 2010. https://doi.org/10.1021/ci100050t

<a name="ref5">[5]</a> Lipinski, C.A. *et al.* "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." *Advanced Drug Delivery Reviews*, 46(1–3), 3–26, 2001. https://doi.org/10.1016/S0169-409X(00)00129-0

<a name="ref6">[6]</a> Pedregosa, F. *et al.* "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830, 2011.

<a name="ref7">[7]</a> Sheridan, R.P. "Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction." *Journal of Chemical Information and Modeling*, 53(4), 783–790, 2013. https://doi.org/10.1021/ci400084k

<a name="ref8">[8]</a> Bemis, G.W.; Murcko, M.A. "The Properties of Known Drugs. 1. Molecular Frameworks." *Journal of Medicinal Chemistry*, 39(15), 2887–2893, 1996. https://doi.org/10.1021/jm9602928

<a name="ref9">[9]</a> Butina, D. "Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets." *Journal of Chemical Information and Computer Sciences*, 39(4), 747–750, 1999. https://doi.org/10.1021/ci9803381

<a name="ref10">[10]</a> Kendall, A.; Gal, Y. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5574–5584, 2017.

<a name="ref11">[11]</a> Landrum, G. *et al.* RDKit: Open-Source Cheminformatics Software, 2006–2024. https://www.rdkit.org

<a name="ref12">[12]</a> Abadi, M. *et al.* "TensorFlow: A System for Large-Scale Machine Learning." *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 265–283, 2016.

---

## Licença

MIT License — consulte `LICENSE` para detalhes.