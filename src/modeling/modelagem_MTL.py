"""
modelagem_MTL.py  (src/modeling/)
=====================================
Treinamento MTL com Morgan Fingerprints.

    - Tronco compartilhado 512→256→128 + 6 cabeças de regressão
    - Masked Huber Loss + Ponderação por Incerteza Homocedástica
    - Carrega índices de split pré-computados de data/splits/<método>/
      (gerados por: python run_pipeline.py --etapa split)
    - Salva modelo .keras por experimento
    - Registra métricas no log Excel centralizado

Configuração: src/config.py  (PATHS, VIAS, Y_COLS, SPLIT)
"""

import gc
import os
os.environ["TF_XLA_FLAGS"]  = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

from config import PATHS, SPLIT, VIAS, Y_COLS


# ==========================================
# CUSTOM LAYER E FUNÇÕES DE PERDA
# ==========================================

class CamadaIncerteza(tf.keras.layers.Layer):
    """
    Armazena os log_vars (hiperparâmetros de incerteza homocedástica) para
    as N tarefas. Cada log_var é um parâmetro treinável — o modelo aprende
    automaticamente o peso relativo de cada via de exposição.
    """
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks

    def build(self, input_shape):
        self.log_vars = self.add_weight(
            name="log_vars",
            shape=(self.num_tasks,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"num_tasks": self.num_tasks})
        return config


def get_huber_uncertainty_loss(task_index: int, layer: CamadaIncerteza, delta: float = 1.0):
    """
    Fábrica de função de perda por tarefa.

    Combina:
        1. Masked Huber Loss — ignora NaN; MSE para |e|≤δ, MAE para |e|>δ.
           δ=1.0 (uma ordem de magnitude em log-scale) isola outliers biológicos.
        2. Incerteza Homocedástica — pondera a loss pela precisão aprendida
           (Kendall & Gal, 2017): L_final = exp(-s)·L + s
           Tarefas com poucos dados (ex: rat_vi) aprendem s maior → menor peso.
    """
    def loss_fn(y_true, y_pred):
        mask        = tf.cast(tf.math.logical_not(tf.math.is_nan(y_true)), tf.float32)
        y_true_safe = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)

        error        = tf.abs(y_true_safe - y_pred)
        huber        = tf.where(error <= delta, 0.5 * tf.square(error), delta * error - 0.5 * delta**2)
        loss_pura    = tf.reduce_sum(huber * mask, axis=-1) / (
            tf.reduce_sum(mask, axis=-1) + tf.keras.backend.epsilon()
        )

        s         = layer.log_vars[task_index]
        precision = tf.exp(-s)
        return precision * loss_pura + s

    loss_fn.__name__ = f"huber_uncert_task_{task_index}"
    return loss_fn


# ==========================================
# ARQUITETURA
# ==========================================

def modelo_multitask(fpSize: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(fpSize,), name="input_fingerprint")

    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0005), use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0005), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0005), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.45)(x)

    camada_incerteza = CamadaIncerteza(num_tasks=len(VIAS), name="incerteza_layer")
    x = camada_incerteza(x)

    outputs = [tf.keras.layers.Dense(1, activation="linear", name=via)(x) for via in VIAS]
    model   = tf.keras.models.Model(inputs=inp, outputs=outputs)

    losses = {via: get_huber_uncertainty_loss(i, camada_incerteza) for i, via in enumerate(VIAS)}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=losses)
    return model


# ==========================================
# CARREGAMENTO DE SPLIT PRÉ-COMPUTADO
# ==========================================

_METODO_PATH_KEY = {
    "scaffold": "splits_scaffold",
    "butina":   "splits_butina",
    "random":   "splits_random",
}


def carregar_split(arquivo: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Carrega índices de split pré-computados de data/splits/<método>/.

    O split não é calculado aqui — apenas lido do .npz gerado pela etapa 2.5.
    Isso garante que nenhum cálculo de similaridade O(n²) ocorra durante o
    treinamento, mantendo a GPU livre.
    """
    metodo   = SPLIT["method"]
    path_key = _METODO_PATH_KEY.get(metodo)
    if path_key is None:
        raise ValueError(f"SPLIT['method'] inválido: '{metodo}'. Escolha: {list(_METODO_PATH_KEY)}")

    stem        = arquivo.stem
    caminho_npz = PATHS[path_key] / f"{stem}__indices.npz"

    if not caminho_npz.exists():
        raise FileNotFoundError(
            f"\n  Índices de split não encontrados: {caminho_npz}\n"
            f"  Execute antes:\n    python run_pipeline.py --etapa split"
        )

    data      = np.load(caminho_npz)
    train_idx = data["train_idx"]
    test_idx  = data["test_idx"]

    metodo_str = f"butina_cutoff{SPLIT['tanimoto_cutoff']}" if metodo == "butina" else metodo
    print(f"  Split '{metodo_str}' carregado: treino={len(train_idx)} | teste={len(test_idx)}")
    return train_idx, test_idx, metodo_str


# ==========================================
# FUNÇÕES DE PIPELINE
# ==========================================

def preparar_dados(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    features = df["Features"].tolist()

    # Sanity check: shapes consistentes antes do stack
    shapes = {np.asarray(f).shape for f in features if not (isinstance(f, float) and np.isnan(f))}
    if len(shapes) > 1:
        raise ValueError(
            f"'Features' com shapes inconsistentes: {shapes}\n"
            "Provável causa: fingerprints de contagem gerados com versão antiga de "
            "Representacao.bitVect_to_array. Regenere os fingerprints."
        )

    X       = np.stack(features).astype(np.float32)
    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = {via: df[col].values[train_idx] for via, col in Y_COLS.items() if col in df.columns}
    y_test  = {via: df[col].values[test_idx]  for via, col in Y_COLS.items() if col in df.columns}

    print(f"  Dados: X_train={X_train.shape} | X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def treinar_modelo(X_train: np.ndarray, y_train: dict) -> tf.keras.Model:
    model = modelo_multitask(fpSize=X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=1,
    )
    return model


def salvar_modelo(
    model: tf.keras.Model,
    nome_base: str,
    metodo_split: str,
) -> Path:
    """
    Salva apenas o modelo .keras.
    Os índices de split já vivem em data/splits/ — não duplicar aqui.

    Para carregar (compile=False obrigatório por causa das losses dinâmicas):
        model = tf.keras.models.load_model(
            'caminho.keras',
            custom_objects={'CamadaIncerteza': CamadaIncerteza},
            compile=False
        )
    """
    stem           = nome_base.replace(".pkl", "")
    caminho_modelo = PATHS["modelos_morgan"] / f"{stem}__{metodo_split}.keras"
    model.save(caminho_modelo)
    print(f"  ✓ Modelo salvo: {caminho_modelo}")
    return caminho_modelo


def avaliar_e_plotar(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: dict,
    nome_arquivo: str,
    metodo_split: str,
) -> dict:
    predicoes = model.predict(X_test, verbose=0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes      = axes.flatten()

    resultado_eval  = model.evaluate(X_test, y_test, verbose=0)
    loss_global_val = resultado_eval[0] if isinstance(resultado_eval, (list, tuple)) else resultado_eval

    resultados = {
        "arquivo_origem": nome_arquivo,
        "metodo_split":   metodo_split,
        "tipo":           "morgan",
        "loss_global":    loss_global_val,
    }

    for i, (via, col) in enumerate(Y_COLS.items()):
        if via not in y_test:
            axes[i].set_title(f"{via}\nSem dados")
            continue

        y_t, y_p     = y_test[via], predicoes[i].flatten()
        mask          = ~np.isnan(y_t)
        y_t_l, y_p_l = y_t[mask], y_p[mask]

        if len(y_t_l) > 1:
            r2  = r2_score(y_t_l, y_p_l)
            mae = mean_absolute_error(y_t_l, y_p_l)
            resultados[f"r2_{via}"]  = r2
            resultados[f"mae_{via}"] = mae
            resultados[f"n_{via}"]   = int(mask.sum())

            axes[i].scatter(y_t_l, y_p_l, alpha=0.4, color="#2ca02c", edgecolors="k", s=20)
            mn, mx = min(y_t_l.min(), y_p_l.min()), max(y_t_l.max(), y_p_l.max())
            axes[i].plot([mn, mx], [mn, mx], "gray", linestyle="--", label="Identidade")
            m, b = np.polyfit(y_t_l, y_p_l, 1)
            axes[i].plot(y_t_l, m * y_t_l + b, color="red", label="Regressão")
            axes[i].set_xlabel("log(LD50) real")
            axes[i].set_ylabel("log(LD50) predito")
            axes[i].set_title(f"{via}  (n={mask.sum()})\nR²={r2:.3f} | MAE={mae:.3f}")
            axes[i].legend(fontsize=8)
        else:
            axes[i].set_title(f"{via}\nSem dados suficientes")

    fig.suptitle(f"{nome_arquivo.replace('.pkl','')} | {metodo_split}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PATHS["plots_morgan"] / f"scatter_{nome_arquivo.replace('.pkl','')}__split_{metodo_split}.png", dpi=300)
    plt.close()
    return resultados


def registrar_log(resultados: dict, caminho_modelo: Path) -> None:
    resultados["caminho_modelo"] = str(caminho_modelo)
    df_novo = pd.DataFrame([resultados])
    if PATHS["log_excel"].exists():
        df_novo = pd.concat([pd.read_excel(PATHS["log_excel"]), df_novo], ignore_index=True)
    df_novo.to_excel(PATHS["log_excel"], index=False)


# ==========================================
# LOOP DE TREINAMENTO
# ==========================================

def main() -> None:
    arquivos_pkl = sorted(PATHS["preprocessed"].glob("MTL_df_final_*.pkl"))
    print(f"Arquivos de fingerprints: {len(arquivos_pkl)}\n")

    for arquivo in arquivos_pkl:
        print(f"\n{'='*60}\n  [{arquivo.name}]\n{'='*60}")

        df_iter = pd.read_pickle(arquivo)
        train_idx, test_idx, metodo_split = carregar_split(arquivo)
        X_train, X_test, y_train, y_test  = preparar_dados(df_iter, train_idx, test_idx)
        model                             = treinar_modelo(X_train, y_train)
        caminho_modelo                    = salvar_modelo(model, arquivo.name, metodo_split)
        resultados                        = avaliar_e_plotar(model, X_test, y_test, arquivo.name, metodo_split)
        registrar_log(resultados, caminho_modelo)

        r2_medio = np.nanmean([v for k, v in resultados.items() if k.startswith("r2_")])
        print(f"  ✓ R² médio: {r2_medio:.3f}")

        # Inspeciona incertezas aprendidas
        try:
            log_vars = model.get_layer("incerteza_layer").get_weights()[0]
            sigmas   = np.exp(log_vars / 2)
            print("  Incerteza por via (σ):", {v: f"{s:.3f}" for v, s in zip(VIAS, sigmas)})
        except Exception:
            pass

        tf.keras.backend.clear_session()
        gc.collect()

    print("\n✓ Treinamento Morgan concluído!")


if __name__ == "__main__":
    main()