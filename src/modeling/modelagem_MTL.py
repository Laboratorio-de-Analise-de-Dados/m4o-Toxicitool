"""
modelagem_MTL.py  (src/modeling/)
=====================================
Treinamento MTL com Morgan Fingerprints.

    - Tronco compartilhado 1024→512→256 + 6 cabeças de regressão
    - Masked Huber Loss + Ponderação por Incerteza Homocedástica (Trava Rígida)
    - Carrega índices de split pré-computados de data/splits/<método>/
    - Salva modelo .keras por experimento
    - Registra métricas no log Excel centralizado
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

# Habilita o "Memory Growth" para evitar que o TF aloque toda a VRAM de uma vez
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  [GPU] Memory Growth ativado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"  [GPU Erro] {e}")

from sklearn.metrics import mean_absolute_error, r2_score

from src.config import PATHS, SPLIT, VIAS, Y_COLS
from src.analysis.classificacao import avaliar_classificacao


# ==========================================
# CUSTOM LAYER E FUNÇÕES DE PERDA
# ==========================================

class CamadaIncerteza(tf.keras.layers.Layer):
    """
    Armazena os log_vars para as N tarefas. 
    Trava rígida em >= -3.0 para permitir precisão maior que a variância unitária.
    """
    def __init__(self, num_tasks: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks

    def build(self, input_shape):
        self.log_vars = self.add_weight(
            name="log_vars",
            shape=(self.num_tasks,),
            initializer=tf.keras.initializers.Constant(value=0.5), # Começa com incerteza moderada
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # ALTERADO: Trava baixada para -3.0 para permitir que o modelo confie mais nas tarefas
        self.log_vars.assign(tf.maximum(self.log_vars, -3.0))
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"num_tasks": self.num_tasks})
        return config


def get_huber_uncertainty_loss(task_index: int, layer: CamadaIncerteza, delta: float = 0.5):
    """
    Fábrica de função de perda por tarefa.
    precision = exp(-s), onde s é log_var.
    ALTERADO: delta de 1.0 para 0.5 para maior robustez a outliers.
    """
    def loss_fn(y_true, y_pred):
        mask        = tf.cast(tf.math.logical_not(tf.math.is_nan(y_true)), tf.float32)
        y_true_safe = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)

        error        = tf.abs(y_true_safe - y_pred)
        huber        = tf.where(error <= delta, 0.5 * tf.square(error), delta * error - 0.5 * delta**2)
        
        count = tf.reduce_sum(mask, axis=-1)
        loss_pura = tf.reduce_sum(huber * mask, axis=-1) / (count + tf.keras.backend.epsilon())

        s = layer.log_vars[task_index]
        precision = tf.exp(-s)
        
        return precision * loss_pura + s

    loss_fn.__name__ = f"huber_uncert_task_{task_index}"
    return loss_fn


# ==========================================
# ARQUITETURA
# ==========================================

def modelo_multitask(fpSize: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(fpSize,), name="input_fingerprint")
    x = tf.keras.layers.BatchNormalization()(inp)

    # ALTERADO: L2 de 0.001 para 0.005 para combater overfitting
    reg = tf.keras.regularizers.l2(0.005)

    x = tf.keras.layers.Dense(1024, kernel_regularizer=reg, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    x = tf.keras.layers.Dense(512, kernel_regularizer=reg, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    x = tf.keras.layers.Dense(256, kernel_regularizer=reg, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # ALTERADO: Dropout de 0.35 para 0.45 na última camada densa
    x = tf.keras.layers.Dropout(0.45)(x)

    camada_incerteza = CamadaIncerteza(num_tasks=len(VIAS), name="incerteza_layer")
    x = camada_incerteza(x)

    outputs = [tf.keras.layers.Dense(1, activation="linear", name=via)(x) for via in VIAS]
    model   = tf.keras.models.Model(inputs=inp, outputs=outputs)

    losses = {via: get_huber_uncertainty_loss(i, camada_incerteza) for i, via in enumerate(VIAS)}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=losses)
    return model


# ==========================================
# CARREGAMENTO DE SPLIT PRÉ-COMPUTADO
# ==========================================

_METODO_PATH_KEY = {
    "scaffold": "splits_scaffold",
    "butina":   "splits_butina",
    "random":   "splits_random",
}


def carregar_split(arquivo: Path, metodo_override: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    metodo   = metodo_override or SPLIT["method"]
    path_key = _METODO_PATH_KEY.get(metodo)
    stem     = arquivo.stem
    caminho_npz = PATHS[path_key] / f"{stem}__indices.npz"

    data      = np.load(caminho_npz)
    return data["train_idx"], data["val_idx"], data["test_idx"], metodo


# ==========================================
# FUNÇÕES DE PIPELINE
# ==========================================

def preparar_dados(df, train_idx, val_idx, test_idx):
    features = np.stack(df["Features"].tolist()).astype(np.float32)
    
    X_train, X_val, X_test = features[train_idx], features[val_idx], features[test_idx]
    y_train = {via: df[col].values[train_idx] for via, col in Y_COLS.items() if col in df.columns}
    y_val   = {via: df[col].values[val_idx]   for via, col in Y_COLS.items() if col in df.columns}
    y_test  = {via: df[col].values[test_idx]  for via, col in Y_COLS.items() if col in df.columns}

    return X_train, X_val, X_test, y_train, y_val, y_test


def treinar_modelo(X_train, y_train, X_val, y_val):
    model = modelo_multitask(fpSize=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64, # Reduzido de 128 para 64 para evitar OOM
        callbacks=[
            # ALTERADO: Patience de 20 para 30 para permitir convergência com mais regularização
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
        ],
        verbose=1,
    )
    loss_global = min(history.history["val_loss"])
    return model, loss_global


def salvar_modelo(model, nome_base, metodo_split):
    stem = nome_base.replace(".pkl", "")
    path = PATHS["modelos_morgan"] / f"{stem}__{metodo_split}.keras"
    model.save(path)
    return path


def avaliar_e_plotar(model, X_test, y_test, nome_arquivo, metodo_split):
    predicoes = model.predict(X_test, verbose=0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    resultados = {"arquivo_origem": nome_arquivo, "metodo_split": metodo_split}

    for i, (via, col) in enumerate(Y_COLS.items()):
        if via not in y_test: 
            axes[i].set_title(f"{via}\nSem dados")
            continue
            
        y_t, y_p = y_test[via], predicoes[i].flatten()
        mask = ~np.isnan(y_t)
        
        if mask.sum() > 1:
            y_t_l, y_p_l = y_t[mask], y_p[mask]
            r2 = r2_score(y_t_l, y_p_l)
            mae = mean_absolute_error(y_t_l, y_p_l)
            resultados[f"r2_{via}"]  = r2
            resultados[f"mae_{via}"] = mae
            
            # TODO 4: Classificação
            avaliar_classificacao(y_t_l, y_p_l, via, metodo_split)

            # Plot
            axes[i].scatter(y_t_l, y_p_l, alpha=0.4, color="#2ca02c", edgecolors="k", s=20)
            mn, mx = min(y_t_l.min(), y_p_l.min()), max(y_t_l.max(), y_p_l.max())
            axes[i].plot([mn, mx], [mn, mx], "gray", linestyle="--")
            axes[i].set_title(f"{via}\nR2={r2:.3f} | MAE={mae:.3f}")
        else:
            axes[i].set_title(f"{via}\nDados insuficientes")

    plt.tight_layout()
    plot_path = PATHS["plots_morgan"] / f"scatter_{nome_arquivo.replace('.pkl','')}__{metodo_split}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  ✓ Gráfico atualizado: {plot_path.name}")
    return resultados


def registrar_log(resultados, caminho_modelo):
    resultados["caminho_modelo"] = str(caminho_modelo)
    df = pd.DataFrame([resultados])
    if PATHS["log_excel"].exists():
        df = pd.concat([pd.read_excel(PATHS["log_excel"]), df], ignore_index=True)
    df.to_excel(PATHS["log_excel"], index=False)


def main():
    arquivos_pkl = sorted(PATHS["preprocessed"].glob("MTL_df_final_*.pkl"))
    for arquivo in arquivos_pkl:
        # Pega o método de split padrão para construir o nome do modelo esperado
        metodo_atual = SPLIT["method"]
        stem = arquivo.name.replace(".pkl", "")
        caminho_modelo_esperado = PATHS["modelos_morgan"] / f"{stem}__{metodo_atual}.keras"

        # Lógica de Resume: Se o modelo já existe, pula para o próximo
        if caminho_modelo_esperado.exists():
            print(f"  [Resume] Pulando: {arquivo.name} (Modelo já treinado)")
            continue

        # Limpeza preventiva de VRAM no início da iteração
        tf.keras.backend.clear_session()
        gc.collect()

        print(f"\n>>> Treinando: {arquivo.name}")
        df = pd.read_pickle(arquivo)
        train_idx, val_idx, test_idx, metodo = carregar_split(arquivo)
        X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados(df, train_idx, val_idx, test_idx)
        
        model, loss_global = treinar_modelo(X_train, y_train, X_val, y_val)
        path  = salvar_modelo(model, arquivo.name, metodo)
        res   = avaliar_e_plotar(model, X_test, y_test, arquivo.name, metodo)
        res["loss_global"] = loss_global
        registrar_log(res, path)

        # Limpeza agressiva de VRAM e RAM ao final da iteração
        del model
        del X_train, X_val, X_test, y_train, y_val, y_test
        del df
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    main()
