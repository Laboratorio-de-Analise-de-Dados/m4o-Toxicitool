
import os
import sys
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

# Adiciona a raiz do projeto ao sys.path para os imports funcionarem
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS, VIAS, Y_COLS
from src.modeling.modelagem_MTL import (
    carregar_split, preparar_dados, CamadaIncerteza, get_huber_uncertainty_loss
)

def _masked_mse_local(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)
    y_true_safe = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    sq_diff = tf.square(y_true_safe - y_pred)
    return tf.reduce_sum(sq_diff * mask, axis=-1) / (tf.reduce_sum(mask, axis=-1) + tf.keras.backend.epsilon())

def rebuild():
    print("Iniciando reconstrução do log de modelos...", flush=True)
    model_dir = PATHS["modelos_morgan"]
    model_files = sorted(list(model_dir.glob("*.keras")))
    
    if not model_files:
        print("Nenhum modelo .keras encontrado em:", model_dir)
        return

    last_pkl_path = None
    df_pkl = None
    registros = []

    for i, model_path in enumerate(model_files):
        print(f"[{i+1}/{len(model_files)}] Processando {model_path.name}...", flush=True)
        
        stem = model_path.stem
        try:
            parts = stem.split("__")
            metodo_split = parts[-1]
            pkl_stem = "__".join(parts[:-1])
            pkl_name = f"{pkl_stem}.pkl"
        except Exception:
            print(f"Erro ao processar nome do arquivo: {stem}", flush=True)
            continue
            
        pkl_path = PATHS["preprocessed"] / pkl_name
        if not pkl_path.exists():
            print(f"PKL não encontrado: {pkl_path}", flush=True)
            continue
            
        # Otimização: Só recarrega o PKL se ele mudou
        if pkl_path != last_pkl_path:
            print(f"  > Carregando novo PKL: {pkl_name}", flush=True)
            df_pkl = pd.read_pickle(pkl_path)
            last_pkl_path = pkl_path
            gc.collect()
        
        try:
            train_idx, val_idx, test_idx, metodo = carregar_split(pkl_path, metodo_override=metodo_split)
        except Exception as e:
            print(f"Erro ao carregar split para {metodo_split}: {e}", flush=True)
            continue
            
        X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados(df_pkl, train_idx, val_idx, test_idx)
        
        custom_objs = {
            'masked_mse': _masked_mse_local,
            'CamadaIncerteza': CamadaIncerteza,
        }
        for j in range(len(VIAS)):
            custom_objs[f"huber_uncert_task_{j}"] = get_huber_uncertainty_loss(j, None)

        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
        except Exception as e:
            print(f"Erro ao carregar modelo {model_path.name}: {e}", flush=True)
            continue
            
        predicoes = model.predict(X_test, verbose=0)
        res = {
            "arquivo_origem": pkl_name,
            "metodo_split": metodo_split,
            "caminho_modelo": str(model_path),
            "loss_global": np.nan
        }
        
        for k, (via, col) in enumerate(Y_COLS.items()):
            if via not in y_test: continue
            y_t = y_test[via]
            y_p = predicoes[k].flatten()
            mask = ~np.isnan(y_t)
            if mask.sum() > 1:
                res[f"r2_{via}"] = r2_score(y_t[mask], y_p[mask])
                res[f"mae_{via}"] = mean_absolute_error(y_t[mask], y_p[mask])
        
        vals_r2 = [v for k, v in res.items() if k.startswith("r2_")]
        vals_mae = [v for k, v in res.items() if k.startswith("mae_")]
        res["r2_medio"] = np.mean(vals_r2) if vals_r2 else np.nan
        res["mae_medio"] = np.mean(vals_mae) if vals_mae else np.nan
        res["n_vias"] = len(vals_r2)
        
        registros.append(res)
        
        if (i + 1) % 5 == 0:
            pd.DataFrame(registros).to_excel(PATHS["log_excel"], index=False)
            print(f"  > Progresso salvo ({i+1} modelos)", flush=True)
        
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
    if registros:
        df_final = pd.DataFrame(registros)
        df_final.to_excel(PATHS["log_excel"], index=False)
        print(f"\n✓ Log reconstruído com sucesso em: {PATHS['log_excel']}", flush=True)
    else:
        print("\n✗ Nenhum registro recuperado.", flush=True)

if __name__ == "__main__":
    rebuild()
