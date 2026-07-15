
import pandas as pd
import numpy as np
import tensorflow as tf
from src.modeling.modelagem_MTL import carregar_split, preparar_dados, CamadaIncerteza, get_huber_uncertainty_loss
from sklearn.metrics import r2_score

def check_overfitting():
    # Load best model according to log (using one of the top ones)
    model_name = "MTL_df_final_quantitativo_2048bits_raio3__desc"
    split_method = "butina"
    pkl_path = f"data/preprocessed/{model_name}.pkl"
    model_path = f"results/modelos_salvos/morgan/{model_name}__{split_method}.keras"
    
    if not tf.io.gfile.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    # Custom objects are needed for loading
    custom_objects = {
        "CamadaIncerteza": CamadaIncerteza,
        "huber_uncert_task_0": get_huber_uncertainty_loss(0, None), # Placeholder
        "huber_uncert_task_1": get_huber_uncertainty_loss(1, None),
        "huber_uncert_task_2": get_huber_uncertainty_loss(2, None),
        "huber_uncert_task_3": get_huber_uncertainty_loss(3, None),
        "huber_uncert_task_4": get_huber_uncertainty_loss(4, None),
        "huber_uncert_task_5": get_huber_uncertainty_loss(5, None),
    }
    
    # We might need to rebuild the model if loading fails due to custom loss factories
    # For now, let's try loading weights into a fresh model
    from src.config import VIAS
    from src.modeling.modelagem_MTL import modelo_multitask
    
    from pathlib import Path
    df = pd.read_pickle(pkl_path)
    train_idx, val_idx, test_idx, _ = carregar_split(Path(pkl_path), metodo_override=split_method)
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados(df, train_idx, val_idx, test_idx)
    
    model = modelo_multitask(fpSize=X_train.shape[1])
    try:
        model.load_weights(model_path)
    except:
        print("Failed to load weights directly, trying load_model...")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            print(f"Could not load model: {e}")
            return

    def eval_set(X, y_dict, name):
        preds = model.predict(X, verbose=0)
        print(f"\n--- Results for {name} ---")
        for i, via in enumerate(VIAS):
            if via not in y_dict: continue
            y_true = y_dict[via]
            y_pred = preds[i].flatten()
            mask = ~np.isnan(y_true)
            if mask.sum() > 1:
                r2 = r2_score(y_true[mask], y_pred[mask])
                print(f"{via}: R2 = {r2:.3f}")

    eval_set(X_train, y_train, "TRAIN")
    eval_set(X_test, y_test, "TEST")

    # Check uncertainty weights
    unc_layer = model.get_layer("incerteza_layer")
    log_vars = unc_layer.get_weights()[0]
    print("\n--- Learned Uncertainties (log_vars) ---")
    for via, lv in zip(VIAS, log_vars):
        print(f"{via}: {lv:.4f} (var={np.exp(lv):.4f})")

if __name__ == "__main__":
    check_overfitting()
