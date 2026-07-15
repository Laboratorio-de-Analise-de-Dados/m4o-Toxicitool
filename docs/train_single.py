import sys
from pathlib import Path
_RAIZ = Path(__file__).resolve().parent
_SRC  = _RAIZ / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from config import PATHS, SPLIT, VIAS, Y_COLS
from modeling.modelagem_MTL import (
    carregar_split, preparar_dados, treinar_modelo, 
    avaliar_e_plotar, registrar_log, CamadaIncerteza
)

def main():
    # Pega apenas a primeira configuração promissora
    arquivo = PATHS["preprocessed"] / "MTL_df_final_binario_2048bits_raio2__desc.pkl"
    print(f"\n{'='*60}\n  Treinando Modelo de Teste: [{arquivo.name}]\n{'='*60}")

    df_iter = pd.read_pickle(arquivo)
    train_idx, val_idx, test_idx, metodo_split = carregar_split(arquivo)
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados(df_iter, train_idx, val_idx, test_idx)
    
    print("\nIniciando treinamento (TF)...")
    model = treinar_modelo(X_train, y_train, X_val, y_val)
    
    caminho_modelo = PATHS["modelos_morgan"] / f"TESTE_{arquivo.stem}__{metodo_split}.keras"
    model.save(caminho_modelo)
    
    print("\nAvaliado no conjunto de TESTE (Scaffold):")
    resultados = avaliar_e_plotar(model, X_test, y_test, arquivo.name, metodo_split)
    
    r2_medio = np.nanmean([v for k, v in resultados.items() if k.startswith("r2_")])
    print(f"\n  🚀 RESULTADO FINAL - R² MÉDIO: {r2_medio:.4f}")
    
    for via in VIAS:
        r2 = resultados.get(f"r2_{via}", "N/A")
        acc = resultados.get(f"acc_{via}", "N/A")
        kappa = resultados.get(f"kappa_{via}", "N/A")
        
        # Formatação robusta para lidar com strings 'N/A' ou floats
        r2_str = f"{r2:<7.3f}" if isinstance(r2, (int, float)) else f"{r2:<7}"
        acc_str = f"{acc:<5.2f}" if isinstance(acc, (int, float)) else f"{acc:<5}"
        kappa_str = f"{kappa:<5.2f}" if isinstance(kappa, (int, float)) else f"{kappa:<5}"
        
        print(f"    - {via:10s}: R² = {r2_str} | Acc = {acc_str} | Kappa = {kappa_str}")

if __name__ == "__main__":
    main()
