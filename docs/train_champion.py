import pandas as pd
import numpy as np
import tensorflow as tf
from src.config import PATHS, SPLIT, VIAS, Y_COLS
from src.modeling.modelagem_MTL import (
    carregar_split, preparar_dados, treinar_modelo, 
    avaliar_e_plotar, registrar_log
)

def main():
    # USANDO A CONFIGURAÇÃO CAMPEÃ DO MANUSCRITO (8192 BITS)
    arquivo = PATHS["preprocessed"] / "MTL_df_final_binario_8192bits_raio2__desc.pkl"
    
    if not arquivo.exists():
        print(f"Erro: Arquivo {arquivo.name} não encontrado. Gere os fingerprints primeiro.")
        return

    print(f"\n{'='*60}\n  TREINANDO MODELO CAMPEÃO (MANUSCRITO): [{arquivo.name}]\n{'='*60}")

    df_iter = pd.read_pickle(arquivo)
    
    # Garante que estamos usando Scaffold como no manuscrito
    SPLIT["method"] = "scaffold"
    train_idx, val_idx, test_idx, metodo_split = carregar_split(arquivo)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados(df_iter, train_idx, val_idx, test_idx)
    
    print(f"\nIniciando treinamento com 8k bits (Batch 64, LR 1e-4)...")
    model = treinar_modelo(X_train, y_train, X_val, y_val)
    
    caminho_modelo = PATHS["modelos_morgan"] / f"CHAMPION_{arquivo.stem}__{metodo_split}.keras"
    model.save(caminho_modelo)
    
    print("\nAvaliado no conjunto de TESTE (Scaffold):")
    resultados = avaliar_e_plotar(model, X_test, y_test, arquivo.name, metodo_split)
    registrar_log(resultados, caminho_modelo)
    
    r2_medio = np.nanmean([v for k, v in resultados.items() if k.startswith("r2_")])
    print(f"\n  🚀 RESULTADO FINAL - R² MÉDIO: {r2_medio:.4f}")
    print(f"  (Meta do Manuscrito: 0.3880)")

if __name__ == "__main__":
    main()
