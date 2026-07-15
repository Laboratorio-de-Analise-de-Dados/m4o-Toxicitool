import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem

# Ajustar sys.path
_ESTE_ARQUIVO = Path(__file__).resolve()
RAIZ = _ESTE_ARQUIVO.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.modeling.modelagem_MTL import CamadaIncerteza, get_huber_uncertainty_loss
from src.representation.Representacao import Representacao
from src.config import VIAS

def debug_model(model_path: Path, smiles: str):
    print(f"\n--- DEBUG MODELO: {model_path.name} ---")
    
    # 1. Carregar modelo
    custom_objects = {"CamadaIncerteza": CamadaIncerteza}
    for i in range(len(VIAS)):
        loss_name = f"huber_uncert_task_{i}"
        custom_objects[loss_name] = get_huber_uncertainty_loss(i, None)
        
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print(f"Input shape esperado: {model.input_shape}")
    
    # 2. Extrair pesos da camada de incerteza
    try:
        layer = model.get_layer("incerteza_layer")
        log_vars = layer.get_weights()[0]
        print(f"LogVars: {log_vars}")
        max_precision = np.exp(3.0)
        confidence = (np.exp(-log_vars) / max_precision) * 100
        print(f"Confiabilidade (%): {confidence}")
    except:
        print("Camada 'incerteza_layer' não encontrada!")

    # 3. Gerar features
    config = {"use_count": "quantitativo" in model_path.name, "fpSize": 2048, "radius": 2, "use_desc": "__desc" in model_path.name}
    if "4096bits" in model_path.name: config["fpSize"] = 4096
    if "8192bits" in model_path.name: config["fpSize"] = 8192
    if "raio3" in model_path.name: config["radius"] = 3
    if "raio5" in model_path.name: config["radius"] = 5
    
    print(f"Configuração detectada: {config}")
    
    df = pd.DataFrame({"SMILES": [smiles]})
    rep = Representacao(dataframe=df)
    df = rep.mol_to_frame(col_smiles="SMILES")
    df = rep.fp_Morgan(col_frames="ROMol", radius=config["radius"], fpSize=config["fpSize"], use_count=config["use_count"])
    df = rep.bitVect_to_array("Fingerprint")
    
    if config["use_desc"]:
        df = rep.calcular_descritores(col_smiles="SMILES", lista_descritores="todos")
        df = rep.concatenar_descritores()
        
    features = np.stack(df["Features"].values).astype(np.float32)
    print(f"Shape das features geradas: {features.shape}")
    
    # 4. Predição
    preds = model.predict(features, verbose=0)
    print("\nPredições (log LD50):")
    for i, via in enumerate(VIAS):
        val = preds[i][0][0]
        print(f"  {via}: {val:.4f} (LD50: {10**val:.2f} mg/kg)")

if __name__ == "__main__":
    p = Path("results/melhor_modelo/MTL_df_final_binario_2048bits_raio3__desc__butina.keras")
    smi = "CC(=O)OCC(=O)C1(O)CCC2C3CCC4=CC(=O)CCC4(C)C3C(O)CC21C"
    debug_model(p, smi)
