"""
cross_validation.py  (src/modeling/)
=======================================
TODO 6 — Cross-Validation Scaffold-Aware.

Implementa K-Fold onde a separação é baseada em scaffolds de Murcko,
garantindo que núcleos químicos idênticos nunca estejam no treino e
na validação do mesmo fold.
"""

import gc
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

_SRC = Path(__file__).resolve().parent.parent # (Pois os arquivos estão dentro de src/modeling/)
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS, VIAS, Y_COLS
from src.modeling.modelagem_MTL import modelo_multitask


def scaffold_kfold(df: pd.DataFrame, n_splits: int = 5, smiles_col: str = "smiles", random_state: int = 42):
    """
    Divide os dados em K folds baseados em Murcko Scaffolds.
    Retorna lista de tuples (train_idx, val_idx).
    """
    print(f"  [CV] Gerando {n_splits} folds baseados em Scaffolds...")
    scaffolds = defaultdict(list)

    for i, smi in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                scaffolds[scaffold].append(i)
            else:
                scaffolds["invalid"].append(i)
        except:
            scaffolds["invalid"].append(i)

    scaffold_sets = list(scaffolds.values())
    
    # KFold sobre a lista de grupos de scaffolds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_groups_idx, val_groups_idx in kf.split(scaffold_sets):
        train_idx = []
        for i in train_groups_idx:
            train_idx.extend(scaffold_sets[i])
        
        val_idx = []
        for i in val_groups_idx:
            val_idx.extend(scaffold_sets[i])
            
        folds.append((np.array(train_idx), np.array(val_idx)))
        
    return folds


def cross_validate_mtl(arquivo_pkl: Path, n_splits: int = 5):
    print(f"\n{'='*60}")
    print(f"  INICIANDO CROSS-VALIDATION (K={n_splits})")
    print(f"  Arquivo: {arquivo_pkl.name}")
    print(f"{'='*60}")

    df = pd.read_pickle(arquivo_pkl)
    X  = np.stack(df["Features"].values).astype(np.float32)
    
    folds = scaffold_kfold(df, n_splits=n_splits)
    
    resultados_folds = []

    for k, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  >>> Fold {k+1}/{n_splits} | Treino={len(train_idx)} | Val={len(val_idx)}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = {via: df[col].values[train_idx] for via, col in Y_COLS.items() if col in df.columns}
        y_val   = {via: df[col].values[val_idx]   for via, col in Y_COLS.items() if col in df.columns}

        model = modelo_multitask(fpSize=X_train.shape[1])
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150, # Reduzido para CV ser mais rápido
            batch_size=256,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            ],
            verbose=0
        )

        # Avaliação no fold de validação
        predicoes = model.predict(X_val, verbose=0)
        res_fold  = {"fold": k + 1}
        
        for i, (via, col) in enumerate(Y_COLS.items()):
            if via not in y_val: continue
            y_t, y_p = y_val[via], predicoes[i].flatten()
            mask = ~np.isnan(y_t)
            if mask.sum() > 1:
                res_fold[f"r2_{via}"] = r2_score(y_t[mask], y_p[mask])
        
        resultados_folds.append(res_fold)
        print(f"      ✓ R² Médio Fold: {np.nanmean([v for k,v in res_fold.items() if k.startswith('r2_')]):.3f}")

        tf.keras.backend.clear_session()
        gc.collect()

    return pd.DataFrame(resultados_folds)


def plotar_cv(df_cv: pd.DataFrame, stem: str):
    cols_r2 = [c for c in df_cv.columns if c.startswith("r2_")]
    df_melt = df_cv.melt(id_vars=["fold"], value_vars=cols_r2, var_name="Via", value_name="R2")
    df_melt["Via"] = df_melt["Via"].str.replace("r2_", "")

    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_melt, x="Via", y="R2", capsize=.1, palette="viridis")
    plt.title(f"Cross-Validation Scaffold-Aware (K={len(df_cv)})\n{stem}", fontsize=14)
    plt.ylabel("R² Médio ± DP")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_dir = PATHS["plots_analise"] / "crossval"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"cv_scaffold_{stem}.png", dpi=300)
    plt.close()


def main():
    arquivos = sorted(PATHS["preprocessed"].glob("MTL_df_final_*.pkl"))
    if not arquivos: return

    # Usamos o primeiro/campeão para validar o protocolo
    arquivo = arquivos[0]
    stem    = arquivo.stem
    
    df_cv = cross_validate_mtl(arquivo)
    
    # Salvar resultados
    path_cv = PATHS["tabelas"] / f"MTL_CrossVal_Scaffold_{stem}.xlsx"
    df_cv.to_excel(path_cv, index=False)
    
    # Resumo estatístico do CV
    resumo = df_cv.drop(columns="fold").agg(["mean", "std"]).T
    path_resumo = PATHS["tabelas"] / f"MTL_CrossVal_Resumo_{stem}.xlsx"
    resumo.to_excel(path_resumo)
    
    plotar_cv(df_cv, stem)
    
    print(f"\n  ✓ Cross-Validation concluído!")
    print(f"  ✓ Resultados por fold: {path_cv.name}")
    print(f"  ✓ Resumo (Média ± DP): {path_resumo.name}")


if __name__ == "__main__":
    main()
