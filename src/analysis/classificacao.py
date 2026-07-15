import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from pathlib import Path
import sys

# Adiciona a RAIZ ao path
_SRC  = Path(__file__).resolve().parent.parent
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS

def log_dl50_para_classe_ghs(log_dl50: np.ndarray) -> np.ndarray:
    """
    Converte log10(DL50) para categoria GHS (1–5).
    Limites em mg/kg: ≤5 | ≤50 | ≤300 | ≤2000 | >2000
    Em log10: ≤0.699 | ≤1.699 | ≤2.477 | ≤3.301 | >3.301
    """
    dl50 = 10 ** log_dl50
    classes = np.full_like(log_dl50, 5, dtype=int)
    classes[dl50 <= 2000] = 4
    classes[dl50 <= 300]  = 3
    classes[dl50 <= 50]   = 2
    classes[dl50 <= 5]    = 1
    return classes

def avaliar_classificacao(y_true_log: np.ndarray, y_pred_log: np.ndarray, via: str, split_name: str) -> dict:
    """
    Avalia o desempenho de classificação a partir das predições de regressão.
    Gera matriz de confusão e calcula métricas.
    """
    y_true_cls = log_dl50_para_classe_ghs(y_true_log)
    y_pred_cls = log_dl50_para_classe_ghs(y_pred_log)
    
    classes_labels = [1, 2, 3, 4, 5]
    
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1_macro = f1_score(y_true_cls, y_pred_cls, average='macro')
    f1_weighted = f1_score(y_true_cls, y_pred_cls, average='weighted')
    kappa = cohen_kappa_score(y_true_cls, y_pred_cls)
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=classes_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Trata divisões por zero se uma classe não existir no true
    
    # Plotagem
    fig_clf = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=classes_labels, yticklabels=classes_labels)
    plt.title(f"Matriz de Confusão Normalizada - {via}\n(Split: {split_name} | Acc: {acc:.2f} | Kappa: {kappa:.2f})")
    plt.xlabel("Classe Predita (GHS)")
    plt.ylabel("Classe Real (GHS)")
    
    output_dir = PATHS["plots"] / "analise" / "classificacao"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"confusion_{via}__{split_name}.png", dpi=300)
    plt.close(fig_clf)
    
    return {
        f"acc_{via}": acc,
        f"f1_macro_{via}": f1_macro,
        f"f1_weighted_{via}": f1_weighted,
        f"kappa_{via}": kappa
    }
