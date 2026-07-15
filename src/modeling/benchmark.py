import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib

_SRC = Path(__file__).resolve().parent.parent # (Pois os arquivos estão dentro de src/modeling/)
_RAIZ = _SRC.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.config import PATHS, VIAS, Y_COLS, SPLIT

def treinar_baselines(X_train, y_train, X_test, y_test, via):
    """Treina RF, Ridge e Dummy para uma única via."""
    
    # Filtra NaNs para a via específica
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    X_tr, y_tr = X_train[mask_train], y_train[mask_train]
    X_te, y_te = X_test[mask_test], y_test[mask_test]
    
    # Check for NaNs/Infs
    if not np.isfinite(X_tr).all():
        print(f"    ! Erro: X_tr em {via} contém NaNs ou Infs")
    if not np.isfinite(y_tr).all():
        print(f"    ! Erro: y_tr em {via} contém NaNs ou Infs")
    
    if len(y_tr) < 10 or len(y_te) < 2:
        print(f"    ! Dados insuficientes para benchmark em {via}")
        return []

    modelos = {
        "Random Forest (ST)": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Dummy (Média)": DummyRegressor(strategy="mean")
    }
    
    resultados_via = []
    for nome, model in modelos.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        rmse = root_mean_squared_error(y_te, preds)
        
        resultados_via.append({
            "Modelo": nome,
            "Via": via,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "N_Treino": len(y_tr)
        })
        
    return resultados_via

def main():
    print(f"\n{'='*60}\n  INICIANDO BENCHMARKS (SINGLE-TASK) - ToxiciTOOL 2.0\n{'='*60}")
    
    # Usamos o arquivo campeón (ex: binário 2048 raio 2) para o benchmark
    # Ou o primeiro que encontrar na pasta preprocessed
    arquivos_pkl = sorted(PATHS["preprocessed"].glob("MTL_df_final_*.pkl"))
    if not arquivos_pkl:
        print("  ✗ Nenhum dado processado encontrado. Execute --etapa fingerprints")
        return
        
    arquivo = arquivos_pkl[0]
    print(f"  Usando base: {arquivo.name}")
    df = pd.read_pickle(arquivo)
    
    # Carregar splits (Scaffold é o padrão agora)
    metodo = SPLIT["method"]
    stem = arquivo.stem
    caminho_npz = PATHS[f"splits_{metodo}"] / f"{stem}__indices.npz"
    
    if not caminho_npz.exists():
        print(f"  ✗ Splits não encontrados para {metodo}. Execute --etapa split")
        return
        
    data = np.load(caminho_npz)
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    
    X = np.stack(df["Features"].values)
    X_train, X_test = X[train_idx], X[test_idx]
    
    todos_resultados = []
    
    for via in VIAS:
        col = Y_COLS[via]
        if col in df.columns:
            print(f"  > Processando benchmark: {via}...")
            y_train = df[col].values[train_idx]
            y_test = df[col].values[test_idx]
            
            res_via = treinar_baselines(X_train, y_train, X_test, y_test, via)
            todos_resultados.extend(res_via)
            
    if todos_resultados:
        df_bench = pd.DataFrame(todos_resultados)
        output_path = PATHS["tabelas"] / "MTL_Benchmark_Baselines.xlsx"
        df_bench.to_excel(output_path, index=False)
        print(f"\n✓ Tabela de Benchmarks salva em {output_path}")
        
        # Resumo no console
        print("\n--- RESUMO R2 BENCHMARKS ---")
        pivot = df_bench.pivot(index="Via", columns="Modelo", values="R2")
        print(pivot.to_string())

if __name__ == "__main__":
    main()
