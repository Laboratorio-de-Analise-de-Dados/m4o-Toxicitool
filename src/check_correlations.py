
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def check_correlations():
    df = pd.read_pickle("data/preprocessed/MTL_df_base_fundida.pkl")
    cols = [c for c in df.columns if c.startswith("log_dl50_")]
    
    corr = df[cols].corr()
    print("--- Task Correlations ---")
    print(corr)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation between Toxicity Tasks (Log LD50)")
    plt.savefig("results/plots/multitask/task_correlation.png")
    print("\nSaved correlation heatmap to results/plots/multitask/task_correlation.png")

    # Check for missing values pattern
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].isna(), cbar=False)
    plt.title("Missing Values Pattern (Black = Present, White = Missing)")
    plt.savefig("results/plots/multitask/missing_values_pattern.png")
    
    # Calculate overlap
    print("\n--- Task Overlap (Number of common molecules) ---")
    overlap = np.zeros((len(cols), len(cols)))
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            overlap[i, j] = df[[c1, c2]].dropna().shape[0]
            
    overlap_df = pd.DataFrame(overlap, index=cols, columns=cols)
    print(overlap_df)

if __name__ == "__main__":
    check_correlations()
