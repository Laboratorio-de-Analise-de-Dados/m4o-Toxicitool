
import polars as pl
import os
import glob
import numpy as np

def diagnose_data():
    raw_files = glob.glob("data/data_raw/*.csv")
    
    summary_list = []
    
    for file in raw_files:
        print(f"Analyzing {file}...")
        df = pl.read_csv(file)
        
        # Rename columns based on common patterns if possible, 
        # but let's just find the toxicity and smiles columns
        cols = df.columns
        smiles_col = [c for c in cols if "SMILES" in c.upper()][0]
        tox_col = [c for c in cols if "TOXICITY" in c.upper() or "VALUE" in c.upper()][0]
        
        df = df.select([
            pl.col(smiles_col).alias("smiles"),
            pl.col(tox_col).cast(pl.Float64, strict=False).alias("ld50")
        ]).drop_nulls()
        
        # Log distribution
        df = df.with_columns([
            pl.col("ld50").map_elements(lambda x: np.log10(x) if x > 0 else None, return_dtype=pl.Float64).alias("log_ld50")
        ]).drop_nulls("log_ld50")
        
        stats = {
            "file": os.path.basename(file),
            "count": df.height,
            "unique_smiles": df.select("smiles").n_unique(),
            "min_log": df["log_ld50"].min(),
            "max_log": df["log_ld50"].max(),
            "mean_log": df["log_ld50"].mean(),
            "std_log": df["log_ld50"].std(),
            "p01": df["log_ld50"].quantile(0.01),
            "p99": df["log_ld50"].quantile(0.99),
        }
        
        # Check duplicates
        dupes = df.group_by("smiles").agg([
            pl.count().alias("n"),
            pl.col("log_ld50").std().alias("std_dev"),
            pl.col("log_ld50").mean().alias("mean_val")
        ]).filter(pl.col("n") > 1)
        
        stats["n_duplicates"] = dupes.height
        stats["max_std_dupes"] = dupes["std_dev"].max()
        stats["mean_std_dupes"] = dupes["std_dev"].mean()
        
        summary_list.append(stats)
        
    summary_df = pl.DataFrame(summary_list)
    print("\n--- Global Statistics ---")
    print(summary_df)
    
    # Combined analysis
    all_dfs = []
    for file in raw_files:
        df = pl.read_csv(file)
        cols = df.columns
        smiles_col = [c for c in cols if "SMILES" in c.upper()][0]
        tox_col = [c for c in cols if "TOXICITY" in c.upper() or "VALUE" in c.upper()][0]
        df = df.select([
            pl.col(smiles_col).alias("smiles"),
            pl.col(tox_col).cast(pl.Float64, strict=False).alias("ld50")
        ]).drop_nulls()
        df = df.with_columns(pl.lit(os.path.basename(file)).alias("source"))
        all_dfs.append(df)
        
    combined = pl.concat(all_dfs)
    combined = combined.with_columns([
        pl.col("ld50").map_elements(lambda x: np.log10(x) if x > 0 else None, return_dtype=pl.Float64).alias("log_ld50")
    ]).drop_nulls("log_ld50")
    
    print("\n--- Overlap Analysis ---")
    overlap = combined.group_by("smiles").agg([
        pl.count().alias("n_sources"),
        pl.col("log_ld50").std().alias("std_dev_global"),
        pl.col("source").unique().alias("sources")
    ]).filter(pl.col("n_sources") > 1).sort("std_dev_global", descending=True)
    
    print(f"Molecules in multiple sources: {overlap.height}")
    print("Molecules with high variance between sources (Top 10):")
    print(overlap.head(10))

if __name__ == "__main__":
    diagnose_data()
