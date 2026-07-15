import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from PIL import Image

# Ajustar sys.path para encontrar os módulos locais
_ESTE_ARQUIVO = Path(__file__).resolve()
RAIZ = _ESTE_ARQUIVO.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from tracer import tracer, trace_step
from src.config import VIAS, PATHS
from src.representation.Representacao import Representacao
from src.preprocessing.Limpeza import Limpeza
from src.analysis.classificacao import log_dl50_para_classe_ghs

# ==========================================
# FUNÇÕES DE SUPORTE
# ==========================================

def get_model_stats(model_path: Path):
    try:
        log_path = PATHS["log_excel"]
        if not log_path.exists(): return None
        df_log = pd.read_excel(log_path)
        match = df_log[df_log["caminho_modelo"].str.contains(model_path.name, na=False)]
        if match.empty: return None
        return match.iloc[0].to_dict()
    except Exception:
        return None

def display_model_stats(stats):
    if not stats:
        tracer.console.print("[yellow]⚠ Aviso: Relatório estatístico não encontrado.[/yellow]")
        return
    table = Table(title="[bold blue]Performance Histórica (Teste)[/bold blue]", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Via de Exposição")
    table.add_column("R²", justify="right")
    table.add_column("MAE", justify="right")
    for via in VIAS:
        r2 = stats.get(f"r2_{via}", np.nan)
        mae = stats.get(f"mae_{via}", np.nan)
        r2_str = f"{r2:.3f}" if pd.notna(r2) and not isinstance(r2, str) else "N/A"
        mae_str = f"{mae:.3f}" if pd.notna(mae) and not isinstance(mae, str) else "N/A"
        table.add_row(via.replace("_", " ").title(), r2_str, mae_str)
    tracer.console.print(table)
    tracer.console.print(f"[dim]Método: {stats.get('metodo_split', 'N/A')}[/dim]\n")

def get_chemical_diagnostic(mol):
    try:
        if mol is None or pd.isna(mol): return None
        return {
            "Fórmula": rdMolDescriptors.CalcMolFormula(mol),
            "Massa Molar": f"{Descriptors.MolWt(mol):.2f} g/mol",
            "LogP": f"{Descriptors.MolLogP(mol):.2f}",
            "H-Donors/Acc": f"{Descriptors.NumHDonors(mol)}/{Descriptors.NumHAcceptors(mol)}",
            "TPSA": f"{Descriptors.TPSA(mol):.2f} Å²",
            "Rotatable": str(Descriptors.NumRotatableBonds(mol)),
            "Aromatic": str(Descriptors.NumAromaticRings(mol))
        }
    except: return None

def save_mol_image(mol, output_path: Path):
    try:
        if mol is None or pd.isna(mol): return False
        dopts = Draw.rdMolDraw2D.MolDrawOptions()
        dopts.bondLineWidth = 2
        img = Draw.MolToImage(mol, size=(600, 600), options=dopts)
        img.save(output_path)
        return True
    except: return False

@trace_step("Analisando configuração do modelo")
def parse_model_config(model_path: Path):
    filename = model_path.name
    config = {"use_count": "quantitativo" in filename, "fpSize": 2048, "radius": 2, "use_desc": "__desc" in filename}
    if "4096bits" in filename: config["fpSize"] = 4096
    elif "8192bits" in filename: config["fpSize"] = 8192
    if "raio3" in filename: config["radius"] = 3
    elif "raio5" in filename: config["radius"] = 5
    return config

@trace_step("Carregando modelo keras")
def load_model(model_path: Path):
    from src.modeling.modelagem_MTL import CamadaIncerteza, get_huber_uncertainty_loss
    custom_objects = {"CamadaIncerteza": CamadaIncerteza}
    for i in range(len(VIAS)):
        custom_objects[f"huber_uncert_task_{i}"] = get_huber_uncertainty_loss(i, None)
    return tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)

@trace_step("Processando input e gerando features")
def extract_features(df_input: pd.DataFrame, config: dict, model_input_shape):
    df = df_input.copy()
    df = df.dropna(subset=["SMILES"])
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df = df[df["SMILES"] != ""].reset_index(drop=True)
    if df.empty: raise ValueError("Nenhum SMILES válido.")
    
    tracer.console.print("  [Preproc] Aplicando sanitização química...")
    limp = Limpeza(dataframe=df)
    df_canon = limp.canonical_smiles(col_smiles="SMILES", sanitize=True)
    df["SMILES_LIMPO"] = df_canon["smiles"]
    df["SMILES_PARA_MODELO"] = df["SMILES_LIMPO"].fillna(df["SMILES"])
    
    rep = Representacao(dataframe=df)
    df = rep.mol_to_frame(col_smiles="SMILES_PARA_MODELO")
    df = rep.fp_Morgan(col_frames="ROMol", radius=config["radius"], fpSize=config["fpSize"], use_count=config["use_count"])
    df = rep.bitVect_to_array("Fingerprint")
    if config["use_desc"]:
        df = rep.calcular_descritores(col_smiles="SMILES_PARA_MODELO", lista_descritores="todos")
        df = rep.concatenar_descritores()
    
    # FILTRAR LINHAS QUE FALHARAM NA GERAÇÃO DE FEATURES
    df["IsValid"] = df["Features"].apply(lambda x: isinstance(x, np.ndarray))
    invalid_count = len(df) - df["IsValid"].sum()
    if invalid_count > 0:
        tracer.console.print(f"[bold yellow]⚠ Aviso: {invalid_count} compostos falharam na geração de features e serão ignorados.[/bold yellow]")
    df = df[df["IsValid"]].reset_index(drop=True)
    
    if df.empty: raise ValueError("Todos os compostos falharam na geração de features.")
    
    features = np.stack(df["Features"].values).astype(np.float32)
    return features, df

@trace_step("Executando predição")
def predict_toxicity(model, features):
    raw_predictions = model.predict(features, verbose=0)
    if not isinstance(raw_predictions, list): raw_predictions = [raw_predictions]
    predictions = {}
    for i, name in enumerate(model.output_names):
        clean_name = name.replace("dense_", "").split("/")[0]
        target_via = next((v for v in VIAS if v in clean_name or clean_name in v), name)
        predictions[target_via] = raw_predictions[i].flatten()
    return predictions

def generate_report(df_processed, predictions, model_stats):
    output_dir = Path("demo/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_list = []
    for i_pos, (idx, row) in enumerate(df_processed.iterrows()):
        descricao = row.get("Descricao", f"Composto {i_pos+1}")
        smi = row["SMILES_PARA_MODELO"]
        mol = row.get("ROMol")
        if mol is None or pd.isna(mol): mol = Chem.MolFromSmiles(str(smi))
        
        tracer.console.print(f"\n[bold cyan]▶ Relatório Toxicológico: {descricao}[/bold cyan]")
        diag = get_chemical_diagnostic(mol)
        diag_text = Text()
        if diag:
            for k, v in diag.items():
                diag_text.append(f"{k}: ", style="bold green"); diag_text.append(f"{v}\n", style="white")
        
        img_filename = f"mol_{i_pos+1}_{int(time.time())}.png"
        img_path = output_dir / img_filename
        save_mol_image(mol, img_path)
        diag_text.append(f"\nImagem: {img_path.name}", style="dim italic")
             
        panel_visual = Panel(diag_text, title="[bold green]Diagnóstico Químico[/bold green]", border_style="green", expand=False, padding=(1, 2))
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", expand=False)
        table.add_column("Via"); table.add_column("Log(LD50)", justify="right"); table.add_column("LD50 (mg/kg)", justify="right"); table.add_column("GHS", justify="center")
        
        res = {"Descricao": descricao, "SMILES": smi}
        if diag: res.update(diag)
        for via in VIAS:
            log_ld50 = predictions.get(via, [np.nan])[i_pos]
            ld50 = 10 ** log_ld50 if not np.isnan(log_ld50) else np.nan
            ghs = log_dl50_para_classe_ghs(np.array([log_ld50]))[0] if not np.isnan(log_ld50) else "N/A"
            table.add_row(via.replace("_", " ").title(), f"{log_ld50:.3f}", f"{ld50:.2f}", f"GHS {ghs}")
            res[f"{via}_log_ld50"] = log_ld50; res[f"{via}_ld50"] = ld50; res[f"{via}_ghs"] = ghs
            
        tracer.console.print(Columns([panel_visual, Panel(table, title="[bold magenta]Predições MTL[/bold magenta]", border_style="magenta")], expand=True))
        results_list.append(res)
        time.sleep(0.1)

    df_res = pd.DataFrame(results_list)
    export_path = output_dir / f"predicoes_{int(time.time())}.csv"
    df_res.to_csv(export_path, index=False)
    tracer.console.print(f"\n[bold green]✓ Resultados salvos em:[/bold green] {export_path}")

def main():
    tracer.console.print("\n[bold white on blue] ToxiciTOOL - Demonstração [/bold white on blue]\n")
    input_dir = Path("demo/input")
    arquivos_input = sorted(list(input_dir.glob("*.csv")))
    if not arquivos_input: sys.exit(1)
    for i, arq in enumerate(arquivos_input): tracer.console.print(f"  [bold cyan]{i+1}.[/bold cyan] {arq.name}")
    esc_in = Prompt.ask("\n[bold green]? Arquivo[/bold green]", choices=[str(i+1) for i in range(len(arquivos_input))], default="1")
    p_input = arquivos_input[int(esc_in) - 1]
    df_input = pd.read_csv(p_input)
    
    model_dir = Path("results/melhor_modelo")
    modelos = sorted(list(model_dir.glob("*.keras")))
    for i, mod in enumerate(modelos): tracer.console.print(f"  [bold cyan]{i+1}.[/bold cyan] {mod.name}")
    esc_mod = Prompt.ask("\n[bold green]? Modelo[/bold green]", choices=[str(i+1) for i in range(len(modelos))], default="1")
    p_model = modelos[int(esc_mod) - 1]
    display_model_stats(get_model_stats(p_model))
    
    tracer.reset_tree("Processamento"); tracer.start_live()
    try:
        model = load_model(p_model)
        features, df_processed = extract_features(df_input, parse_model_config(p_model), model.input_shape)
        predictions = predict_toxicity(model, features)
    except Exception as e: tracer.console.print(f"[bold red]Falha: {e}[/bold red]"); sys.exit(1)
    finally: tracer.stop_live()
    generate_report(df_processed, predictions, get_model_stats(p_model))

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
