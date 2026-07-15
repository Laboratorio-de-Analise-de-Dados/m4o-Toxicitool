import sys
import os
import yaml
import gc
import time
from pathlib import Path

# Ajustar sys.path para encontrar os módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tracer import tracer, trace_step
from preprocessing.Processa import ProcessadorDados
from deRedes.constroi_rede import ConstrutorRede
from deRedes.enriquecimento import AnalisadorEnriquecimento

# Caminhos Base
_ESTE_ARQUIVO = Path(__file__).resolve()
RAIZ          = _ESTE_ARQUIVO.parent.parent.parent
TABELAS_BASE  = RAIZ / "results" / "cytoscape" / "tabelas"
DADOS_PROCESSADOS = RAIZ / "data" / "processed" / "redes"

def carregar_configuracao():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def etapa_rede_e_analise(gene_id: str, direction: str, limit: int, df_polars, construtor: ConstrutorRede, analisador: AnalisadorEnriquecimento):
    # Usar o tracer manualmente para injetar o limite e cutoff no título da etapa
    nome_etapa = f"Construção de Rede (Limite: {limit}, Cutoff: {construtor.string_cutoff})"
    node = tracer.push(nome_etapa)
    start_time = time.time()
    
    try:
        # Construir Rede
        network_title = construtor.construir_rede_gene(gene_id, direction, limit, df_polars)
        
        if network_title:
            label_analise = f"{gene_id}_{direction.replace('/', '_')}_lim{limit}"
            try:
                raw_file = analisador.executa_enriquecimento_string(network_title, label_analise)
                if raw_file:
                    analisador.filtrar_resultados(raw_file)
            except Exception as e:
                tracer.console.print(f"\n[bold yellow]⚠ [WARNING][/bold yellow] Pulo de enriquecimento para {network_title} devido a erro: {e}")
        else:
            tracer.console.print(f"    [dim]󰛨 [SKIP][/dim] Pulo de análise para {gene_id} {direction} por falta de rede.")
            
        duration = time.time() - start_time
        tracer.pop(node, nome_etapa, success=True, duration=duration)
        
    except Exception as e:
        duration = time.time() - start_time
        tracer.pop(node, nome_etapa, success=False, duration=duration, error=e)
        raise e

def run_pipeline():
    config = carregar_configuracao()
    
    # Inicializar Classes
    processador = ProcessadorDados(output_dir=DADOS_PROCESSADOS)
    construtor = ConstrutorRede(string_cutoff=config.get('string_cutoff', 0.5))
    analisador = AnalisadorEnriquecimento(base_output_dir=TABELAS_BASE)

    if not construtor.conectar_cytoscape():
        tracer.console.print("[bold red]󰅙 Cytoscape não está acessível.[/bold red]")
        return

    for analysis in config['analyses']:
        analysis_id = analysis['id']
        excel_path = RAIZ / analysis['source_excel']
        up_sheet = analysis['up_sheet']
        down_sheet = analysis['down_sheet']
        directions = analysis['directions']
        
        tracer.console.print(f"\n[bold blue]󰓗 Iniciando Bloco de Análise: {analysis_id}[/bold blue]")
        
        if not excel_path.exists():
             tracer.console.print(f"[bold red]󰅙 Arquivo Excel não encontrado: {excel_path}[/bold red]")
             continue

        # 1. Processar dados fora do Live para que a mensagem de extração fique no histórico
        resultados_processamento = processador.processar_analise(
            analysis_id, excel_path, up_sheet, down_sheet, directions
        )
        
        # 2. Iterar sobre as direções (UP, DOWN, UP/DOWN)
        for direction in directions:
            if direction not in resultados_processamento:
                continue
            
            df_polars = resultados_processamento[direction]
            
            # --- FASE PERSISTENTE NO TERMINAL ---
            # Exibir a tabela Polars ANTES de iniciar a box do pipeline
            tracer.display_df(df_polars, title=f"Dados Preparados: {analysis_id} ({direction})")
            time.sleep(3)
            
            # --- FASE LIVE (PROGRESS BOX) ---
            # Reseta o tracer para este bloco específico de direção
            tracer.reset_tree(f"Redes: {analysis_id} - {direction}")
            tracer.start_live()
            
            try:
                for limit in config['network_limits']:
                    etapa_rede_e_analise(analysis_id, direction, limit, df_polars, construtor, analisador)
                    gc.collect()
                    tracer.console.print(f"[dim]    󰔟 Pausa de 3s para estabilização...[/dim]")
                    time.sleep(3)
            finally:
                tracer.stop_live() # Finaliza a box para esta direção e imprime no terminal
            
            # Pausa entre direções
            tracer.console.print(f"[green]󰄬 Bloco {direction} concluído.[/green]")
            time.sleep(2)

        tracer.console.print(f"\n[bold green]✅ TODAS AS REDES PARA {analysis_id} FORAM GERADAS.[/bold green]")
        time.sleep(5)

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        if tracer.live:
            tracer.stop_live()
        tracer.console.print(f"\n[bold red]󰅙 Erro Crítico:[/bold red] {e}")
    finally:
        tracer.console.print("\n[bold green]󰄬 Processo de execução finalizado.[/bold green]")
