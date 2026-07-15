import time
from functools import wraps
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from typing import Optional, Any
import polars as pl

class ExecutionTracer:
    """Responsável por rastrear e renderizar o mapa de execução com feedback profissional."""
    
    def __init__(self, title: str):
        self.console = Console()
        self.title = title
        self.root = Tree(f"[bold cyan]󱓞 {title}[/bold cyan]")
        self.stack = [self.root]
        self.live: Optional[Live] = None

    def start_live(self):
        """Inicia a exibição Live no terminal."""
        if self.live is None:
            self.live = Live(
                self.get_renderable(),
                console=self.console,
                refresh_per_second=10,
                transient=False
            )
            self.live.start()

    def stop_live(self):
        """Finaliza a exibição Live."""
        if self.live:
            self.live.stop()
            self.live = None

    def get_renderable(self):
        """Retorna o painel formatado para renderização."""
        return Panel(
            self.root,
            title=f"[bold white]{self.title}[/bold white]",
            border_style="bright_blue",
            box=box.DOUBLE_EDGE,
            padding=(1, 2)
        )

    def push(self, name: str):
        """Adiciona um passo ao rastreamento."""
        node = self.stack[-1].add(f"[bold yellow]󱎫 {name}...[/bold yellow]")
        self.stack.append(node)
        if self.live:
            self.live.update(self.get_renderable())
        return node

    def pop(self, node, name: str, success: bool, duration: float, error: Optional[Exception] = None):
        """Finaliza um passo do rastreamento e reporta erros."""
        self.stack.pop()
        if success:
            node.label = f"[bold green]󰄬 {name}[/bold green] [dim]({duration:.3f}s)[/dim]"
        else:
            node.label = f"[bold red]󰅙 {name}[/bold red] [dim]({duration:.3f}s)[/dim]"
            if error:
                # O Live será pausado para mostrar o erro se necessário, 
                # mas o tracer.pop geralmente é chamado dentro do decorator
                pass
        
        if self.live:
            self.live.update(self.get_renderable())

    def reset_tree(self, new_title: str):
        """Reseta a árvore para um novo bloco de análise."""
        self.title = new_title
        self.root = Tree(f"[bold cyan]󱓞 {new_title}[/bold cyan]")
        self.stack = [self.root]

    def display_df(self, df: pl.DataFrame, title: str = "Amostra de Dados"):
        """Exibe um Polars DataFrame de forma elegante e persistente."""
        # Se houver um live rodando, paramos para imprimir a tabela no histórico
        is_live = self.live is not None
        if is_live:
            self.stop_live()

        table = Table(title=f"[bold blue]{title}[/bold blue]", box=box.ROUNDED, header_style="bold magenta")
        sample_df = df.head(5)
        
        for col in sample_df.columns:
            table.add_column(str(col))
        
        for row in sample_df.iter_rows():
            table.add_row(*[str(val) if val is not None else "" for val in row])
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print(f"[dim]Total de linhas: {df.height} | Total de colunas: {df.width}[/dim]\n")
        
        if is_live:
            self.start_live()

# Instância global do tracer
tracer = ExecutionTracer("GENE FLOW - Transcriptomics Pipeline")

def trace_step(name: str):
    """Decorador para rastrear o ciclo de vida e mostrar status em tempo real."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            node = tracer.push(name)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                tracer.pop(node, name, success=True, duration=duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                tracer.pop(node, name, success=False, duration=duration, error=e)
                # Se houver erro, paramos o live para mostrar o traceback limpo
                if tracer.live:
                    tracer.stop_live()
                tracer.console.print(f"\n[bold red]── ERRO EM: {name} ──[/bold red]")
                tracer.console.print(f"[red]{str(e)}[/red]\n")
                raise e
        return wrapper
    return decorator
