import sys
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

def mol_to_blocks(smiles: str, width: int = 40):
    """Gera uma representação usando blocos ANSI coloridos (Pixel Art)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Text("SMILES Inválido")
        
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    # Linhas mais finas para um visual mais delicado
    dopts.bondLineWidth = 2
    dopts.clearBackground = True
    
    img = Draw.MolToImage(mol, size=(400, 400), options=dopts)
    
    w, h = img.size
    aspect = h / w
    # No terminal, a fonte tem altura quase o dobro da largura.
    char_height = int(width * aspect * 0.5) 
    
    pixel_width = width
    pixel_height = char_height * 2
    
    img = img.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    pixels = img.load()
    
    text = Text()
    
    for y in range(0, pixel_height, 2):
        for x in range(pixel_width):
            r1, g1, b1 = pixels[x, y]
            if y + 1 < pixel_height:
                r2, g2, b2 = pixels[x, y + 1]
            else:
                r2, g2, b2 = 255, 255, 255
                
            # Tratar branco/quase branco como fundo transparente
            bg_threshold = 240
            is_bg1 = (r1 > bg_threshold and g1 > bg_threshold and b1 > bg_threshold)
            is_bg2 = (r2 > bg_threshold and g2 > bg_threshold and b2 > bg_threshold)
            
            if is_bg1 and is_bg2:
                text.append(" ") # Fundo transparente
            elif not is_bg1 and is_bg2:
                # Só o pixel de cima tem cor
                text.append("▀", style=f"rgb({r1},{g1},{b1})")
            elif is_bg1 and not is_bg2:
                # Só o pixel de baixo tem cor
                text.append("▄", style=f"rgb({r2},{g2},{b2})")
            else:
                # Ambos têm cor: usar bloco superior com fundo colorido
                text.append("▀", style=f"rgb({r1},{g1},{b1}) on rgb({r2},{g2},{b2})")
                
        text.append("\n")
        
    return text

if __name__ == "__main__":
    console = Console()
    
    testes = [
        ("Aspirina", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Ibuprofeno", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    ]
    
    for nome, sm in testes:
        arte = mol_to_blocks(sm, width=40)
        panel = Panel(arte, title=f"[bold green]{nome} (Color Blocks)[/bold green]", border_style="cyan", expand=False)
        console.print(panel)
