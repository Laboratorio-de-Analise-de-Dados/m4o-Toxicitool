import sys
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from rich.console import Console
from rich.panel import Panel

def mol_to_braille(smiles: str, width: int = 40):
    """Gera uma representação Braille/ASCII de um SMILES usando RDKit e Pillow."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "SMILES Inválido"
        
    # Gera a imagem 2D. Fundo branco, traços pretos
    # Removemos anotações de hidrogênio e aumentamos a espessura da linha para melhor visibilidade
    dopts = Draw.rdMolDraw2D.MolDrawOptions()
    dopts.bondLineWidth = 1
    dopts.clearBackground = True
    
    # Renderiza a imagem num tamanho padrão
    img = Draw.MolToImage(mol, size=(400, 400), options=dopts)
    
    # Converte para escala de cinza
    img = img.convert("L")
    w, h = img.size
    
    # 1 caractere Braille representa um bloco de 2x4 pixels.
    # Ajustamos a altura baseada no aspect ratio da imagem e na proporção do Braille (2:4)
    aspect = h / w
    char_height = int(width * aspect * 0.5) 
    
    # Redimensiona a imagem para corresponder à grade de pixels do Braille
    pixel_width = width * 2
    pixel_height = char_height * 4
    
    # Usa LANCZOS para suavizar o desenho ao diminuir
    img = img.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
    pixels = img.load()
    
    # Mapa de bits do caractere Braille Unicode (Base: 0x2800)
    # Coluna 1 | Coluna 2
    # bit 0(1) | bit 3(8)
    # bit 1(2) | bit 4(16)
    # bit 2(4) | bit 5(32)
    # bit 6(64)| bit 7(128)
    dot_map = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80]
    ]
    
    threshold = 200 # Limiar de cor: < 200 é considerado "tinta" (linhas do RDKit)
    
    linhas = []
    for y in range(0, pixel_height, 4):
        row_str = ""
        for x in range(0, pixel_width, 2):
            char_val = 0x2800 # Base Braille vazia (⠀)
            for dy in range(4):
                for dx in range(2):
                    if x + dx < pixel_width and y + dy < pixel_height:
                        # Se o pixel for escuro, ativamos o ponto Braille correspondente
                        if pixels[x + dx, y + dy] < threshold:
                            char_val += dot_map[dy][dx]
            row_str += chr(char_val)
        linhas.append(row_str)
        
    return "\n".join(linhas)

if __name__ == "__main__":
    console = Console()
    
    # Teste com Aspirina e Cafeína
    testes = [
        ("Aspirina", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Cafeína", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    ]
    
    for nome, sm in testes:
        arte = mol_to_braille(sm, width=35)
        panel = Panel(arte, title=f"[bold green]{nome}[/bold green]", border_style="cyan", expand=False)
        console.print(panel)
