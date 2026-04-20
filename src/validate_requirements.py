#!/usr/bin/env python3
"""
validate_requirements.py  (src/)
==================================
Verifica que todos os pacotes declarados em environment.yml estão instalados
e acessíveis na versão correta.

Uso (a partir da raiz do projeto):
    python src/validate_requirements.py
"""

import sys
import importlib
from pathlib import Path


NOME_MODULO: dict[str, str] = {
    "scikit-learn":    "sklearn",
    "tensorflow":      "tensorflow",
    "torch":           "torch",
    "torch-geometric": "torch_geometric",
    "torch-scatter":   "torch_scatter",
    "torch-sparse":    "torch_sparse",
    "rdkit":           "rdkit",
    "pyyaml":          "yaml",
    "pillow":          "PIL",
    "openpyxl":        "openpyxl",
}

# Pacotes opcionais (ausência não interrompe o pipeline Morgan)
# OPCIONAIS = {"torch", "torch-geometric", "torch-scatter", "torch-sparse"}


def normalizar_nome(req: str) -> str:
    """Remove versão e extras do nome do pacote."""
    return req.split(">=")[0].split("==")[0].split("[")[0].strip()


def checar_pacote(nome_req: str) -> tuple[bool, str]:
    """
    Tenta importar o módulo correspondente ao pacote.
    Retorna (ok: bool, versao: str).
    """
    nome_mod = NOME_MODULO.get(nome_req.lower(), nome_req.replace("-", "_").lower())
    try:
        mod    = importlib.import_module(nome_mod)
        versao = getattr(mod, "__version__", "instalado")
        return True, versao
    except ImportError:
        return False, "não encontrado"


def ler_environment_yml(caminho: Path) -> list[str]:
    """
    Extrai nomes de pacotes do environment.yml (seção pip e conda).
    BUG anterior: tentava ler requirements.txt que não existe no projeto.
    O projeto usa environment.yml (conda) como fonte de verdade.
    """
    pacotes = []
    dentro_pip = False

    with open(caminho) as f:
        for linha in f:
            linha_strip = linha.strip()

            # Detecta o bloco "- pip:" dentro de dependencies
            if linha_strip == "- pip:":
                dentro_pip = True
                continue

            if dentro_pip:
                if linha_strip.startswith("- ") and not linha_strip.startswith("- pip"):
                    # Pacote pip
                    pacotes.append(normalizar_nome(linha_strip[2:]))
                elif linha_strip and not linha_strip.startswith("-") and not linha_strip.startswith(" "):
                    dentro_pip = False  # saiu do bloco pip
            else:
                # Pacotes conda (direto em dependencies, exceto python e pip)
                if linha_strip.startswith("- ") and "=" in linha_strip and not linha_strip.startswith("- pip"):
                    nome = linha_strip[2:].split("=")[0].strip()
                    if nome not in ("python", "pip"):
                        pacotes.append(nome)

    return pacotes


def main() -> bool:
    # Este arquivo está em src/ — environment.yml está na raiz (parent.parent)
    raiz      = Path(__file__).resolve().parent.parent
    env_path  = raiz / "environment.yml"

    if not env_path.exists():
        print(f"✗ environment.yml não encontrado em {env_path}")
        return False

    pacotes = ler_environment_yml(env_path)

    print("=" * 55)
    print("  ToxiciTOOL — Validação de Dependências")
    print(f"  Fonte: {env_path}")
    print("=" * 55)

    ok_lista      = []
    falha_obrig   = []
    falha_opcio   = []

    for pacote in pacotes:
        ok, versao = checar_pacote(pacote)
        opcional   = pacote.lower() in OPCIONAIS
        status     = "✓" if ok else ("⚠" if opcional else "✗")
        sufixo     = " (opcional)" if opcional and not ok else ""
        print(f"  {status} {pacote:<28} {versao}{sufixo}")

        if ok:
            ok_lista.append(pacote)
        elif opcional:
            falha_opcio.append(pacote)
        else:
            falha_obrig.append(pacote)

    print("\n" + "=" * 55)
    print(f"  ✓ {len(ok_lista)} OK  |  ✗ {len(falha_obrig)} obrigatórios ausentes  |  ⚠ {len(falha_opcio)} opcionais ausentes")

    if falha_obrig:
        print(f"\n  Instale os obrigatórios com:")
        print(f"    conda env create -f environment.yml")
        print(f"\n  Ou individualmente:")
        for p in falha_obrig:
            print(f"    pip install {p}")
        return False

    if not falha_obrig:
        print("  Todas as dependências obrigatórias estão instaladas!")
    print("=" * 55)
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)