# -----Base-----#
import pandas as pd
import numpy as np

# -----RDKit-----#
try:
    from rdkit import Chem

    # rdMolStandardize mudou de caminho entre versões do RDKit:
    #   conda (rdkit <= 2022) : rdkit.Chem.rdMolStandardize
    #   pip   (rdkit >= 2023) : rdkit.Chem.MolStandardize.rdMolStandardize
    # Tentamos os dois para garantir compatibilidade em qualquer ambiente.
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize  # pip / RDKit >= 2023
    except ImportError:
        from rdkit.Chem import rdMolStandardize  # conda / RDKit < 2023

except Exception as _rdkit_err:
    raise ImportError(
        "Limpeza.py requires RDKit.\n"
        "  pip  : pip install rdkit\n"
        "  conda: conda install -c conda-forge rdkit\n"
        f"Original error: {_rdkit_err!s}"
    )


class Limpeza:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Input:
            - dataframe: pd.DataFrame contendo a representação SMILES e dados de Via

        Nota: o DataFrame original é preservado em _dataframe_original.
        Cada chamada a dados_limpos() trabalha sobre uma cópia fresca,
        tornando a instância segura para reutilização.
        """
        self._dataframe_original = dataframe.copy()
        self.dataframe = self._dataframe_original.copy()

        # Instancia ferramentas do RDKit uma única vez (ganho de performance)
        self.uncharger            = rdMolStandardize.Uncharger()
        self.tautomer_enumerator  = rdMolStandardize.TautomerEnumerator()
        self.metal_disconnector   = rdMolStandardize.MetalDisconnector()
        self.cleanup              = rdMolStandardize.Cleanup

    def dados_limpos(
        self,
        col_smiles: str,
        col_valor: str,
        col_via: str,
        sanitize: bool = True,
        fragmento: bool = False,
        cutoff: float = 0.2,
    ) -> pd.DataFrame:
        """
        Pipeline completo de limpeza molecular.

        Input:
            - col_smiles : nome da coluna SMILES
            - col_valor  : nome da coluna com valor numérico (ex: log_dl50)
            - col_via    : nome da coluna de via de administração
            - sanitize   : aplica pipeline de padronização profunda (RDKit)
            - fragmento  : mantém apenas o maior fragmento orgânico
            - cutoff     : |CV| máximo para aceitar duplicatas (0.2 = 20%)
        """
        # Reinicia o estado a cada chamada — seguro para reutilização da instância
        self.dataframe = self._dataframe_original.copy()

        print(f"1. Iniciando padronização de {len(self.dataframe)} moléculas...")
        df = self.canonical_smiles(col_smiles=col_smiles, sanitize=sanitize)

        df.dropna(subset=["smiles"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if fragmento:
            print("2. Extraindo fragmentos principais...")
            df = self.fragmento_principal()
            df.dropna(subset=["smiles"], inplace=True)

        print("3. Removendo duplicatas (respeitando a Via de Administração)...")
        df = self.limpa_repetidos_inteligente(
            col_valor=col_valor,
            col_via=col_via,
            cutoff=cutoff,
        )

        df.reset_index(drop=True, inplace=True)
        print(f"Processo concluído. Dataset final: {len(df)} linhas.")
        return df

    # ─────────────────────────────────────────────────────────────────
    # MÉTODOS DE PROCESSAMENTO
    # ─────────────────────────────────────────────────────────────────

    def canonical_smiles(self, col_smiles: str, sanitize: bool = True) -> pd.DataFrame:
        """
        Converte SMILES brutos para SMILES canônicos padronizados.
        SMILES inválidos ou que falham na sanitização viram NaN.
        """
        canonical = []

        for raw_smile in self.dataframe[col_smiles]:
            try:
                mol = Chem.MolFromSmiles(str(raw_smile))
                if mol is None:
                    raise ValueError("SMILES inválido")

                if sanitize:
                    mol = self.cleanup(mol)
                    mol = self.metal_disconnector.Disconnect(mol)
                    mol = self.uncharger.uncharge(mol)
                    mol = self.tautomer_enumerator.Canonicalize(mol)

                smile = Chem.MolToSmiles(mol, isomericSmiles=False)

            except Exception:
                smile = np.nan

            canonical.append(smile)

        self.dataframe["smiles"] = canonical

        if col_smiles != "smiles":
            self.dataframe.drop(columns=[col_smiles], inplace=True, errors="ignore")

        return self.dataframe

    def fragmento_principal(self) -> pd.DataFrame:
        """
        Mantém apenas o maior fragmento orgânico (remove contra-íons, água, sais).
        Requer que a coluna 'smiles' já exista (chamar após canonical_smiles).
        """
        fragmentos = []
        for smi in self.dataframe["smiles"]:
            try:
                mol      = Chem.MolFromSmiles(smi)
                mol_frag = rdMolStandardize.FragmentParent(mol)
                fragmentos.append(Chem.MolToSmiles(mol_frag, isomericSmiles=False))
            except Exception:
                fragmentos.append(np.nan)

        self.dataframe["smiles"] = fragmentos
        return self.dataframe

    def limpa_repetidos_inteligente(
        self,
        col_valor: str,
        col_via: str,
        cutoff: float = 0.2,
    ) -> pd.DataFrame:
        """
        Deduplicação inteligente por (smiles + via).

        Regras:
            - 1 registro    → mantém como está
            - |CV| <= cutoff → dados consistentes entre labs → retorna média
            - |CV| >  cutoff → dados inconsistentes → descarta o grupo

        
        """
        cols_grupo = ["smiles", col_via]
        resultados = []

        for _, grupo in self.dataframe.groupby(cols_grupo):
            if len(grupo) == 1:
                resultados.append(grupo.iloc[0:1])
                continue

            valores = pd.to_numeric(grupo[col_valor], errors="coerce").dropna()
            if len(valores) == 0:
                continue

            media  = float(valores.mean())
            desvio = float(valores.std())

            # FIX: abs() corrige CV negativo em escala log
            cv = 0.0 if media == 0 else abs(desvio / media)

            if cv <= cutoff:
                linha           = grupo.iloc[0:1].copy()
                linha[col_valor] = media
                resultados.append(linha)
            # cv > cutoff → grupo descartado (labs inconsistentes)

        if not resultados:
            return pd.DataFrame(columns=self.dataframe.columns)

        df_limpo = pd.concat(resultados, ignore_index=True)
        return df_limpo