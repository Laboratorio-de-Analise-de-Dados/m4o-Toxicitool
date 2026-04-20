# -----Base-----#
import pandas as pd
import numpy as np

# -----RDKit-----#
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import PandasTools, Descriptors, rdFingerprintGenerator
except Exception as _rdkit_err:
    raise ImportError(
        "Representacao.py requires RDKit.\n"
        "Install via conda: `conda install -c conda-forge rdkit`\n"
        f"Original error: {_rdkit_err!s}"
    )


class Representacao:
    """
    Gera representações vetoriais de moléculas a partir de SMILES.

    Fluxo com descritores (recomendado):
        rep = Representacao(dataframe=df)
        df  = rep.fingerprint(col_smiles='smiles', radius=2, fpSize=2048, use_count=True)
        df  = rep.calcular_descritores(col_smiles='smiles')
        df  = rep.concatenar_descritores()   # combina 'Features' + 'Descritores' → 'Features'

    Fluxo apenas fingerprints (legado):
        rep = Representacao(dataframe=df)
        df  = rep.fingerprint(col_smiles='smiles', radius=2, fpSize=2048)
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    # ─────────────────────────────────────────────────────────────────
    # PIPELINE COMPLETO DE FINGERPRINTS
    # ─────────────────────────────────────────────────────────────────

    def fingerprint(
        self,
        col_smiles: str,
        radius: int = 2,
        fpSize: int = 2048,
        use_count: bool = False,
    ) -> pd.DataFrame:
        """SMILES → ROMol → Fingerprint → array numpy em 'Features'."""
        df = self.mol_to_frame(col_smiles=col_smiles)
        df = self.fp_Morgan(col_frames="ROMol", radius=radius, fpSize=fpSize, use_count=use_count)
        df.dropna(subset=["Fingerprint"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = self.bitVect_to_array("Fingerprint")
        return df

    # ─────────────────────────────────────────────────────────────────
    # SMILES → ROMol
    # ─────────────────────────────────────────────────────────────────

    def mol_to_frame(self, col_smiles: str) -> pd.DataFrame:
        """Adiciona coluna 'ROMol'. SMILES inválidos geram NaN."""
        PandasTools.AddMoleculeColumnToFrame(frame=self.dataframe, smilesCol=col_smiles)
        return self.dataframe

    # ─────────────────────────────────────────────────────────────────
    # FINGERPRINTS MORGAN
    # ─────────────────────────────────────────────────────────────────

    def fp_Morgan(
        self,
        col_frames: str,
        radius: int = 2,
        fpSize: int = 2048,
        use_count: bool = False,
    ) -> pd.DataFrame:
        """
        Gera fingerprints Morgan.
            use_count=False → ExplicitBitVect  (binário)
            use_count=True  → UIntSparseIntVect (contagem)
        """
        morgan_lista = []
        for idx in self.dataframe.index:
            try:
                mol = self.dataframe[col_frames].loc[idx]
                if mol is None:
                    raise ValueError
                gen    = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
                morgan = gen.GetCountFingerprint(mol) if use_count else gen.GetFingerprint(mol)
            except Exception:
                morgan = np.nan
            morgan_lista.append(morgan)

        self.dataframe["Fingerprint"] = morgan_lista
        return self.dataframe

    def bitVect_to_array(self, col_fp: str) -> pd.DataFrame:
        """
        Converte fingerprints RDKit para arrays numpy densos em 'Features'.

        ExplicitBitVect   → GetNumBits()  → dtype int8
        UIntSparseIntVect → GetLength()   → dtype float32
        """
        fp_arrays = []
        for idx in self.dataframe.index:
            try:
                fp_obj = self.dataframe[col_fp].loc[idx]
                if fp_obj is np.nan or fp_obj is None:
                    raise ValueError("Fingerprint ausente")

                if hasattr(fp_obj, "GetNumBits"):
                    n_bits = int(fp_obj.GetNumBits())
                    dtype  = np.int8
                elif hasattr(fp_obj, "GetLength"):
                    n_bits = int(fp_obj.GetLength())   # ← correto para count FPs
                    dtype  = np.float32
                else:
                    raise ValueError(f"Tipo desconhecido: {type(fp_obj)}")

                if n_bits == 0:
                    raise ValueError("Fingerprint de tamanho zero")

                fp_arr = np.zeros((n_bits,), dtype=dtype)
                DataStructs.ConvertToNumpyArray(fp_obj, fp_arr)
            except Exception:
                fp_arr = np.nan

            fp_arrays.append(fp_arr)

        self.dataframe["Features"] = fp_arrays
        return self.dataframe

    # ─────────────────────────────────────────────────────────────────
    # DESCRITORES FÍSICO-QUÍMICOS
    # ─────────────────────────────────────────────────────────────────

    def calcular_descritores(
        self,
        col_smiles: str,
        lista_descritores: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calcula descritores físico-químicos via RDKit.Descriptors e os armazena
        em uma coluna 'Descritores' (array numpy float32 por molécula), além de
        salvar um escalonador RobustScaler ajustado em self.scaler.
        
        Parâmetros:
            col_smiles        : coluna com SMILES canônicos
            lista_descritores : lista de nomes de Chem.Descriptors a calcular.
                                Se None, usa DESCRITORES['lista'] de config.py.
        """
        from sklearn.preprocessing import RobustScaler

        if lista_descritores is None:
            from config import DESCRITORES
            lista_descritores = DESCRITORES["lista"]

        # Valida que todos os nomes existem no RDKit
        descritores_disponiveis = dict(Descriptors.descList)
        invalidos = [d for d in lista_descritores if d not in descritores_disponiveis]
        if invalidos:
            raise ValueError(
                f"Descritores não encontrados no RDKit: {invalidos}\n"
                f"Verifique DESCRITORES['lista'] em config.py."
            )

        print(f"  [Descritores] Calculando {len(lista_descritores)} descritores...")

        linhas = []
        for smi in self.dataframe[col_smiles]:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError
                valores = [descritores_disponiveis[d](mol) for d in lista_descritores]
            except Exception:
                # Molécula inválida → preenche com NaN para ser removida depois
                valores = [np.nan] * len(lista_descritores)
            linhas.append(valores)

        matriz = np.array(linhas, dtype=np.float64)

        # Substitui NaN pela mediana da coluna (moléculas inválidas isoladas)
        for j in range(matriz.shape[1]):
            col = matriz[:, j]
            mediana = np.nanmedian(col)
            matriz[np.isnan(col), j] = mediana

        # Ajusta e aplica RobustScaler
        self.scaler = RobustScaler()
        matriz_scaled = self.scaler.fit_transform(matriz).astype(np.float32)

        self.dataframe["Descritores"] = [matriz_scaled[i] for i in range(len(matriz_scaled))]

        print(f"  [Descritores] Coluna 'Descritores' adicionada. Shape por molécula: ({len(lista_descritores)},)")
        return self.dataframe

    def concatenar_descritores(self) -> pd.DataFrame:
        """
        Concatena 'Features' (fingerprint) e 'Descritores' em um novo vetor
        'Features', substituindo o original.

        Resultado: Features.shape = (fpSize + n_descritores,)

        Moléculas sem 'Descritores' válido recebem NaN e são descartadas.
        Deve ser chamado APÓS bitVect_to_array() e calcular_descritores().
        """
        if "Descritores" not in self.dataframe.columns:
            raise RuntimeError(
                "Coluna 'Descritores' não encontrada. "
                "Chame calcular_descritores() antes de concatenar_descritores()."
            )
        if "Features" not in self.dataframe.columns:
            raise RuntimeError(
                "Coluna 'Features' não encontrada. "
                "Chame bitVect_to_array() antes de concatenar_descritores()."
            )

        novos_features = []
        for idx in self.dataframe.index:
            fp  = self.dataframe["Features"].loc[idx]
            desc = self.dataframe["Descritores"].loc[idx]

            try:
                if isinstance(fp, float) or isinstance(desc, float):
                    raise ValueError("NaN detectado")
                combinado = np.concatenate([
                    np.asarray(fp,   dtype=np.float32),
                    np.asarray(desc, dtype=np.float32),
                ])
            except Exception:
                combinado = np.nan

            novos_features.append(combinado)

        self.dataframe["Features"] = novos_features
        self.dataframe.drop(columns=["Descritores"], inplace=True, errors="ignore")
        print(f"  [Descritores] Concatenação concluída. Shape final: ({len(novos_features[0])},)")
        return self.dataframe