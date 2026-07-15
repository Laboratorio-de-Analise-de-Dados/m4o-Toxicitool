# -----Base-----#
import pandas as pd
import numpy as np
from pathlib import Path

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
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def mol_to_frame(self, col_smiles: str) -> pd.DataFrame:
        """Adiciona coluna 'ROMol'. SMILES inválidos geram NaN."""
        if col_smiles in self.dataframe.columns:
            self.dataframe[col_smiles] = self.dataframe[col_smiles].fillna("").astype(str)
        
        PandasTools.AddMoleculeColumnToFrame(frame=self.dataframe, smilesCol=col_smiles)
        return self.dataframe

    def fp_Morgan(
        self,
        col_frames: str,
        radius: int = 2,
        fpSize: int = 2048,
        use_count: bool = False,
    ) -> pd.DataFrame:
        morgan_lista = []
        for idx in self.dataframe.index:
            try:
                mol = self.dataframe[col_frames].loc[idx]
                if mol is None or pd.isna(mol):
                    raise ValueError
                gen    = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
                morgan = gen.GetCountFingerprint(mol) if use_count else gen.GetFingerprint(mol)
            except Exception:
                morgan = np.nan
            morgan_lista.append(morgan)

        self.dataframe["Fingerprint"] = morgan_lista
        return self.dataframe

    def bitVect_to_array(self, col_fp: str) -> pd.DataFrame:
        fp_arrays = []
        for idx in self.dataframe.index:
            try:
                fp_obj = self.dataframe[col_fp].loc[idx]
                if fp_obj is np.nan or fp_obj is None or pd.isna(fp_obj):
                    raise ValueError

                if hasattr(fp_obj, "GetNumBits"):
                    n_bits = int(fp_obj.GetNumBits())
                    dtype  = np.int8
                elif hasattr(fp_obj, "GetLength"):
                    n_bits = int(fp_obj.GetLength())
                    dtype  = np.float32
                else:
                    raise ValueError

                fp_arr = np.zeros((n_bits,), dtype=dtype)
                DataStructs.ConvertToNumpyArray(fp_obj, fp_arr)
            except Exception:
                fp_arr = np.nan

            fp_arrays.append(fp_arr)

        self.dataframe["Features"] = fp_arrays
        return self.dataframe

    def calcular_descritores(
        self,
        col_smiles: str,
        lista_descritores: list[str] | str | None = None,
    ) -> pd.DataFrame:
        from sklearn.preprocessing import RobustScaler
        import joblib

        if lista_descritores is None:
            from config import DESCRITORES
            lista_descritores = DESCRITORES["lista"]

        descritores_disponiveis = dict(Descriptors.descList)
        
        # ORDEM ORIGINAL DO RDKIT (Crítica para o Scaler pre-treinado)
        if lista_descritores == "todos":
            nomes_calculo = [d[0] for d in Descriptors.descList]
        else:
            nomes_calculo = lista_descritores

        print(f"  [Descritores] Calculando {len(nomes_calculo)} descritores...")

        linhas = []
        for smi in self.dataframe[col_smiles]:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None: raise ValueError
                
                valores = []
                for d in nomes_calculo:
                    val = descritores_disponiveis[d](mol)
                    # FIX IPC: Impede que o descritor Ipc exploda para infinito
                    if d == "Ipc" and val > 1e10: val = 1e10
                    valores.append(val)
            except Exception:
                valores = [np.nan] * len(nomes_calculo)
            linhas.append(valores)

        matriz = np.array(linhas, dtype=np.float64)

        # Tratar NaNs
        for j in range(matriz.shape[1]):
            col = matriz[:, j]
            mask_nan = np.isnan(col)
            if np.all(mask_nan): matriz[:, j] = 0
            elif np.any(mask_nan): matriz[mask_nan, j] = np.nanmedian(col)

        # Carregar colunas válidas e scaler
        colunas_path = Path("data/preprocessed/colunas_validas_descritores.npy")
        scaler_path = Path("data/preprocessed/scaler_descritores.joblib")
        
        if colunas_path.exists() and scaler_path.exists() and lista_descritores == "todos":
            colunas_validas = np.load(colunas_path)
            matriz = matriz[:, colunas_validas]
            self.scaler = joblib.load(scaler_path)
            matriz_scaled = self.scaler.transform(matriz).astype(np.float32)
        else:
            variancias = np.var(matriz, axis=0)
            colunas_validas = variancias > 1e-6
            matriz = matriz[:, colunas_validas]
            self.scaler = RobustScaler()
            matriz_scaled = self.scaler.fit_transform(matriz).astype(np.float32)

        self.dataframe["Descritores"] = [matriz_scaled[i] for i in range(len(matriz_scaled))]
        return self.dataframe

    def concatenar_descritores(self) -> pd.DataFrame:
        novos_features = []
        for idx in self.dataframe.index:
            fp  = self.dataframe["Features"].loc[idx]
            desc = self.dataframe["Descritores"].loc[idx]

            try:
                if isinstance(fp, float) or isinstance(desc, float) or pd.isna(fp).any() or pd.isna(desc).any():
                    raise ValueError
                combinado = np.concatenate([
                    np.asarray(fp,   dtype=np.float32),
                    np.asarray(desc, dtype=np.float32),
                ])
            except Exception:
                combinado = np.nan

            novos_features.append(combinado)

        self.dataframe["Features"] = novos_features
        self.dataframe.drop(columns=["Descritores"], inplace=True, errors="ignore")
        return self.dataframe
