"""
Splitter.py  (src/modeling/)
==============================
Módulo de splitting consciente de similaridade química para pipelines MTL.

Estratégias disponíveis:
    random_split        — split aleatório (baseline)
    scaffold_split      — split por Murcko Scaffold, O(n)
    butina_split        — Butina clássico, O(n²) memória (datasets pequenos)
    butina_batch_split  — Butina via leader-follower, O(n×k) memória (datasets grandes)

Referências:
    Butina, D. J. Chem. Inf. Comput. Sci. (1999), 39(4), 747.
    Bemis, G. W.; Murcko, M. A. J. Med. Chem. (1996), 39(15), 2887.
"""

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


class Splitter:

    def __init__(self, dataframe: pd.DataFrame, smiles_col: str = "smiles") -> None:
        self.df         = dataframe.reset_index(drop=True)
        self.smiles_col = smiles_col

    # ──────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────

    def random_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split aleatório simples (baseline)."""
        from sklearn.model_selection import train_test_split

        indices  = np.arange(len(self.df))
        stratify = None
        if stratify_col and stratify_col in self.df.columns:
            stratify = (~self.df[stratify_col].isna()).to_numpy(dtype=int)

        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return train_idx, test_idx

    def scaffold_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split por Murcko Scaffold. O(n) — recomendado para datasets grandes.

        Agrupa moléculas pelo mesmo núcleo de Murcko e distribui grupos inteiros
        entre treino e teste, evitando que variantes da mesma série química
        vazem entre os conjuntos.
        """
        print("  [Scaffold] Extraindo Murcko Scaffolds...")
        scaffolds = defaultdict(list)

        for i, smi in enumerate(self.df[self.smiles_col]):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False
                    )
                    scaffolds[scaffold].append(i)
            except Exception:
                pass

        print(f"  [Scaffold] {len(scaffolds)} scaffolds únicos encontrados.")

        rng           = np.random.default_rng(random_state)
        scaffold_sets = list(scaffolds.values())
        rng.shuffle(scaffold_sets)
        scaffold_sets.sort(key=lambda x: len(x), reverse=True)

        n_total, n_teste = len(self.df), int(np.ceil(len(self.df) * test_size))
        test_indices, train_indices = [], []

        for scaf_list in scaffold_sets:
            if len(test_indices) < n_teste:
                test_indices.extend(scaf_list)
            else:
                train_indices.extend(scaf_list)

        sobrando = set(range(n_total)) - set(test_indices) - set(train_indices)
        train_indices.extend(sobrando)

        print(f"  [Scaffold] Treino={len(train_indices)} | Teste={len(test_indices)}")
        return np.array(train_indices), np.array(test_indices)

    def butina_split(
        self,
        test_size: float = 0.2,
        tanimoto_cutoff: float = 0.4,
        radius: int = 2,
        fpSize: int = 2048,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Butina clássico. Constrói a matriz triangular completa de distâncias.

        Complexidade: O(n²) memória — use apenas para n ≲ 8 000.
        Para datasets maiores, prefira butina_batch_split.

        Referência: Butina (1999) J. Chem. Inf. Comput. Sci. 39(4), 747.
        """
        print("  [Butina] Gerando fingerprints para clustering...")
        fps = self._gerar_fingerprints(radius=radius, fpSize=fpSize)

        if not fps:
            raise ValueError("Não foi possível gerar fingerprints. Verifique os SMILES.")

        print(f"  [Butina] Calculando matriz de distâncias ({len(fps)} moléculas)...")
        dist_matrix = self._matriz_distancias_tanimoto(fps)

        print(f"  [Butina] Clusterizando (cutoff={tanimoto_cutoff})...")
        clusters = Butina.ClusterData(
            dist_matrix, nPts=len(fps), distThresh=tanimoto_cutoff, isDistData=True
        )
        print(f"  [Butina] {len(clusters)} clusters gerados")

        train_idx, test_idx = self._distribuir_clusters(clusters, test_size, random_state)

        sim_media = self._similaridade_media_treino_teste(fps, train_idx, test_idx)
        print(f"  [Butina] Treino={len(train_idx)} | Teste={len(test_idx)}")
        print(f"  [Butina] Sim. Tanimoto média (teste→treino): {sim_media:.3f}")
        return train_idx, test_idx

    def butina_batch_split(
        self,
        test_size: float = 0.2,
        tanimoto_cutoff: float = 0.4,
        radius: int = 2,
        fpSize: int = 2048,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aproximação escalável do Butina por leader-follower clustering.

        test_size : fração para teste
        tanimoto_cutoff : limiar de distância (cutoff=0.4 → sim ≥ 0.6)
        radius : raio Morgan dos FPs de clustering
        fpSize : tamanho do vetor de FP
        random_state : semente para embaralhamento antes do clustering  
        """
        print("  [Butina-batch] Gerando fingerprints...")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

        valid_fps  = []   # lista de fingerprints RDKit
        valid_idxs = []   # índices originais do DataFrame

        for i, smi in enumerate(self.df[self.smiles_col]):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError
                valid_fps.append(gen.GetFingerprint(mol))
                valid_idxs.append(i)
            except Exception:
                pass   # SMILES inválido → alocado no treino pelo fallback

        n_valid   = len(valid_fps)
        n_invalid = len(self.df) - n_valid
        if n_invalid:
            print(f"  [Butina-batch] {n_invalid} SMILES inválidos → alocados no treino")
        print(f"  [Butina-batch] {n_valid} fingerprints válidos")

        if n_valid == 0:
            raise ValueError("Nenhum fingerprint válido gerado. Verifique os SMILES.")

        # ── Embaralha para eliminar viés da ordem original ─────────────
        rng  = np.random.default_rng(random_state)
        perm = rng.permutation(n_valid)
        fps_ord  = [valid_fps[i]  for i in perm]
        idxs_ord = [valid_idxs[i] for i in perm]

        # ── Leader-follower ────────────────────────────────────────────
        print(f"  [Butina-batch] Clusterizando {n_valid} moléculas (cutoff={tanimoto_cutoff})...")

        leaders  = [fps_ord[0]]          # fingerprints dos líderes atuais
        clusters = [[idxs_ord[0]]]       # clusters = listas de índices originais do df

        for pos in range(1, n_valid):
            fp       = fps_ord[pos]
            orig_idx = idxs_ord[pos]

            # BulkTanimotoSimilarity é C-otimizado no RDKit — rápido mesmo com k grande
            sims     = DataStructs.BulkTanimotoSimilarity(fp, leaders)
            best_sim = max(sims)

            if best_sim >= (1.0 - tanimoto_cutoff):
                clusters[int(np.argmax(sims))].append(orig_idx)
            else:
                leaders.append(fp)
                clusters.append([orig_idx])

            if (pos + 1) % 5_000 == 0:
                pct = 100 * (pos + 1) / n_valid
                print(f"    {pos+1:>7}/{n_valid} ({pct:5.1f}%) | {len(clusters)} clusters")

        print(f"  [Butina-batch] {len(clusters)} clusters gerados")

        # ── Distribui clusters entre treino e teste ────────────────────
        train_idx, test_idx = self._distribuir_clusters(
            [tuple(c) for c in clusters], test_size, random_state
        )

        # ── Relatório de similaridade ──────────────────────────────────
        fp_map     = dict(zip(valid_idxs, valid_fps))
        n_amostras = min(500, len(test_idx))
        amostras   = rng.choice(test_idx, size=n_amostras, replace=False)
        fps_train  = [fp_map[i] for i in train_idx if i in fp_map]

        if fps_train:
            sim_vals = [
                max(DataStructs.BulkTanimotoSimilarity(fp_map[i], fps_train))
                for i in amostras
                if i in fp_map
            ]
            sim_media = float(np.mean(sim_vals)) if sim_vals else 0.0
            print(f"  [Butina-batch] Treino={len(train_idx)} | Teste={len(test_idx)}")
            print(f"  [Butina-batch] Sim. Tanimoto média (teste→treino): {sim_media:.3f}")
            print(f"                 (quanto menor, mais honesto é o split)")

        return train_idx, test_idx

    # ──────────────────────────────────────────────────────────────────
    # MÉTODOS INTERNOS
    # ──────────────────────────────────────────────────────────────────

    def _gerar_fingerprints(self, radius: int, fpSize: int) -> list:
        """
        Gera fingerprints Morgan para todas as moléculas válidas.
        Retorna lista de fps válidos (sem os None de SMILES inválidos).

        Nota: os índices desta lista NÃO correspondem aos índices originais
        do DataFrame. Use butina_batch_split para manter o mapeamento correto.
        """
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
        fps_validos = []
        n_invalidos = 0

        for smi in self.df[self.smiles_col]:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError
                fps_validos.append(gen.GetFingerprint(mol))
            except Exception:
                n_invalidos += 1

        if n_invalidos:
            print(f"  [Butina] {n_invalidos} SMILES inválidos ignorados no clustering")

        return fps_validos

    def _matriz_distancias_tanimoto(self, fps: list) -> list:
        """
        Matriz triangular inferior de distâncias Tanimoto.
        Formato Butina: lista 1D, distância[i,j] = 1 - sim[i,j] para i > j.
        """
        dist_matrix = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dist_matrix.extend(1 - s for s in sims)
        return dist_matrix

    def _distribuir_clusters(
        self,
        clusters: list,
        test_size: float,
        random_state: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distribui clusters inteiros para treino/teste respeitando test_size.
        Clusters maiores têm prioridade no teste para evitar dominância de singletons.
        Índices não cobertos pelos clusters (SMILES inválidos) vão para treino.
        """
        rng = np.random.default_rng(random_state)

        clusters_ord = sorted(clusters, key=len, reverse=True)
        rng.shuffle(clusters_ord)

        n_total = len(self.df)
        n_teste = int(np.ceil(n_total * test_size))

        test_indices, train_indices = [], []

        for cluster in clusters_ord:
            cluster_list = list(cluster)
            if len(test_indices) < n_teste:
                test_indices.extend(cluster_list)
            else:
                train_indices.extend(cluster_list)

        # Garante cobertura total (ex: SMILES inválidos sem cluster)
        cobertos = set(test_indices) | set(train_indices)
        sobrando = set(range(n_total)) - cobertos
        train_indices.extend(sobrando)

        return np.array(train_indices), np.array(test_indices)

    def _similaridade_media_treino_teste(
        self,
        fps: list,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_amostras: int = 500,
    ) -> float:
        """
        Similaridade Tanimoto média entre moléculas de teste e seu vizinho
        mais próximo no treino. Métrica de 'honestidade' do split.

        Atenção: fps deve estar indexado da mesma forma que train_idx/test_idx.
        Para butina_split (clássico), os índices são relativos à lista fps_validos.
        """
        n        = min(n_amostras, len(test_idx))
        amostras = np.random.choice(test_idx, size=n, replace=False)
        fps_train = [fps[i] for i in train_idx if i < len(fps)]

        sims = []
        for i in amostras:
            if i < len(fps):
                sim_viz = DataStructs.BulkTanimotoSimilarity(fps[i], fps_train)
                sims.append(max(sim_viz) if sim_viz else 0.0)

        return float(np.mean(sims)) if sims else 0.0