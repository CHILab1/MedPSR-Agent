"""
agents/similarity_engine.py
────────────────────────────
Agent 4 — Patient similarity engine.

Uses pre-computed embeddings + Milvus for ANN search,
then performs re-ranking with weighted multi-embedding cosine similarity.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from pymilvus import Collection, connections
    HAS_MILVUS = True
except ImportError:
    Collection = None
    connections = None
    HAS_MILVUS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    def cosine_similarity(X, Y=None):
        """NumPy fallback for cosine similarity."""
        X_arr = np.asarray(X, dtype=np.float32)
        Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(1, -1)
        norm_x = np.linalg.norm(X_arr, axis=1, keepdims=True)
        norm_y = np.linalg.norm(Y_arr, axis=1, keepdims=True)
        norm_x[norm_x == 0] = 1e-9
        norm_y[norm_y == 0] = 1e-9
        return np.dot(X_arr, Y_arr.T) / (norm_x * norm_y.T)

from config.settings import Settings
from utils.logging import get_logger, LogTimer

logger = get_logger("similarity_engine")


class SimilarityEngine:
    """
    Agent 4: Similarity engine based on Milvus + re-ranking.
    Two-stage pipeline:
      1. ANN search on Milvus (diagnosis embeddings) -> candidate pool
      2. Re-ranking with weighted cosine similarity (numeric + categorical)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings_numeric: Optional[np.ndarray] = None
        self._embeddings_categorical: Optional[np.ndarray] = None
        self._embeddings_diagnosis: Optional[np.ndarray] = None
        self._hadm_to_idx: Dict[int, int] = {}
        self._idx_to_hadm: Dict[int, int] = {}
        self._hadm_to_subject: Dict[int, int] = {}
        self._idx_to_subject: Dict[int, int] = {}
        self._collection_diagnosis: Optional[Collection] = None
        self._loaded = False
        self.use_milvus = True

    def load(
        self,
        hadm_to_idx: Dict[int, int],
        hadm_to_subject: Optional[Dict[int, int]] = None,
    ):
        """
        Loads embeddings, connects to Milvus, and initializes subject mapping.

        Args:
            hadm_to_idx: Map hadm_id -> index (provided by DataGatherer)
            hadm_to_subject: Optional map hadm_id -> subject_id (for subject filtering)
        """
        if self._loaded:
            return

        with LogTimer("SimilarityEngine — Loading", logger):
            self._hadm_to_idx = hadm_to_idx
            self._idx_to_hadm = {v: k for k, v in hadm_to_idx.items()}

            if hadm_to_subject is not None:
                self._hadm_to_subject = {int(k): int(v) for k, v in hadm_to_subject.items()}
            else:
                self._load_subject_mapping()

            self._idx_to_subject = {
                idx: self._hadm_to_subject[hid]
                for hid, idx in self._hadm_to_idx.items()
                if hid in self._hadm_to_subject
            }

            self._connect_milvus()
            self._load_embeddings()
            self._load_milvus_collection()
            self._loaded = True

    def _load_subject_mapping(self):
        """Loads hadm_id -> subject_id mapping from slim parquet if available."""
        cp = self.settings.checkpoint_path
        if not cp:
            self._hadm_to_subject = {}
            return

        slim_path = os.path.join(cp, "combined_data_slim.parquet")
        if os.path.exists(slim_path):
            try:
                import pandas as pd
                df_subj = pd.read_parquet(slim_path, columns=["hadm_id", "subject_id"])
                self._hadm_to_subject = dict(
                    zip(df_subj["hadm_id"].astype(int), df_subj["subject_id"].astype(int))
                )
            except Exception:
                self._hadm_to_subject = {}
        else:
            self._hadm_to_subject = {}

    def _connect_milvus(self):
        """Connects to Milvus."""
        if not HAS_MILVUS:
            logger.info("pymilvus not installed; local fallback will be used.")
            self.use_milvus = False
            return

        host = self.settings.milvus.host
        port = self.settings.milvus.port
        logger.info(f"Connecting to Milvus ({host}:{port})...")
        try:
            connections.connect("default", host=host, port=port)
            self.use_milvus = True
            logger.info("Connected to Milvus.")
        except Exception as e:
            logger.warning(
                f"Connecting to Milvus failed: {e}. Fallback to local search will be used if necessary."
            )
            self.use_milvus = False

    def _load_embeddings(self):
        """Loads three types of embeddings from .npy files."""
        cp = self.settings.checkpoint_path
        paths = {
            "numeric": os.path.join(cp, "patient_embeddings_numeric.npy"),
            "categorical": os.path.join(cp, "patient_embeddings_categorical.npy"),
            "diagnosis": os.path.join(cp, "patient_embeddings_diagnosis.npy"),
        }

        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Embedding file not found: {path}")

        self._embeddings_numeric = np.load(paths["numeric"])
        self._embeddings_categorical = np.load(paths["categorical"])
        self._embeddings_diagnosis = np.load(paths["diagnosis"])

        logger.info(
            f"Embeddings loaded — numeric: {self._embeddings_numeric.shape}, "
            f"categorical: {self._embeddings_categorical.shape}, "
            f"diagnosis: {self._embeddings_diagnosis.shape}"
        )

    def _load_milvus_collection(self):
        """Loads Milvus diagnosis collection."""
        if not self.use_milvus or not HAS_MILVUS:
            return

        coll_name = self.settings.milvus.collection_diagnosis
        try:
            self._collection_diagnosis = Collection(coll_name)
            self._collection_diagnosis.load()
            logger.info(f"Milvus collection '{coll_name}': {self._collection_diagnosis.num_entities} entities")
        except Exception as e:
            logger.warning(f"Failed to load Milvus collection '{coll_name}': {e}. Fallback to local search.")
            self.use_milvus = False

    def find_similar(
        self,
        target_hadm_id: int,
        k: int = 10,
        weights: Optional[List[float]] = None,
        pool_size: int = 200,
        exclude_same_subject: Optional[bool] = None,
    ) -> List[Tuple[int, float]]:
        """
        Finds the K most similar patients to target.

        Args:
            target_hadm_id: Target patient hadm_id
            k: Number of similar patients to return
            weights: [w_diagnosis, w_numeric, w_categorical] (sum = 1.0)
            pool_size: Candidate pool size from Milvus
            exclude_same_subject: Exclude admissions with same subject_id as target

        Returns:
            List of tuples (hadm_id, similarity_score) sorted by score descending
        """
        if weights is None:
            weights = self.settings.similarity.weights

        if exclude_same_subject is None:
            exclude_same_subject = getattr(
                self.settings.similarity, "exclude_same_subject", True
            )

        w_diag, w_num, w_cat = weights[0], weights[1], weights[2]

        target_hadm_int = int(target_hadm_id)
        target_idx = self._hadm_to_idx.get(target_hadm_int)
        if target_idx is None:
            logger.error(f"hadm_id {target_hadm_id} not found in index")
            return []

        target_subject_id = self._hadm_to_subject.get(target_hadm_int)

        with LogTimer(f"Searching for {k} similar patients (pool={pool_size})", logger):
            candidate_indices = []
            diag_scores = {}

            # ── STAGE 1: ANN search on Milvus (diagnosis) ──
            if self.use_milvus and self._collection_diagnosis is not None:
                try:
                    diag_embedding = (
                        self._embeddings_diagnosis[target_idx].astype(np.float32).tolist()
                    )
                    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
                    results = self._collection_diagnosis.search(
                        data=[diag_embedding],
                        anns_field="embedding",
                        param=search_params,
                        limit=pool_size,
                        output_fields=["id"],
                    )

                    for hit in results[0]:
                        if hit.id == target_idx:
                            continue
                        if exclude_same_subject and target_subject_id is not None:
                            cand_subject = self._idx_to_subject.get(hit.id)
                            if cand_subject is not None and cand_subject == target_subject_id:
                                continue
                        candidate_indices.append(hit.id)
                        diag_scores[hit.id] = hit.distance
                except Exception as e:
                    logger.warning(f"Milvus search error: {e}. Falling back to local search.")
                    candidate_indices = []

            # Local fallback (if Milvus is inactive or candidate search yielded nothing)
            if not candidate_indices and self._embeddings_diagnosis is not None:
                target_diag = self._embeddings_diagnosis[target_idx].reshape(1, -1)
                sims = cosine_similarity(target_diag, self._embeddings_diagnosis)[0]
                cand_sorted = np.argsort(sims)[::-1]

                for idx in cand_sorted:
                    if idx == target_idx:
                        continue
                    if exclude_same_subject and target_subject_id is not None:
                        cand_subject = self._idx_to_subject.get(idx)
                        if cand_subject is not None and cand_subject == target_subject_id:
                            continue
                    candidate_indices.append(idx)
                    diag_scores[idx] = float(sims[idx])
                    if len(candidate_indices) >= pool_size:
                        break

            if not candidate_indices:
                logger.warning("No candidates found")
                return []

            logger.info(f"STAGE 1: {len(candidate_indices)} candidates from Milvus")

            # ── STAGE 2: Re-ranking with weighted cosine similarity ──
            target_num = self._embeddings_numeric[target_idx].reshape(1, -1)
            target_cat = self._embeddings_categorical[target_idx].reshape(1, -1)
            candidates_num = self._embeddings_numeric[candidate_indices]
            candidates_cat = self._embeddings_categorical[candidate_indices]

            sim_num = cosine_similarity(target_num, candidates_num)[0]
            sim_cat = cosine_similarity(target_cat, candidates_cat)[0]

            scored = []
            for i, idx in enumerate(candidate_indices):
                total = (
                    diag_scores[idx] * w_diag
                    + sim_num[i] * w_num
                    + sim_cat[i] * w_cat
                )
                scored.append((idx, total))

            scored.sort(key=lambda x: x[1], reverse=True)

            top = []
            seen_subjects = set()
            if target_subject_id is not None:
                seen_subjects.add(target_subject_id)

            idx_to_hadm = self._idx_to_hadm or {v: k for k, v in self._hadm_to_idx.items()}

            for idx, score in scored:
                if exclude_same_subject:
                    cand_subj = self._idx_to_subject.get(idx)
                    if cand_subj is not None:
                        if cand_subj in seen_subjects:
                            continue
                        seen_subjects.add(cand_subj)
                top.append((idx, score))
                if len(top) == k:
                    break

            logger.info(f"STAGE 2: Re-ranking completed, top {len(top)} selected")

            result = [(idx_to_hadm[idx], score) for idx, score in top if idx in idx_to_hadm]
            return result

    def close(self):
        """Close Milvus connection."""
        if not HAS_MILVUS or not self.use_milvus:
            return
        try:
            connections.disconnect("default")
            logger.info("Milvus connection closed.")
        except Exception:
            pass
