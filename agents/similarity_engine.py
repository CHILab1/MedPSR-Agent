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
from pymilvus import Collection, connections
from sklearn.metrics.pairwise import cosine_similarity

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
        self._collection_diagnosis: Optional[Collection] = None
        self._loaded = False

    def load(self, hadm_to_idx: Dict[int, int]):
        """
        Loads embeddings and connects to Milvus.

        Args:
            hadm_to_idx: Map hadm_id -> index (provided by DataGatherer)
        """
        if self._loaded:
            return

        with LogTimer("SimilarityEngine — Loading", logger):
            self._hadm_to_idx = hadm_to_idx
            self._connect_milvus()
            self._load_embeddings()
            self._load_milvus_collection()
            self._loaded = True

    def _connect_milvus(self):
        """Connects to Milvus."""
        host = self.settings.milvus.host
        port = self.settings.milvus.port
        logger.info(f"Connecting to Milvus ({host}:{port})...")
        connections.connect("default", host=host, port=port)
        logger.info("Connected to Milvus.")

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
        coll_name = self.settings.milvus.collection_diagnosis
        self._collection_diagnosis = Collection(coll_name)
        try:
            self._collection_diagnosis.load()
        except Exception:
            pass  # Already loaded
        logger.info(f"Milvus collection '{coll_name}': {self._collection_diagnosis.num_entities} entities")

    def find_similar(
        self,
        target_hadm_id: int,
        k: int = 10,
        weights: Optional[List[float]] = None,
        pool_size: int = 200,
    ) -> List[Tuple[int, float]]:
        """
        Finds the K most similar patients to target.

        Args:
            target_hadm_id: Target patient hadm_id
            k: Number of similar patients to return
            weights: [w_diagnosis, w_numeric, w_categorical] (sum = 1.0)
            pool_size: Candidate pool size from Milvus

        Returns:
            List of tuples (hadm_id, similarity_score) sorted by score descending
        """
        if weights is None:
            weights = self.settings.similarity.weights

        w_diag, w_num, w_cat = weights[0], weights[1], weights[2]

        target_idx = self._hadm_to_idx.get(int(target_hadm_id))
        if target_idx is None:
            logger.error(f"hadm_id {target_hadm_id} not found in index")
            return []

        with LogTimer(f"Searching for {k} similar patients (pool={pool_size})", logger):
            # ── STAGE 1: ANN search on Milvus (diagnosis) ──
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

            candidate_indices = [hit.id for hit in results[0] if hit.id != target_idx]
            diag_scores = {hit.id: hit.distance for hit in results[0]}

            if not candidate_indices:
                logger.warning("No candidates found in Milvus")
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
            top = scored[:k]

            logger.info(f"STAGE 2: Re-ranking completed, top {len(top)} selected")

            # Convert indices -> hadm_id
            idx_to_hadm = {v: k for k, v in self._hadm_to_idx.items()}
            result = [(idx_to_hadm[idx], score) for idx, score in top if idx in idx_to_hadm]

            return result

    def close(self):
        """Close Milvus connection."""
        try:
            connections.disconnect("default")
            logger.info("Milvus connection closed.")
        except Exception:
            pass

