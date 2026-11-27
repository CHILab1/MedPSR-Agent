"""
agents/context_constructor.py
─────────────────────────────
Agent 2 — Structured context constructor.

Receives instructions from Orchestrator, coordinates DataGatherer and
SimilarityEngine, and builds the clean JSON for Specialist LLM.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import Settings
from utils.logging import get_logger, LogTimer

logger = get_logger("context_constructor")


class ContextConstructor:
    """
    Agent 2: Assembly of structured context for LLM.
    Coordinates DataGatherer (Agent 3) and SimilarityEngine (Agent 4).
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_context(
        self,
        routing: dict,
        data_gatherer,
        similarity_engine=None,
    ) -> Dict[str, Any]:
        """
        Builds complete context JSON.

        Args:
            routing: Routing JSON from orchestrator
            data_gatherer: Loaded DataGatherer instance
            similarity_engine: Loaded SimilarityEngine instance (optional)

        Returns:
            Structured dictionary ready for LLM prompt
        """
        patient_id = routing["patient_id"]
        attributes = routing.get("attributes_list")
        use_similarity = routing.get("similarity", False)
        k_similar = routing.get("k_similar", self.settings.similarity.k_similar)
        weights = routing.get("weights_similarity", self.settings.similarity.weights)
        exclude_same_subject = routing.get(
            "exclude_same_subject",
            getattr(self.settings.similarity, "exclude_same_subject", True),
        )

        with LogTimer("ContextConstructor — Context Assembly", logger):
            # ── 1. Target patient data ──
            logger.info(f"Retrieving target patient data: {patient_id}")
            target_data = data_gatherer.get_patient_data(
                hadm_id=patient_id,
                attributes_list=attributes,
                is_target=True,
            )
            if target_data is None:
                raise ValueError(f"Patient {patient_id} not found in dataset")

            # ── Base structure ──
            context = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model_version": "AgenticPipeline_v1",
                    "target_hadm_id": patient_id,
                    "routing": {
                        "topics": routing.get("topics", []),
                        "similarity_enabled": use_similarity,
                        "k_similar": k_similar,
                        "weights": weights,
                        "exclude_same_subject": exclude_same_subject,
                    },
                },
                "target_patient": target_data,
            }

            # ── 2. Similar patients (if requested) ──
            if use_similarity and similarity_engine is not None:
                logger.info(f"Searching for {k_similar} similar patients...")
                similar_results = similarity_engine.find_similar(
                    target_hadm_id=patient_id,
                    k=k_similar,
                    weights=weights,
                    exclude_same_subject=exclude_same_subject,
                )

                if similar_results:
                    similar_hadm_ids = [hid for hid, _ in similar_results]
                    scores = {hid: score for hid, score in similar_results}

                    logger.info(
                        f"Retrieving data for {len(similar_hadm_ids)} similar patients..."
                    )
                    similar_patients = data_gatherer.get_patients_data_batch(
                        hadm_ids=similar_hadm_ids,
                        attributes_list=attributes,
                    )

                    # Add similarity score to each patient
                    for pat in similar_patients:
                        hid = pat.get("hadm_id")
                        if hid in scores:
                            pat["similarity_score"] = round(scores[hid], 4)

                    target_subj = target_data.get("subject_id")
                    if target_subj is not None and exclude_same_subject:
                        similar_patients = [
                            p for p in similar_patients if p.get("subject_id") != target_subj
                        ]

                    context["similar_patients"] = similar_patients
                    logger.info(f"[success]{len(similar_patients)} similar patients added[/success]")
                else:
                    context["similar_patients"] = []
                    logger.warning("No similar patients found")
            else:
                logger.info("Similarity not requested — context includes target patient only")

            # ── 3. Diagnosis glossary ──
            context["diagnosis_glossary"] = self._build_glossary(
                context, data_gatherer
            )

        return context

    def _build_glossary(
        self,
        context: Dict[str, Any],
        data_gatherer,
    ) -> Dict[str, str]:
        """Collects all ICD codes and builds the glossary."""
        all_codes = set()

        # Target codes
        target = context.get("target_patient", {})
        self._collect_codes(target, all_codes)

        # Similar patient codes
        for pat in context.get("similar_patients", []):
            self._collect_codes(pat, all_codes)

        return data_gatherer.build_diagnosis_glossary(all_codes)

    @staticmethod
    def _collect_codes(patient_dict: dict, codes: set):
        """Collects ICD codes from a patient dictionary."""
        for key in ("current_diagnoses", "diagnoses", "previous_diagnoses"):
            if key in patient_dict:
                codes.update(patient_dict[key])

