"""
agents/data_gatherer.py
───────────────────────
Agent 3 — Access to MIMIC-IV data.

Loads pre-generated checkpoints (slim parquet, prescriptions pickle,
ICD glossary) and provides structured data for any patient.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logging import get_logger, LogTimer

logger = get_logger("data_gatherer")


class DataGatherer:
    """
    Agent 3: Access to MIMIC-IV data from checkpoints.
    Loads once, then serves requests for any hadm_id.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._combined_data: Optional[pd.DataFrame] = None
        self._prescriptions_map: Dict[int, dict] = {}
        self._diagnosis_desc_map: Dict[str, str] = {}
        self._hadm_to_idx: Dict[int, int] = {}
        self._loaded = False

    def load(self):
        """Loads all data from checkpoints. Call only once."""
        if self._loaded:
            return

        with LogTimer("DataGatherer — Data Loading", logger):
            self._load_slim_parquet()
            self._load_prescriptions()
            self._load_diagnosis_glossary()
            self._loaded = True

    def _load_slim_parquet(self):
        """Loads slim parquet with combined data."""
        path = os.path.join(self.settings.checkpoint_path, "combined_data_slim.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Run the pipeline in 03_03_v4.ipynb notebook first."
            )
        self._combined_data = pd.read_parquet(path)
        self._hadm_to_idx = {
            int(h): i for i, h in enumerate(self._combined_data["hadm_id"].values)
        }
        logger.info(
            f"Slim parquet loaded: {len(self._combined_data)} admissions, "
            f"{len(self._combined_data.columns)} columns"
        )

    def _load_prescriptions(self):
        """Loads prescriptions map by date."""
        path = os.path.join(self.settings.checkpoint_path, "prescriptions_by_date.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._prescriptions_map = pickle.load(f)
            logger.info(f"Prescriptions loaded: {len(self._prescriptions_map)} admissions")
        else:
            logger.warning(f"Prescriptions file not found: {path}")

    def _load_diagnosis_glossary(self):
        """Loads ICD code -> description map."""
        path = os.path.join(self.settings.hosp_path, "d_icd_diagnoses.csv.gz")
        if os.path.exists(path):
            d_icd = pd.read_csv(path)
            d_icd["icd_code"] = d_icd["icd_code"].astype(str).str.strip()
            d_icd["long_title"] = d_icd["long_title"].astype(str)
            self._diagnosis_desc_map = dict(zip(d_icd["icd_code"], d_icd["long_title"]))
            logger.info(f"ICD glossary loaded: {len(self._diagnosis_desc_map)} entries")
        else:
            logger.warning(f"ICD glossary not found: {path}")

    # ── Data Access ──

    def patient_exists(self, hadm_id: int) -> bool:
        """Verifies if a patient exists in the dataset."""
        return int(hadm_id) in self._hadm_to_idx

    def get_patient_index(self, hadm_id: int) -> Optional[int]:
        """Returns DataFrame index for a hadm_id."""
        return self._hadm_to_idx.get(int(hadm_id))

    def get_available_hadm_ids(self, sample: int = 10) -> List[int]:
        """Returns a sample of available hadm_ids."""
        ids = list(self._hadm_to_idx.keys())
        return ids[:sample]

    def get_patient_data(
        self,
        hadm_id: int,
        attributes_list: Optional[List[str]] = None,
        is_target: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts structured data for a patient.

        Args:
            hadm_id: Admission ID
            attributes_list: List of attributes to include (None = all)
            is_target: True if target patient (excludes outcome)

        Returns:
            Structured dictionary ready for LLM context
        """
        idx = self._hadm_to_idx.get(int(hadm_id))
        if idx is None:
            logger.warning(f"hadm_id {hadm_id} not found in dataset")
            return None

        row = self._combined_data.iloc[idx]
        rank = 0 if is_target else 1
        return self._build_patient_dict(row, rank, attributes_list)

    def get_patients_data_batch(
        self,
        hadm_ids: List[int],
        attributes_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts data for a list of patients (similar patients)."""
        results = []
        for i, hid in enumerate(hadm_ids, 1):
            data = self.get_patient_data(hid, attributes_list, is_target=False)
            if data:
                data["rank"] = i
                results.append(data)
        return results

    def build_diagnosis_glossary(self, codes: Set[str]) -> Dict[str, str]:
        """Builds description glossary for a set of ICD codes."""
        return {
            str(code).strip(): self._diagnosis_desc_map.get(
                str(code).strip(), "Description not found"
            )
            for code in codes
        }

    @property
    def combined_data(self) -> pd.DataFrame:
        """Direct access to DataFrame (used by SimilarityEngine)."""
        return self._combined_data

    # ── Patient dictionary construction ──

    def _build_patient_dict(
        self,
        row,
        rank: int = 0,
        attributes_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Builds structured dictionary of patient data.
        Replicates _get_patient_structured_data() logic from notebook.
        """
        include_all = attributes_list is None
        attrs = set(attributes_list) if attributes_list else set()

        data: Dict[str, Any] = {
            "rank": rank,
            "hadm_id": _convert(row["hadm_id"]),
        }

        # Demographics
        if include_all or "demographics" in attrs:
            data["demographics"] = {
                "age": _convert(row["age"]),
                "gender": row["gender"],
                "race": _get_str(row.get("race")),
                "marital_status": _get_str(row.get("marital_status")),
                "language": _get_str(row.get("language")),
            }

        # Admission details
        if include_all or "admission_details" in attrs:
            data["admission_details"] = {
                "type": _get_str(row.get("admission_type")),
                "insurance": _get_str(row.get("insurance")),
            }

        # Outcome — only for similar patients (rank > 0)
        if rank > 0 and (include_all or "outcome" in attrs):
            data["outcome"] = {
                "status": "DIED" if row.get("mortality", 0) == 1 else "SURVIVED",
                "los_days": round(_convert(row.get("los_days", 0)), 2),
            }

        # Current diagnoses
        if include_all or "diagnoses" in attrs:
            diags = row.get("diagnoses", [])
            if hasattr(diags, "__iter__") and len(diags) > 0:
                data["current_diagnoses"] = [str(d) for d in diags]

        # Previous diagnoses
        if include_all or "previous_diagnoses" in attrs:
            prev = row.get("previous_diagnoses", [])
            if hasattr(prev, "__iter__") and len(prev) > 0:
                data["previous_diagnoses"] = [str(d) for d in prev]
            else:
                data["previous_diagnoses"] = []

        # Vital signs
        if include_all or "vitals" in attrs:
            vitals = {}
            for col, label in self.settings.vital_map.items():
                if col in row.index and pd.notna(row[col]) and row[col] != 0:
                    vitals[label] = round(_convert(row[col]), 1)
            if vitals:
                data["vitals_avg"] = vitals

        # Laboratory
        if include_all or "labs" in attrs:
            labs = {}
            for col, label in self.settings.lab_map.items():
                if col in row.index and pd.notna(row[col]) and row[col] != 0:
                    labs[label] = round(_convert(row[col]), 1)
            if labs:
                data["labs_avg"] = labs

        # Prescriptions
        if include_all or "prescriptions" in attrs:
            prescriptions = self._prescriptions_map.get(int(row["hadm_id"]), {})
            if prescriptions:
                data["prescriptions"] = prescriptions

        return data


# ── Helpers ──

def _convert(val):
    """Converts numpy types to native Python types."""
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.ndarray, list)):
        return [_convert(x) for x in val]
    return val


def _get_str(val, default: str = "UNKNOWN") -> str:
    """Returns a clean string or default."""
    if val is None or pd.isna(val) or str(val).lower() in ("nan", "none", ""):
        return default
    return str(val)

