"""
config/settings.py
──────────────────
Centralized configuration for the agentic system.
Reads from environment variables / .env file with fallbacks to default values.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Default path to local data directory (data/ inside project)
_DEFAULT_DATA_PATH = os.getenv(
    "MIMIC_DATA_PATH",
    str(_PROJECT_ROOT / "data")
)


@dataclass
class MilvusConfig:
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: str = os.getenv("MILVUS_PORT", "19530")
    collection_numeric: str = "mimiciv_numeric_embeddings"
    collection_categorical: str = "mimiciv_categorical_embeddings"
    collection_diagnosis: str = "mimiciv_diagnosis_embeddings"


@dataclass
class LLMConfig:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    orchestrator_model: str = os.getenv("ORCHESTRATOR_MODEL", "gemma4:e2b")
    specialist_model: str = os.getenv("SPECIALIST_MODEL", "gemini-3.1-flash-lite-preview")
    synthesizer_model: str = os.getenv("SYNTHESIZER_MODEL", "gemma4:31b-cloud")
    api_version: str = os.getenv("GEMINI_API_VERSION", "v1beta")
    # Specialist generation parameters
    specialist_temperature: float = 0.6
    specialist_top_p: float = 0.90
    specialist_top_k: int = 40

@dataclass
class SimilarityDefaults:
    k_similar: int = 10
    pool_size: int = 200
    # Weights: [diagnosis, numeric, categorical]
    weights: List[float] = field(default_factory=lambda: [0.40, 0.40, 0.20])


@dataclass
class Settings:
    """Global system configuration."""

    # ── Paths ──
    mimic_data_path: str = _DEFAULT_DATA_PATH
    hosp_path: str = ""
    icu_path: str = ""
    checkpoint_path: str = ""

    # ── Sub-configs ──
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    similarity: SimilarityDefaults = field(default_factory=SimilarityDefaults)

    # ── Available attribute map for context ──
    available_attributes: List[str] = field(default_factory=lambda: [
        "demographics",
        "admission_details",
        "vitals",
        "labs",
        "prescriptions",
        "diagnoses",
        "previous_diagnoses",
        "outcome",
    ])

    # ── Available analysis topics ──
    available_topics: List[str] = field(default_factory=lambda: [
        "pattern_analysis",
        "consulto_farmacologico",
    ])

    # ── Vital map: parquet_column -> readable label ──
    vital_map: Dict[str, str] = field(default_factory=lambda: {
        "heart_rate_mean": "heart_rate",
        "mbp_mean": "mean_bp",
        "resp_rate_mean": "resp_rate",
        "spo2_mean": "spo2",
        "temperature_mean": "temperature",
    })

    # ── Lab map: parquet_column -> readable label ──
    lab_map: Dict[str, str] = field(default_factory=lambda: {
        "creatinine_mean": "Creatinine", "bun_mean": "BUN",
        "glucose_mean": "Glucose",       "sodium_mean": "Sodium",
        "potassium_mean": "Potassium",   "calcium_mean": "Calcium",
        "magnesium_mean": "Magnesium",   "anion_gap_mean": "Anion Gap",
        "wbc_mean": "WBC",               "hemoglobin_mean": "Hgb",
        "platelets_mean": "Platelets",   "hematocrit_mean": "Hct",
        "crp_mean": "CRP",               "lactate_mean": "Lactate",
        "alt_mean": "ALT",               "ast_mean": "AST",
        "bilirubin_total_mean": "Total Bili", "albumin_mean": "Albumin",
        "alkaline_phosphatase_mean": "ALP",
        "inr_mean": "INR",               "ptt_mean": "PTT",
        "ph_mean": "pH",                 "pco2_mean": "pCO2",
        "po2_mean": "pO2",               "base_excess_mean": "Base Excess",
        "troponin_t_mean": "Troponin T", "ck_mb_mean": "CK-MB",
    })

    def __post_init__(self):
        base = Path(self.mimic_data_path)
        if not self.hosp_path:
            self.hosp_path = str(base / "hosp")
        if not self.icu_path:
            self.icu_path = str(base / "icu")
        if not self.checkpoint_path:
            self.checkpoint_path = str(base / "pipeline_checkpoints")


# ── Singleton ──

_settings_instance: Settings | None = None


def get_settings(**overrides) -> Settings:
    """Returns singleton instance of Settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings(**overrides)
    return _settings_instance

