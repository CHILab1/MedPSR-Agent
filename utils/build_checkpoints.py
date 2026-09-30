"""
build_checkpoints.py — MIMIC-IV Data Preprocessing & Checkpoint Generation
══════════════════════════════════════════════════════════════════════════════
Extracted from 03_03_v4.ipynb notebook for pipeline reproducibility.

Executes full preprocessing of raw MIMIC-IV v3.1 data:
  1. Extraction and trend calculation of critical lab parameters (labevents)
  2. Extraction and trend calculation of vital signs (chartevents / ICU)
  3. Demographic preprocessing, diagnosis aggregation, and longitudinal history
  4. Clinical categorization and date indexing of prescriptions
  5. Outcome metrics calculation (LOS, mortality)
  6. Numerical Golden Features generation (52 standardized continuous features)
  7. Categorical embeddings generation (Demographic OHE)
  8. FastText training and calculation of IDF-weighted diagnosis embeddings (alpha=2.0)
  9. Export of slim parquet checkpoint, numpy arrays, and prescriptions dictionary
  10. (Optional) Vector collection indexing in Milvus

Generated outputs in checkpoint_path:
  - combined_data_slim.parquet
  - patient_embeddings_numeric.npy
  - patient_embeddings_categorical.npy
  - patient_embeddings_diagnosis.npy
  - prescriptions_by_date.pkl
  - scaler.joblib

Usage:
  python build_checkpoints.py --mimic-path ./data/hosp --icu-path ./data/icu
  python build_checkpoints.py --skip-milvus
"""

import argparse
import gzip
import io
import math
import os
import pickle
import sys
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText, KeyedVectors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


class MIMICIVCheckpointBuilder:
    """
    Checkpoint and vector representation builder for the agentic pipeline.
    Executes all data processing, feature engineering, and embedding steps.
    """

    def __init__(
        self,
        mimiciv_path: str,
        icu_path: str,
        categories_path: Optional[str] = None,
        checkpoint_path: str = "./data/pipeline_checkpoints",
        models_path: str = "./data/pipeline_models",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        skip_milvus: bool = False,
        use_checkpoints: bool = True,
        diagnosis_embedding_dim: int = 100,
        power_factor: float = 2.0,
        embedding_weights: Optional[Dict[str, float]] = None,
    ):
        self.mimiciv_path = mimiciv_path
        self.icu_path = icu_path
        self.categories_path = categories_path
        self.checkpoint_path = checkpoint_path
        self.models_path = models_path
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.skip_milvus = skip_milvus
        self.use_checkpoints = use_checkpoints
        self.diagnosis_embedding_dim = diagnosis_embedding_dim
        self.power_factor = power_factor

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        # Modelli e mappe
        self.ft_model_path = os.path.join(self.models_path, "ft_4.wv")
        self.idf_map_path = os.path.join(self.models_path, "idf_map_4.pkl")
        self.diagnosis_ft_model = None
        self.diagnosis_idf_map = None

        # Checkpoint files
        self.cp_lab_features = os.path.join(self.checkpoint_path, "lab_features.parquet")
        self.cp_vitals_features = os.path.join(self.checkpoint_path, "vitals_features.parquet")
        self.cp_combined_data = os.path.join(self.checkpoint_path, "combined_data_final.parquet")
        self.cp_combined_data_slim = os.path.join(self.checkpoint_path, "combined_data_slim.parquet")
        self.cp_embeddings_numeric = os.path.join(self.checkpoint_path, "patient_embeddings_numeric.npy")
        self.cp_embeddings_categorical = os.path.join(self.checkpoint_path, "patient_embeddings_categorical.npy")
        self.cp_embeddings_diagnosis = os.path.join(self.checkpoint_path, "patient_embeddings_diagnosis.npy")
        self.cp_scaler = os.path.join(self.checkpoint_path, "scaler.joblib")
        self.cp_prescriptions_by_date = os.path.join(self.checkpoint_path, "prescriptions_by_date.pkl")

        self.collection_names = {
            "numeric": "mimiciv_numeric_embeddings",
            "categorical": "mimiciv_categorical_embeddings",
            "diagnosis": "mimiciv_diagnosis_embeddings",
        }

        self.embedding_weights = embedding_weights or {
            "numeric": 4.0,
            "categorical": 2.0,
            "diagnosis": 4.0,
        }

        # Data holders
        self.admissions: Optional[pd.DataFrame] = None
        self.patients: Optional[pd.DataFrame] = None
        self.diagnoses_icd: Optional[pd.DataFrame] = None
        self.prescriptions: Optional[pd.DataFrame] = None
        self.lab_features: Optional[pd.DataFrame] = None
        self.vitals_features: Optional[pd.DataFrame] = None
        self.combined_data: Optional[pd.DataFrame] = None

        self.patient_embeddings_numeric: Optional[np.ndarray] = None
        self.patient_embeddings_categorical: Optional[np.ndarray] = None
        self.patient_embeddings_diagnosis: Optional[np.ndarray] = None
        self._prescriptions_by_date_map: Dict[int, dict] = {}
        self.scaler = StandardScaler()

    # ══════════════════════════════════════════════════════════════════════════
    #  ESECUZIONE PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        """Executes full checkpoint generation workflow."""
        print("\n" + "=" * 70)
        print("  STARTING PIPELINE CHECKPOINT GENERATION (from 03_03_v4.ipynb)")
        print("=" * 70)

        # PHASE 1: Combined data
        if self.use_checkpoints and os.path.exists(self.cp_combined_data):
            print(f"[*] Checkpoint 'combined_data_final' found. Loading...")
            self.combined_data = pd.read_parquet(self.cp_combined_data)
            print(f"[*] Loaded {len(self.combined_data)} records.")
        else:
            self.load_mimiciv_data()
            self.preprocess_data()

        # PHASE 2: Prescriptions grouped by date
        if self.use_checkpoints and os.path.exists(self.cp_prescriptions_by_date):
            print(f"[*] Checkpoint 'prescriptions_by_date' found. Loading...")
            with open(self.cp_prescriptions_by_date, "rb") as f:
                self._prescriptions_by_date_map = pickle.load(f)
            print(f"[*] Prescriptions map loaded: {len(self._prescriptions_by_date_map)} admissions.")
        else:
            self._process_prescriptions_by_date()

        # PHASE 3: Embedding Creation
        embeddings_exist = (
            self.use_checkpoints
            and os.path.exists(self.cp_embeddings_numeric)
            and os.path.exists(self.cp_embeddings_categorical)
            and os.path.exists(self.cp_embeddings_diagnosis)
        )

        if embeddings_exist:
            print("[*] Separate embedding checkpoints found. Loading...")
            self.patient_embeddings_numeric = np.load(self.cp_embeddings_numeric)
            self.patient_embeddings_categorical = np.load(self.cp_embeddings_categorical)
            self.patient_embeddings_diagnosis = np.load(self.cp_embeddings_diagnosis)
            if os.path.exists(self.cp_scaler):
                self.scaler = joblib.load(self.cp_scaler)
            self._load_diagnosis_embedding_models()
        else:
            self.create_patient_embeddings()

        # PHASE 4: Saving Slim Parquet for agentic pipeline
        self.save_slim_checkpoint()

        # PHASE 5: Milvus Indexing (if enabled)
        if not self.skip_milvus:
            print("\n[*] Connecting and inserting embeddings into Milvus...")
            self._save_embeddings_to_milvus_multiple()
        else:
            print("\n[*] Milvus indexing skipped (--skip-milvus).")

        print("\n" + "=" * 70)
        print("  CHECKPOINT GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    #  DATA LOADING
    # ══════════════════════════════════════════════════════════════════════════

    def load_mimiciv_data(self):
        """Carica i dati principali MIMIC-IV da CSV compressi."""
        print("\n--- 1. Loading MIMIC-IV tables ---")
        try:
            print(f"Reading patients: {self.mimiciv_path}/patients.csv.gz")
            self.patients = pd.read_csv(f"{self.mimiciv_path}/patients.csv.gz")

            print(f"Reading admissions: {self.mimiciv_path}/admissions.csv.gz")
            self.admissions = pd.read_csv(f"{self.mimiciv_path}/admissions.csv.gz")

            # Diagnoses with ICD category mapping
            cat_path = self.categories_path
            if not cat_path or not os.path.exists(cat_path):
                # Standard fallback
                possible_paths = [
                    os.path.join(self.mimiciv_path, "new_categories_mapping", "diagnoses_4.csv"),
                    os.path.join(self.mimiciv_path, "diagnoses_4.csv"),
                    "./data/new_categories_mapping/diagnoses_4.csv",
                    f"{self.mimiciv_path}/diagnoses_icd.csv.gz",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        cat_path = p
                        break

            if cat_path and os.path.exists(cat_path):
                print(f"Reading mapped diagnoses: {cat_path}")
                self.diagnoses_icd = pd.read_csv(cat_path)
            else:
                raise FileNotFoundError(
                    f"File categorie diagnosi non trovato. Fornisci --categories-path (es. diagnoses_4.csv)"
                )

            print(f"Reading prescriptions: {self.mimiciv_path}/prescriptions.csv.gz")
            self.prescriptions = pd.read_csv(f"{self.mimiciv_path}/prescriptions.csv.gz")

            # Lab features (from checkpoint or calculation)
            if self.use_checkpoints and os.path.exists(self.cp_lab_features):
                print(f"Loading lab_features from {self.cp_lab_features}...")
                self.lab_features = pd.read_parquet(self.cp_lab_features)
            else:
                self._process_labevents_advanced()

            # Vitals features (from checkpoint or calculation)
            if self.use_checkpoints and os.path.exists(self.cp_vitals_features):
                print(f"Loading vitals_features from {self.cp_vitals_features}...")
                self.vitals_features = pd.read_parquet(self.cp_vitals_features)
            else:
                self._process_vitals_events_advanced()

        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            raise

    # ══════════════════════════════════════════════════════════════════════════
    #  LABORATORY & VITALS PROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    def _get_row_count(self, filepath: str) -> int:
        """Conta le righe per la progress bar."""
        try:
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    return sum(1 for _ in f) - 1
            with open(filepath, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1
        except Exception:
            return 100_000_000

    def _calculate_lab_trends(self, lab_data: pd.DataFrame, lab_name: str) -> pd.DataFrame:
        trend = lab_data.copy().sort_values(["hadm_id", "charttime"])
        trend["time_diff"] = trend.groupby("hadm_id")["charttime"].diff().dt.total_seconds() / 3600
        trend["value_diff"] = trend.groupby("hadm_id")["valuenum"].diff()
        trend["time_diff"] = trend["time_diff"].replace(0, np.nan)
        trend["slope"] = trend["value_diff"] / trend["time_diff"]

        trend_stats = trend.groupby("hadm_id").agg(
            {"slope": ["mean", "max", "min"], "value_diff": ["mean", "sum"]}
        ).reset_index()
        trend_stats.columns = [
            "hadm_id",
            f"{lab_name}_slope_mean",
            f"{lab_name}_slope_max",
            f"{lab_name}_slope_min",
            f"{lab_name}_change_mean",
            f"{lab_name}_change_total",
        ]
        trend_stats[f"{lab_name}_trend_worsening"] = (trend_stats[f"{lab_name}_slope_mean"] > 0.1).astype(int)
        trend_stats[f"{lab_name}_trend_improving"] = (trend_stats[f"{lab_name}_slope_mean"] < -0.1).astype(int)
        return trend_stats

    def _process_labevents_advanced(self):
        print("\nProcessing labevents in chunks...")
        critical_labs = {
            "creatinine": [50912], "wbc": [51301, 51300], "hemoglobin": [51222],
            "platelets": [51265], "sodium": [50983], "potassium": [50971],
            "glucose": [50931], "bicarbonate": [50882], "bun": [51006],
            "lactate": [50813], "hematocrit": [51221], "rbc": [51279],
            "mcv": [51250], "mchc": [51249], "rdw": [51277], "chloride": [50902],
            "calcium": [50893], "magnesium": [50960], "phosphate": [50970],
            "anion_gap": [50868], "alt": [50861], "ast": [50878],
            "bilirubin_total": [50885], "alkaline_phosphatase": [50863],
            "albumin": [50862], "inr": [51237], "pt": [51274], "ptt": [51275],
            "ph": [50820], "pco2": [50818], "po2": [50821], "base_excess": [50802],
            "troponin_t": [51003], "ck": [50910], "ck_mb": [50911], "crp": [50889],
        }

        filepath = f"{self.mimiciv_path}/labevents.csv.gz"
        cols_to_use = ["hadm_id", "subject_id", "itemid", "valuenum", "charttime"]
        item_ids_to_check = set(iid for sub in critical_labs.values() for iid in sub)

        all_chunks = []
        try:
            total_rows = self._get_row_count(filepath)
            pbar = tqdm(total=total_rows, desc="Elaborazione labevents", unit=" rows")
            for chunk in pd.read_csv(filepath, usecols=cols_to_use, chunksize=2_000_000):
                filtered = chunk[chunk["itemid"].isin(item_ids_to_check)]
                if not filtered.empty:
                    filtered = filtered.dropna(subset=["valuenum", "hadm_id"])
                    filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
                    filtered = filtered.dropna(subset=["charttime"])
                    all_chunks.append(filtered)
                pbar.update(len(chunk))
            pbar.close()

            if not all_chunks:
                self.lab_features = pd.DataFrame(columns=["hadm_id"])
                return

            labevents = pd.concat(all_chunks, ignore_index=True)
            lab_features_list = []
            for lab_name, itemids in critical_labs.items():
                data = labevents[labevents["itemid"].isin(itemids)]
                if not data.empty:
                    stats = data.groupby("hadm_id").agg({"valuenum": ["mean", "max", "min", "std"]}).reset_index()
                    stats.columns = ["hadm_id", f"{lab_name}_mean", f"{lab_name}_max", f"{lab_name}_min", f"{lab_name}_std"]
                    trends = self._calculate_lab_trends(data, lab_name)
                    stats = stats.merge(trends, on="hadm_id", how="left")
                    lab_features_list.append(stats)

            if lab_features_list:
                self.lab_features = lab_features_list[0]
                for df in lab_features_list[1:]:
                    self.lab_features = self.lab_features.merge(df, on="hadm_id", how="outer")
                self.lab_features.to_parquet(self.cp_lab_features, index=False)
                print(f"Checkpoint lab_features saved ({len(self.lab_features)} ricoveri).")
            else:
                self.lab_features = pd.DataFrame(columns=["hadm_id"])
        except Exception as e:
            print(f"Error reading labevents: {e}")
            self.lab_features = pd.DataFrame(columns=["hadm_id"])

    def _calculate_vital_trends(self, vital_data: pd.DataFrame, vital_name: str) -> pd.DataFrame:
        trend = vital_data.copy().sort_values(["hadm_id", "charttime"])
        trend["time_diff"] = trend.groupby("hadm_id")["charttime"].diff().dt.total_seconds() / 3600
        trend["value_diff"] = trend.groupby("hadm_id")["valuenum"].diff()
        valid = trend["time_diff"] > 0
        trend.loc[valid, "slope"] = trend.loc[valid, "value_diff"] / trend.loc[valid, "time_diff"]

        trend_stats = trend.groupby("hadm_id").agg(
            {"slope": ["mean", "max", "min"], "value_diff": ["mean", "std"]}
        ).reset_index()
        trend_stats.columns = [
            "hadm_id",
            f"{vital_name}_slope_mean",
            f"{vital_name}_slope_max",
            f"{vital_name}_slope_min",
            f"{vital_name}_change_mean",
            f"{vital_name}_change_std",
        ]
        return trend_stats

    def _process_vitals_events_advanced(self):
        print("\nExtracting vital signs from chartevents...")
        vital_items = {
            "heart_rate": [220045, 211],
            "sbp": [220050, 51, 442, 455, 6701, 220179],
            "dbp": [220051, 8368, 8440, 8441, 8555, 220180],
            "mbp": [220052, 456, 52, 6702, 220181],
            "resp_rate": [220210, 618, 615, 224690, 224689],
            "spo2": [220277, 646, 834],
            "temperature": [223762, 676, 677, 678, 679, 223761],
        }

        filepath = f"{self.icu_path}/chartevents.csv.gz"
        cols_to_use = ["hadm_id", "subject_id", "itemid", "valuenum", "charttime"]
        item_ids_to_check = set(iid for sub in vital_items.values() for iid in sub)

        all_chunks = []
        try:
            total_rows = self._get_row_count(filepath)
            pbar = tqdm(total=total_rows, desc="Elaborazione chartevents", unit=" rows")
            for chunk in pd.read_csv(
                filepath,
                usecols=cols_to_use,
                chunksize=5_000_000,
                dtype={"hadm_id": "float64", "subject_id": "int64", "itemid": "int64", "valuenum": "float64"},
            ):
                filtered = chunk[chunk["itemid"].isin(item_ids_to_check)]
                if not filtered.empty:
                    filtered = filtered.dropna(subset=["valuenum", "hadm_id"])
                    filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
                    filtered = filtered.dropna(subset=["charttime"])
                    all_chunks.append(filtered)
                pbar.update(len(chunk))
            pbar.close()

            if not all_chunks:
                self.vitals_features = pd.DataFrame(columns=["hadm_id"])
                return

            chartevents = pd.concat(all_chunks, ignore_index=True)
            chartevents["hadm_id"] = chartevents["hadm_id"].astype(int)

            vitals_features_list = []
            for vital_name, itemids in vital_items.items():
                data = chartevents[chartevents["itemid"].isin(itemids)]
                if not data.empty:
                    stats = data.groupby("hadm_id").agg({"valuenum": ["mean", "max", "min", "std"]}).reset_index()
                    stats.columns = ["hadm_id", f"{vital_name}_mean", f"{vital_name}_max", f"{vital_name}_min", f"{vital_name}_std"]
                    trends = self._calculate_vital_trends(data, vital_name)
                    stats = stats.merge(trends, on="hadm_id", how="left")
                    stats[f"{vital_name}_stable"] = (stats[f"{vital_name}_std"] < stats[f"{vital_name}_mean"] * 0.2).astype(int)
                    vitals_features_list.append(stats)

            if vitals_features_list:
                self.vitals_features = vitals_features_list[0]
                for df in vitals_features_list[1:]:
                    self.vitals_features = self.vitals_features.merge(df, on="hadm_id", how="outer")

                stable_cols = [c for c in self.vitals_features.columns if "_stable" in c]
                if stable_cols:
                    self.vitals_features["vital_stability_score"] = self.vitals_features[stable_cols].mean(axis=1)

                self.vitals_features.to_parquet(self.cp_vitals_features, index=False)
                print(f"Checkpoint vitals_features saved ({len(self.vitals_features)} ricoveri).")
            else:
                self.vitals_features = pd.DataFrame(columns=["hadm_id"])
        except Exception as e:
            print(f"Error reading chartevents: {e}")
            self.vitals_features = pd.DataFrame(columns=["hadm_id"])

    # ══════════════════════════════════════════════════════════════════════════
    #  TABULAR PREPROCESSING & LONGITUDINAL HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def preprocess_data(self):
        """Merges admission, demographic, diagnosis, prescription, and historical data."""
        print("\n--- 2. Preprocessing & Feature Engineering ---")

        self.admissions["admittime"] = pd.to_datetime(self.admissions["admittime"])
        self.admissions["dischtime"] = pd.to_datetime(self.admissions["dischtime"])

        admissions_patients = self.admissions.merge(self.patients, on="subject_id")
        admissions_patients["age"] = admissions_patients["anchor_age"]

        age_bins = [0, 18, 30, 45, 60, 75, 91]
        age_labels = ["Pediatric", "Young_Adult", "Adult", "Middle_Age", "Senior", "Elderly"]
        admissions_patients["age_group"] = pd.cut(
            admissions_patients["age"], bins=age_bins, labels=age_labels
        ).astype(str)

        # Grouped diagnoses
        if "ICD_CATEGORY" in self.diagnoses_icd.columns:
            self.diagnoses_icd["ICD_CATEGORY"] = self.diagnoses_icd["ICD_CATEGORY"].astype(str).fillna("Unknown")
            diag_code_col = "mapped_icd10" if "mapped_icd10" in self.diagnoses_icd.columns else "icd_code"
            diagnoses_grouped = self.diagnoses_icd.groupby("hadm_id").agg(
                diagnoses=(diag_code_col, list),
                diagnosis_categories_list=("ICD_CATEGORY", "unique"),
            ).reset_index()
        else:
            diagnoses_grouped = self.diagnoses_icd.groupby("hadm_id").agg(
                diagnoses=("icd_code", list),
                diagnosis_categories_list=("icd_code", "unique"),
            ).reset_index()

        diagnoses_grouped["comorbidity_count"] = diagnoses_grouped["diagnoses"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        diagnoses_grouped["complex_case"] = diagnoses_grouped["comorbidity_count"] > 5

        # Aggregated prescriptions count
        prescriptions_grouped = self.prescriptions.groupby("hadm_id").agg(
            drugs=("drug", list),
            drug_count=("dose_val_rx", "count"),
        ).reset_index()

        print("Merging tables...")
        self.combined_data = admissions_patients.merge(diagnoses_grouped, on="hadm_id", how="left")
        self.combined_data = self.combined_data.merge(prescriptions_grouped, on="hadm_id", how="left")

        if self.lab_features is not None and not self.lab_features.empty:
            self.combined_data = self.combined_data.merge(self.lab_features, on="hadm_id", how="left")

        if self.vitals_features is not None and not self.vitals_features.empty:
            self.combined_data = self.combined_data.merge(self.vitals_features, on="hadm_id", how="left")

        # Diagnosis history (previous patient admissions sorted chronologically)
        print("Calculating previous admissions diagnosis history...")
        self.combined_data = self.combined_data.sort_values(["subject_id", "admittime"])

        previous_diagnoses_col = []
        for _, group in tqdm(self.combined_data.groupby("subject_id", sort=False), desc="Building History"):
            history_so_far = set()
            for current_diags in group["diagnoses"]:
                previous_diagnoses_col.append(list(history_so_far))
                if isinstance(current_diags, list):
                    history_so_far.update(current_diags)
                elif pd.notna(current_diags):
                    history_so_far.add(current_diags)

        self.combined_data["previous_diagnoses"] = previous_diagnoses_col

        # Outcome (LOS and mortality)
        self.combined_data["los_days"] = (
            self.combined_data["dischtime"] - self.combined_data["admittime"]
        ).dt.total_seconds() / (24 * 3600)

        if "hospital_expire_flag" in self.combined_data.columns:
            self.combined_data["mortality"] = self.combined_data["hospital_expire_flag"].fillna(0).astype(int)
        elif "deathtime" in self.combined_data.columns:
            self.combined_data["mortality"] = self.combined_data["deathtime"].notna().astype(int)
        else:
            self.combined_data["mortality"] = 0

        self.combined_data.to_parquet(self.cp_combined_data, index=False)
        print(f"Final checkpoint saved: {self.cp_combined_data} ({len(self.combined_data)} ricoveri)")

    # ══════════════════════════════════════════════════════════════════════════
    #  PRESCRIPTIONS BY DATE & PHARMACOLOGICAL CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    def _process_prescriptions_by_date(self):
        print("\nBuilding prescriptions by date map...")
        df = self.prescriptions.dropna(subset=["drug", "starttime"]).copy()
        df["date"] = pd.to_datetime(df["starttime"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])

        drug_cat_dict = {
            "ANTIBIOTIC_ANTIBACTERIAL": ["vancomycin", "cefepime", "piperacillin", "tazobactam", "zosyn", "meropenem", "ceftriaxone", "ciprofloxacin", "levofloxacin", "azithromycin", "metronidazole", "flagyl", "gentamicin", "tobramycin", "amoxicillin", "ampicillin", "doxycycline", "bactrim", "sulfamethoxazole", "linezolid", "cefazolin"],
            "ANTIFUNGAL": ["fluconazole", "diflucan", "micafungin", "caspofungin", "amphotericin", "voriconazole", "nystatin"],
            "ANTIVIRAL": ["acyclovir", "valacyclovir", "ganciclovir", "oseltamivir", "tamiflu", "remdesivir"],
            "CARDIO_ANTIHYPERTENSIVE": ["metoprolol", "atenolol", "carvedilol", "labetalol", "amlodipine", "diltiazem", "nicardipine", "lisinopril", "enalapril", "losartan", "valsartan", "hydralazine", "clonidine", "nitroprusside", "nitroglycerin", "isosorbide"],
            "CARDIO_DIURETIC": ["furosemide", "lasix", "spironolactone", "aldactone", "hydrochlorothiazide", "hctz", "bumetanide", "torsemide", "acetazolamide"],
            "CARDIO_VASOPRESSOR": ["norepinephrine", "levophed", "epinephrine", "dopamine", "dobutamine", "phenylephrine", "vasopressin", "milrinone"],
            "CARDIO_ANTIARRHYTHMIC": ["amiodarone", "digoxin", "adenosine", "sotalol", "flecainide"],
            "ANTICOAGULANT_ANTIPLATELET": ["heparin", "enoxaparin", "lovenox", "warfarin", "coumadin", "aspirin", "clopidogrel", "plavix", "apixaban", "eliquis", "rivaroxaban", "xarelto", "bivalirudin", "argatroban"],
            "ANALGESIC_OPIOID": ["morphine", "fentanyl", "hydromorphone", "dilaudid", "oxycodone", "oxycontin", "percocet", "tramadol", "methadone", "codeine"],
            "ANALGESIC_NON_OPIOID": ["acetaminophen", "tylenol", "ibuprofen", "motrin", "ketorolac", "toradol", "celecoxib", "naproxen"],
            "SEDATIVE_ANXIOLYTIC": ["propofol", "diprivan", "midazolam", "versed", "lorazepam", "ativan", "diazepam", "valium", "dexmedetomidine", "precedex", "haloperidol", "haldol", "quetiapine", "seroquel", "olanzapine"],
            "ANTIDEPRESSANT": ["sertraline", "fluoxetine", "citalopram", "escitalopram", "venlafaxine", "duloxetine", "bupropion", "trazodone", "mirtazapine"],
            "ANTIEPILEPTIC": ["levetiracetam", "keppra", "phenytoin", "dilantin", "valproic", "depakote", "carbamazepine", "lamotrigine"],
            "DIABETES": ["insulin", "metformin", "glipizide", "glyburide", "glimepiride", "pioglitazone", "sitagliptin", "linagliptin", "canagliflozin", "empagliflozin"],
            "CORTICOSTEROID": ["prednisone", "methylprednisolone", "solu-medrol", "dexamethasone", "decadron", "hydrocortisone", "fludrocortisone"],
            "GI_ACID_REFLUX": ["omeprazole", "pantoprazole", "lansoprazole", "esomeprazole", "famotidine", "pepcid", "ranitidine"],
            "GI_LAXATIVE_BOWEL": ["lactulose", "senna", "docusate", "bisacodyl", "polyethylene glycol", "miralax", "psyllium"],
            "BRONCHODILATOR": ["albuterol", "ipratropium", "duoneb", "combivent", "tiotropium", "salmeterol", "formoterol", "montelukast"],
            "FLUIDS_ELECTROLYTES": ["albumin", "magnesium sulfate", "magnesium oxide", "potassium chloride", "kcl", "calcium gluconate", "sodium bicarbonate"],
            "THYROID": ["levothyroxine", "synthroid", "liothyronine", "methimazole"],
            "CHEMOTHERAPY": ["chemo", "methotrexate", "cyclophosphamide", "doxorubicin", "cisplatin", "carboplatin", "rituximab", "fluorouracil"],
        }

        def _get_cat(drug_name):
            d_low = str(drug_name).lower()
            for cat, kws in drug_cat_dict.items():
                if any(kw in d_low for kw in kws):
                    return cat
            if "laxative" in d_low or "stool" in d_low: return "LAXATIVE"
            if "vitamin" in d_low or "vit " in d_low: return "VITAMIN"
            if "antacid" in d_low or "ppi" in d_low or "proton" in d_low: return "GI_AGENT"
            if "antihistamine" in d_low or "allergy" in d_low: return "ANTIHISTAMINE"
            if "chemotherapy" in d_low or "chemo" in d_low: return "CHEMOTHERAPY"
            return "OTHER"

        unique_drugs = df["drug"].unique()
        cat_map = {d: _get_cat(d) for d in tqdm(unique_drugs, desc="Mapping farmaci")}
        df["category"] = df["drug"].map(cat_map)

        result_map = {}
        df_sorted = df.sort_values("starttime")
        for hadm_id_val, pat_df in tqdm(df_sorted.groupby("hadm_id", sort=False), desc="Aggregazione per data"):
            dates_dict = {}
            for date_str, grp in pat_df.groupby("date", sort=True):
                drugs_for_date = []
                seen = set()
                for _, rx_row in grp.iterrows():
                    drug_name = str(rx_row["drug"])
                    cat = str(rx_row["category"])
                    key = (drug_name, cat)
                    if key not in seen:
                        seen.add(key)
                        parts = [drug_name]
                        dose_val = rx_row.get("dose_val_rx")
                        dose_unit = rx_row.get("dose_unit_rx")
                        route = rx_row.get("route")
                        if pd.notna(dose_val) and str(dose_val).strip(): parts.append(str(dose_val))
                        if pd.notna(dose_unit) and str(dose_unit).strip(): parts.append(str(dose_unit))
                        if pd.notna(route) and str(route).strip(): parts.append(str(route))
                        parts.append(f"[{cat}]")
                        drugs_for_date.append(" ".join(parts))
                dates_dict[date_str] = drugs_for_date
            result_map[int(hadm_id_val)] = dates_dict

        self._prescriptions_by_date_map = result_map
        with open(self.cp_prescriptions_by_date, "wb") as f:
            pickle.dump(result_map, f)
        print(f"Prescriptions by date saved: {self.cp_prescriptions_by_date} ({len(result_map)} ricoveri)")

    # ══════════════════════════════════════════════════════════════════════════
    #  FASTTEXT TRAINING & IDF WEIGHTS
    # ══════════════════════════════════════════════════════════════════════════

    def _create_diagnosis_corpus(self) -> List[List[str]]:
        corpus = []
        for _, row in self.combined_data.iterrows():
            codes = row.get("diagnosis_categories_list")
            if codes is not None and hasattr(codes, "__iter__") and len(codes) > 0:
                c_list = [str(c) for c in codes if str(c) not in ["nan", "None", ""]]
                if c_list:
                    corpus.append(c_list)
        return corpus

    def _calculate_idf(self, corpus: List[List[str]]) -> Dict[str, float]:
        n_docs = len(corpus)
        df_count = defaultdict(int)
        for codes in corpus:
            for c in set(codes):
                df_count[c] += 1
        return {code: math.log(n_docs / (freq + 1)) for code, freq in df_count.items()}

    def _train_diagnosis_embedding_models(self):
        corpus = self._create_diagnosis_corpus()
        if not corpus:
            print("Empty diagnosis corpus, cannot train FastText.")
            return

        print(f"Calculating IDF on {len(corpus)} ricoveri...")
        self.diagnosis_idf_map = self._calculate_idf(corpus)
        with open(self.idf_map_path, "wb") as f:
            pickle.dump(self.diagnosis_idf_map, f)

        print(f"Training FastText (dim={self.diagnosis_embedding_dim}, window=5, min_count=2, epochs=5)...")
        ft_model = FastText(
            sentences=corpus,
            vector_size=self.diagnosis_embedding_dim,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            min_n=2,
            max_n=5,
            epochs=5,
        )
        ft_model.wv.save(self.ft_model_path)
        self.diagnosis_ft_model = ft_model.wv
        print(f"FastText model saved in {self.ft_model_path}")

    def _load_diagnosis_embedding_models(self):
        if os.path.exists(self.idf_map_path):
            with open(self.idf_map_path, "rb") as f:
                self.diagnosis_idf_map = pickle.load(f)
        if os.path.exists(self.ft_model_path):
            self.diagnosis_ft_model = KeyedVectors.load(self.ft_model_path)

    def _get_weighted_diagnosis_embedding(self, codes_list) -> np.ndarray:
        dim = self.diagnosis_embedding_dim
        if codes_list is None:
            return np.zeros(dim)
        if hasattr(codes_list, "size") and codes_list.size == 0:
            return np.zeros(dim)
        if not hasattr(codes_list, "__iter__") or len(codes_list) == 0:
            return np.zeros(dim)

        unique_codes = list(set(str(c) for c in codes_list if str(c) not in ["nan", "None", ""]))
        if not unique_codes or self.diagnosis_ft_model is None:
            return np.zeros(dim)

        vectors = []
        weights = []
        for code in unique_codes:
            if code in self.diagnosis_ft_model:
                vectors.append(self.diagnosis_ft_model[code])
                base_w = self.diagnosis_idf_map.get(code, 1.0) if self.diagnosis_idf_map else 1.0
                weights.append(base_w ** self.power_factor)

        if not vectors:
            return np.zeros(dim)
        if not weights:
            return np.mean(vectors, axis=0)
        return np.average(vectors, axis=0, weights=weights)

    # ══════════════════════════════════════════════════════════════════════════
    #  CREATING THE 3 EMBEDDING SPACES
    # ══════════════════════════════════════════════════════════════════════════

    def create_patient_embeddings(self):
        """Genera i tre embedding complementari: Numeric (52), Categorical (8), Diagnosis (100)."""
        print("\n--- 3. Creating Embedding Spaces ---")

        # 1. Numerical Features (Golden Features)
        target_features = ["age", "comorbidity_count"]
        vitals_focus = [
            "heart_rate_mean", "heart_rate_std", "mbp_mean",
            "resp_rate_mean", "spo2_mean", "temperature_mean",
        ]
        labs_focus = [
            "creatinine_mean", "bun_mean", "sodium_mean", "potassium_mean",
            "calcium_mean", "magnesium_mean", "chloride_mean", "phosphate_mean", "anion_gap_mean",
            "wbc_mean", "hemoglobin_mean", "platelets_mean", "hematocrit_mean",
            "rbc_mean", "mcv_mean", "rdw_mean", "crp_mean",
            "alt_mean", "ast_mean", "bilirubin_total_mean", "alkaline_phosphatase_mean", "albumin_mean",
            "inr_mean", "pt_mean", "ptt_mean",
            "lactate_mean", "ph_mean", "pco2_mean", "po2_mean", "base_excess_mean",
            "glucose_mean", "bicarbonate_mean",
            "troponin_t_mean", "ck_mean", "ck_mb_mean",
        ]
        trends_focus = [
            "creatinine_slope_mean", "lactate_slope_mean", "bun_slope_mean",
            "wbc_slope_mean", "hemoglobin_slope_mean", "platelets_slope_mean",
            "ph_slope_mean", "pco2_slope_mean", "troponin_t_slope_mean",
        ]

        candidate_cols = target_features + vitals_focus + labs_focus + trends_focus
        selected_numeric = [c for c in candidate_cols if c in self.combined_data.columns]
        print(f"Selected continuous numerical features: {len(selected_numeric)}")

        if selected_numeric:
            num_data = self.combined_data[selected_numeric].fillna(0).replace([np.inf, -np.inf], 0)
            scaled_num = self.scaler.fit_transform(num_data)
            weighted_numeric = scaled_num * self.embedding_weights["numeric"]
            norms = np.linalg.norm(weighted_numeric, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.patient_embeddings_numeric = weighted_numeric / norms
        else:
            self.patient_embeddings_numeric = np.zeros((len(self.combined_data), 1))

        # 2. Categorical Features (Demographics)
        cat_features = ["gender", "age_group"]
        avail_cats = [c for c in cat_features if c in self.combined_data.columns]
        if avail_cats:
            cat_df = self.combined_data[avail_cats].astype(str).fillna("UNKNOWN")
            cat_ohe = pd.get_dummies(cat_df, columns=avail_cats, dtype=int).values
            weighted_cat = cat_ohe * self.embedding_weights["categorical"]
            norms = np.linalg.norm(weighted_cat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.patient_embeddings_categorical = weighted_cat / norms
        else:
            self.patient_embeddings_categorical = np.zeros((len(self.combined_data), 1))

        # 3. Diagnoses (IDF-weighted FastText)
        if not os.path.exists(self.ft_model_path) or not os.path.exists(self.idf_map_path):
            self._train_diagnosis_embedding_models()
        else:
            self._load_diagnosis_embedding_models()

        print("Calculating diagnosis vectors for each admission...")
        diag_embs = []
        for _, row in tqdm(self.combined_data.iterrows(), total=len(self.combined_data), desc="Embedding diagnosi"):
            codes = row.get("diagnosis_categories_list", [])
            diag_embs.append(self._get_weighted_diagnosis_embedding(codes))

        diag_matrix = np.array(diag_embs)
        weighted_diag = diag_matrix * self.embedding_weights["diagnosis"]
        norms = np.linalg.norm(weighted_diag, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.patient_embeddings_diagnosis = weighted_diag / norms

        # Saving to disk
        print("Saving .npy files and scaler...")
        np.save(self.cp_embeddings_numeric, self.patient_embeddings_numeric)
        np.save(self.cp_embeddings_categorical, self.patient_embeddings_categorical)
        np.save(self.cp_embeddings_diagnosis, self.patient_embeddings_diagnosis)
        joblib.dump(self.scaler, self.cp_scaler)
        print("Embeddings saved.")

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVING SLIM PARQUET (for pipeline DataGatherer)
    # ══════════════════════════════════════════════════════════════════════════

    def save_slim_checkpoint(self):
        """Estrae le sole colonne richieste da DataGatherer e ContextConstructor."""
        slim_cols = [
            "hadm_id", "subject_id",
            "age", "gender", "race", "marital_status", "language",
            "admission_type", "insurance",
            "mortality", "los_days",
            "diagnoses", "diagnosis_categories_list", "previous_diagnoses",
            # Vital Signs
            "heart_rate_mean", "mbp_mean", "resp_rate_mean", "spo2_mean", "temperature_mean",
            # Laboratories
            "creatinine_mean", "bun_mean", "glucose_mean", "sodium_mean", "potassium_mean",
            "calcium_mean", "magnesium_mean", "anion_gap_mean", "wbc_mean", "hemoglobin_mean",
            "platelets_mean", "hematocrit_mean", "crp_mean", "lactate_mean",
            "alt_mean", "ast_mean", "bilirubin_total_mean", "albumin_mean",
            "alkaline_phosphatase_mean", "inr_mean", "ptt_mean",
            "ph_mean", "pco2_mean", "po2_mean", "base_excess_mean",
            "troponin_t_mean", "ck_mb_mean",
        ]

        avail = [c for c in slim_cols if c in self.combined_data.columns]
        slim_df = self.combined_data[avail].copy()
        slim_df.to_parquet(self.cp_combined_data_slim, index=False)
        mb = os.path.getsize(self.cp_combined_data_slim) / 1e6
        print(f"[*] Slim parquet saved: {self.cp_combined_data_slim} ({mb:.1f} MB, {len(avail)} colonne)")

    # ══════════════════════════════════════════════════════════════════════════
    #  MILVUS MANAGEMENT & INDEXING
    # ══════════════════════════════════════════════════════════════════════════

    def _save_embeddings_to_milvus_multiple(self):
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            print(f"Connected to Milvus ({self.milvus_host}:{self.milvus_port})")

            embedding_map = {
                "numeric": self.patient_embeddings_numeric,
                "categorical": self.patient_embeddings_categorical,
                "diagnosis": self.patient_embeddings_diagnosis,
            }

            for embed_type, embs in embedding_map.items():
                if embs is None or embs.shape[1] < 2:
                    continue

                coll_name = self.collection_names[embed_type]
                dim = embs.shape[1]

                if not utility.has_collection(coll_name):
                    fields = [
                        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                        FieldSchema(name="hadm_id", dtype=DataType.INT64),
                        FieldSchema(name="subject_id", dtype=DataType.INT64),
                        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                    ]
                    schema = CollectionSchema(fields, f"MIMIC-IV {embed_type.capitalize()} Embeddings")
                    collection = Collection(coll_name, schema)
                    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
                else:
                    collection = Collection(coll_name)

                if collection.num_entities == 0:
                    print(f"Inserting {len(embs)} vectors into '{coll_name}'...")
                    total = len(self.combined_data)
                    batch_size = 5000
                    for start in range(0, total, batch_size):
                        end = min(start + batch_size, total)
                        ids = list(range(start, end))
                        hadms = self.combined_data.iloc[start:end]["hadm_id"].astype(int).tolist()
                        subjects = self.combined_data.iloc[start:end]["subject_id"].astype(int).tolist()
                        vectors = embs[start:end].astype(np.float32).tolist()
                        collection.insert([ids, hadms, subjects, vectors])
                    collection.flush()
                    collection.load()
                    print(f"Collection '{coll_name}' caricata: {collection.num_entities} entities.")
                else:
                    collection.load()
                    print(f"Collection '{coll_name}' already populated: {collection.num_entities} entities.")

        except Exception as e:
            print(f"Milvus indexing warning: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV Preprocessing & Checkpoint Builder per Pipeline Multi-Agente",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mimic-path", type=str, default="./data/hosp", help="Path cartella hosp MIMIC-IV")
    parser.add_argument("--icu-path", type=str, default="./data/icu", help="Path cartella icu MIMIC-IV")
    parser.add_argument("--categories-path", type=str, default=None, help="Path al file CSV mapping diagnosi (diagnoses_4.csv)")
    parser.add_argument("--checkpoint-path", type=str, default="./data/pipeline_checkpoints", help="Cartella output checkpoint")
    parser.add_argument("--models-path", type=str, default="./data/pipeline_models", help="Cartella modelli FastText/IDF")
    parser.add_argument("--milvus-host", type=str, default="localhost", help="Host Milvus")
    parser.add_argument("--milvus-port", type=str, default="19530", help="Porta Milvus")
    parser.add_argument("--skip-milvus", action="store_true", help="Salta la connessione e indicizzazione in Milvus")
    parser.add_argument("--no-checkpoints", action="store_true", help="Ignora i checkpoint parziali e ricalcola tutto da zero")
    parser.add_argument("--diagnosis-dim", type=int, default=100, help="Dimensione vettore FastText diagnosi")
    parser.add_argument("--power-factor", type=float, default=2.0, help="Esponente alpha di amplificazione IDF")

    args = parser.parse_args()

    builder = MIMICIVCheckpointBuilder(
        mimiciv_path=args.mimic_path,
        icu_path=args.icu_path,
        categories_path=args.categories_path,
        checkpoint_path=args.checkpoint_path,
        models_path=args.models_path,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        skip_milvus=args.skip_milvus,
        use_checkpoints=not args.no_checkpoints,
        diagnosis_embedding_dim=args.diagnosis_dim,
        power_factor=args.power_factor,
    )
    builder.run()


if __name__ == "__main__":
    main()
