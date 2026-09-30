
# MedPSR-Agent: Multi-Agent Clinical Decision Support System with Patient Similarity Retrieval on MIMIC-IV

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Database Milvus](https://img.shields.io/badge/VectorDB-Milvus_v2.x-orange.svg)](https://milvus.io/)
[![Dataset MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV_v3.1-green.svg)](https://physionet.org/content/mimiciv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper:  
**"A Multi-Agent Clinical Decision Support System with Patient Similarity Retrieval on MIMIC-IV"**

---

## Overview & Abstract

We present a multi-agent clinical decision support framework that combines structured electronic health record (EHR) data from **MIMIC-IV v3.1** with large language model (LLM)-driven clinical reasoning. 

The system processes hospital admissions through a specialized data pipeline producing three complementary patient embedding spaces:
1. **Numeric Representation:** Standardized vital signs, lab panels, and physiological measurements.
2. **Categorical Representation:** Demographic attributes, admission metadata, and procedural encodings.
3. **Diagnostic Representation:** High-dimensional ICD-9/10 representations generated via **IDF-weighted FastText** embeddings.

All vectors are indexed within a **Milvus** vector database to enable sub-second patient similarity retrieval.

### Multi-Agent Architecture

The reasoning workflow is orchestrated across **six specialized agents**:
- **Query Router Agent:** Interprets free-text clinician inquiries, extracts clinical intents, and extracts search parameters.
- **Similarity Retrieval Engine:** Executes a two-stage retrieval pipeline using Approximate Nearest Neighbor (ANN) search followed by multi-space re-ranking.
- **Context Constructor Agent:** Gathers and formats longitudinal patient trajectories and retrieved peer cohorts into structured JSON evidence payloads.
- **Specialist Clinical Agents (Diagnostic, Prognostic, Treatment):** Domain-specific LLM instances performing temporally aware, evidence-backed clinical evaluations.
- **Synthesis Agent:** Merges disparate specialist findings into a cohesive, structured clinical consultation report.

```
[Physician Query] ──> [Query Router]
                             │
                             ▼
  [Milvus Vector DB] ──> [Similarity Engine] (ANN + Multi-Embedding Re-ranking)
                             │
                             ▼
                     [Context Constructor] (Structured JSON Evidence)
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     [Diagnostic Agent] [Prognostic Agent] [Therapeutic Agent]
            └────────────────┬────────────────┘
                             ▼
                     [Synthesis Agent] ──> [Structured Clinical Report]
```

---

## 📂 Repository Structure & File Manifest

Below is the directory map and functional description of the repository modules:

```text
clinsim-agent/
├── configs/
│   ├── agents.yaml               # Agent system prompts, temperature settings, and model targets
│   ├── database.yaml             # Milvus collection schemas, index types, and connection settings
│   └── embeddings.yaml           # Dimension definitions and weights for multi-vector ranking
│
├── data_pipeline/
│   ├── __init__.py
│   ├── extract_mimic.py          # SQL/DuckDB queries to extract raw cohorts from MIMIC-IV v3.1
│   ├── preprocess_clinical.py    # Handling missing values, unit standardization, and normalization
│   └── fasttext_diagnosis.py     # IDF-weighted FastText encoder training and inference for ICD codes
│
├── storage/
│   ├── __init__.py
│   ├── milvus_client.py          # Milvus connection handlers and collection lifecycle management
│   └── indexer.py                # Batch-indexing numeric, categorical, and diagnostic vectors
│
├── agents/
│   ├── __init__.py
│   ├── router.py                 # Physician query parsing and intent routing
│   ├── retriever.py              # Two-stage ANN retrieval and multi-embedding re-ranking logic
│   ├── context_builder.py        # Assembles structured JSON evidence and temporal timelines
│   ├── specialists.py            # Diagnostic, prognostic, and therapeutic specialist agents
│   └── synthesizer.py            # Synthesizes specialist outputs into a final clinical report
│
├── evaluation/
│   ├── __init__.py
│   ├── case_study_eval.py        # Qualitative case study reproduction and trace outputs
│   └── retrieval_metrics.py      # NDCG, Precision@K, and clinical cohesion metrics
│
├── tests/
│   ├── test_pipeline.py          # Unit tests for embedding generation
│   └── test_retrieval.py         # Integration tests for vector search and re-ranking
│
├── .env.example                  # Environment variable template (Milvus host, LLM API keys)
├── requirements.txt              # Python package dependencies
├── setup.py                      # Package installation script
└── README.md                     # Project documentation and reproduction guidelines
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python $\ge$ 3.10
- Running [Milvus 2.x](https://milvus.io/docs/install_standalone-docker.md) instance (Docker or Cloud)
- Credentialed access to [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/)

### 2. Environment Configuration
Clone the repository and install required packages:
```bash
git clone https://github.com/your-username/MedPSR-Agent.git
cd MedPSR-Agent
pip install -r requirements.txt
cp .env.example .env
```
Fill in `.env` with your OpenAI/Anthropic/Local LLM API credentials and Milvus endpoint.

### 3. Pipeline Execution
1. **Preprocess and compute embeddings:**
   ```bash
   python data_pipeline/preprocess_clinical.py --mimic_dir /path/to/mimiciv/
   python data_pipeline/fasttext_diagnosis.py --build_idf
   ```
2. **Index vectors into Milvus:**
   ```bash
   python storage/indexer.py --config configs/database.yaml
   ```
3. **Execute clinical inference:**
   ```bash
   python -m agents.router --query "Evaluate ICU patient with acute renal deterioration and refractory hypoxemia."
   ```

---

## 🔒 Data Use & Ethics Notice
This project uses the **MIMIC-IV v3.1** database. Users must complete human-subjects training via CITI and sign the PhysioNet Data Use Agreement before utilizing or preprocessing data with this framework. No protected health information (PHI) is included in this repository.
