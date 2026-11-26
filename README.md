# Clinical Decision Support System with Patient Similarity Retrieval on MIMIC-IV

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Database Milvus](https://img.shields.io/badge/VectorDB-Milvus_v2.x-orange.svg)](https://milvus.io/)
[![Dataset MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV_v3.1-green.svg)](https://physionet.org/content/mimiciv/)

A Multi-Agent Clinical Decision Support System (CDSS) built on **MIMIC-IV v3.1**, combining patient similarity vector search (via Milvus), multi-agent LLM reasoning (via Gemini / Gemma), and automated evaluation (via Ragas).

---

## Project Overview

This repository contains the architecture, agent framework, vector search engine, and evaluation benchmarks for a Clinical Decision Support System. 

The system leverages structured Electronic Health Record (EHR) data from hospital admissions in MIMIC-IV to perform:
1. **Multi-Space Patient Similarity Retrieval:** Searching across numeric (vital signs, lab measurements), categorical (demographics, admission details), and diagnostic (IDF-weighted ICD representations) embedding spaces indexed in a **Milvus** vector database.
2. **Context Construction:** Assembling longitudinal patient timelines, lab/vital trends, medication histories, and retrieved peer patient cohorts into structured JSON evidence payloads.
3. **Multi-Agent Clinical Reasoning:** Orchestrating query routing, domain specialist analysis (diagnostic pattern analysis, pharmacological consultation, risk assessment), and final clinical report synthesis.
4. **Automated Ragas Evaluation:** Assessing response quality, clinical validity, contradiction rate, answer relevancy, and context precision using a custom Ragas adapter for the Google `google-genai` SDK.

---

## Repository Structure & File Manifest

The codebase is organized into modular packages and execution utilities:

```text
github/
├── agents/                           # Multi-Agent Framework
│   ├── __init__.py
│   ├── context_constructor.py        # Assembles structured clinical evidence & cohort timelines
│   ├── data_gatherer.py              # Interfaces with Milvus DB & preprocessed clinical records
│   ├── orchestrator.py               # Coordinates query routing, specialist execution & synthesis
│   ├── report_synthesizer.py         # Merges specialist evaluations into a structured report
│   ├── similarity_engine.py          # Multi-vector similarity retrieval & re-ranking engine
│   └── specialist_llm.py             # Interface for LLM-driven domain specialists (Gemini/Gemma)
│
├── config/                           # System Configuration
│   ├── __init__.py
│   └── settings.py                   # Centralized settings (LLM models, Milvus params, feature maps)
│
├── core/                             # Core Execution Pipeline & CLI
│   ├── __init__.py
│   ├── main.py                       # CLI entry point for clinical consultations
│   └── pipeline.py                   # ClinicalPipeline orchestrating end-to-end execution
│
├── utils/                            # Data Processing & Ragas Evaluation Utilities
│   ├── build_checkpoints.py          # MIMIC-IV data processing, feature engineering & Milvus indexing
│   ├── generate_ragas_query_dataset.py # Auto-generates evaluation datasets and ground truth queries
│   ├── ragas_gemini_new_sdk_v2.py    # Custom Ragas LLM/Embedding wrapper for google-genai SDK
│   ├── ragas_custom_metrics_new.py   # Custom Ragas evaluation metrics (Clinical Validity, etc.)
│   ├── evaluate_with_ragas.py        # Benchmark execution script (Full Pipeline, Ablation, Monolithic)
│   ├── evaluate_retrieval_and_routing.py # Orchestrator routing, Jaccard threshold & FastText vs. TF-IDF benchmarks
│   ├── statistical_analysis.py       # Bootstrap CIs, SE, Wilcoxon/McNemar tests & Holm-Bonferroni correction
│   ├── results_full_pipeline_ragas.json  # Ragas evaluation results for full pipeline (100 cases)
│   ├── results_ragas_ablation_and_monolithic.json # Comparative evaluation results (Ablation vs Baseline)
│   ├── results_orchestrator_eval.csv # Orchestrator & Similarity retrieval evaluation (100 / 70 active cases)
│   ├── results_jaccard_thresholds.json # Multi-threshold Jaccard sensitivity data (0.2 to 0.5)
│   ├── results_fasttext_vs_tfidf.json # FastText vs TF-IDF retrieval benchmark (100 patients)
│   ├── results_statistical_analysis.json # Computed statistical summary (SE, CIs, p-values)
│   └── results_statistical_report.md # Human-readable statistical report and LaTeX tables
│
└── README.md                         # Project documentation
```

---

## Component Summary

### 1. Multi-Agent Framework (`agents/`)
- **Query Router & Orchestrator:** Categorizes clinician inquiries (e.g., diagnostic pattern analysis, pharmacological consultation) and dispatches tasks to specialized agent prompts.
- **Similarity Engine:** Executes multi-vector ANN retrieval across Milvus collections, weighting diagnostic codes, vital signs, and lab measurements to retrieve relevant peer patient cohorts.
- **Context Constructor:** Formats raw EHR records, temporal trends, and retrieved cohort statistics into structured, hallucination-resistant prompts for LLM specialists.
- **Specialist LLMs & Synthesizer:** Domain-focused LLMs evaluate the patient history alongside peer cohort evidence and generate a consolidated clinical consultation report.

### 2. Configuration & Pipeline (`config/` & `core/`)
- **Settings (`settings.py`):** Configures Milvus database connections, LLM endpoints (`gemma4:e2b`, `gemini-3.1-flash-lite-preview`, `gemma4:31b-cloud`, plus `gemini-3.1-pro-preview` as the RAGAS judge), weight vector allocations for similarity search, and feature mapping dictionaries for labs/vitals.
- **CLI Interface (`main.py`):** Enables running queries for specific admission IDs (`hadm_id`), with options to set target similarity count (`--k_similar`), toggle vector retrieval (`--no-similarity`), export context payloads (`--save-context`), and enable verbose logging.

### 3. Evaluation & Benchmarking (`utils/`)
- **Custom Ragas Adapter (`ragas_gemini_new_sdk_v2.py`):** Integrates Ragas evaluation metrics directly with Google's latest `google-genai` SDK.
- **Custom Metrics (`ragas_custom_metrics_new.py`):** Implements domain-specific evaluators such as Clinical Validity, Contradiction Rate, Answer Relevancy, and Context Precision.
- **Benchmark Datasets & Results (`results_*.json`, `results_*.csv`):** Contains evaluation outputs across 100 test cases comparing:
  - **Full Pipeline:** Multi-agent reasoning with patient similarity retrieval.
  - **Ablation Study:** Multi-agent reasoning without patient similarity retrieval.
  - **Monolithic Baseline:** Single LLM evaluation without specialized agents or similarity context.
  - **Routing & Retrieval Benchmarks (`evaluate_retrieval_and_routing.py`):** Orchestrator routing and Similarity Engine retrieval metrics (`results_orchestrator_eval.csv`, with 70 active similarity cases filtered via `sim_status == "success"`), Jaccard threshold sensitivity (`results_jaccard_thresholds.json`), and FastText vs. TF-IDF retrieval comparison (`results_fasttext_vs_tfidf.json`).
- **Statistical Analysis (`statistical_analysis.py`):** Reproduces all standard errors (SE), 10,000-resample 95% percentile bootstrap confidence intervals, paired Wilcoxon signed-rank and exact McNemar tests, and Holm-Bonferroni adjustments, exporting `results_statistical_analysis.json` and `results_statistical_report.md`.

---

## Data & Privacy Notice

* **Patient Privacy:** Raw MIMIC-IV clinical datasets, patient vector indices, and full database checkpoints are excluded from this repository in compliance with the **PhysioNet Credentialed Data Use Agreement (DUA)** and HIPAA privacy regulations.
* **Usage Requirements:** Running feature extraction or vector indexing locally requires credentialed access to MIMIC-IV v3.1 via PhysioNet, alongside configuring local environment settings (`.env`) for LLM API keys and Milvus vector storage endpoints.
