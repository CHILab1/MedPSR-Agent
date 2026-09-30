"""
generate_ragas_dataset.py — RAGAS Dataset Generator
══════════════════════════════════════════════════════
Generates an evaluation dataset for RAGAS by executing the pipeline
agentic pipeline on N random hadm_ids for a set of predefined clinical questions.

Output structure in results/:
  results/
    run_<timestamp>/
      raw/
        <hadm_id>_<question_id>.json    ← full pipeline output
      ragas_dataset.json                ← flat dataset for RAGAS
      ragas_dataset.csv                 ← CSV version
      run_summary.json                  ← run statistics

Usage:
    python generate_ragas_dataset.py
    python generate_ragas_dataset.py --n-patients 20 --seed 42
    python generate_ragas_dataset.py --n-patients 5 --no-similarity --dry-run
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Force UTF-8 output on Windows ──
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Add project directory to path ──
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings
from core.pipeline import AgenticPipeline
from utils.logging import get_logger, console

logger = get_logger("ragas_generator")

# ══════════════════════════════════════════════════════════════════════════════
#  CLINICAL QUESTIONS — predefined set for evaluation
#  Covers various topics: pattern_analysis, consulto_farmacologico, similarity
# ══════════════════════════════════════════════════════════════════════════════

CLINICAL_QUESTIONS = [
    {"id": "Q01", "text": "Qual è l'età e il sesso di questo paziente?",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics"]},

    {"id": "Q02", "text": "Estrai i dati anagrafici principali del paziente dal database.",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics"]},

    {"id": "Q03", "text": "Quali sono le informazioni di base del paziente (età, genere, data di ricovero)?",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics", "admission_details"]},

    {"id": "Q04", "text": "Fornisci un riepilogo dei dati demografici di questo caso.",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics"]},

    {"id": "Q05", "text": "Riassumi le caratteristiche generali del paziente.",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics", "admission_details"]},

    {"id": "Q06", "text": "Quali sono i dettagli di ammissione di questo ricovero?",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["admission_details"]},

    {"id": "Q07", "text": "Descrivi brevemente chi è il paziente in esame.",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics"]},

    {"id": "Q08", "text": "Estrai i dettagli demografici essenziali per questo paziente.",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics"]},

    {"id": "Q09", "text": "Quali informazioni di base abbiamo su questo soggetto?",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics"]},

    {"id": "Q10", "text": "Fai una lista dei dati anagrafici e di ricovero del paziente.",
     "category": "info_extraction",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["demographics", "admission_details"]},

    # ─── COMORBIDITY (Q11-Q20): analisi comorbidità, richiedono similarity ───
    {"id": "Q11", "text": "Quali sono le possibili comorbidità non ancora diagnosticate a questo paziente?",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q12", "text": "Analizza le patologie presenti in questo caso clinico.",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q13", "text": "Identifica i pattern di comorbidità e confrontali con pazienti simili nel database.",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q14", "text": "Fai un'analisi approfondita su quali altre malattie potrebbero insorgere nel paziente.",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses", "previous_diagnoses"]},

    {"id": "Q15", "text": "Descrivi il quadro generale delle diagnosi attribuite al paziente durante il ricovero.",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": False,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q16", "text": "Esistono correlazioni tra le diverse diagnosi di questo paziente e quelle di casi simili?",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q17", "text": "Qual è la percentuale di decesso dei pazienti simili?",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["outcome"]},

    {"id": "Q18", "text": "Elenca le condizioni cliniche secondarie che potrebbero influenzare il quadro del paziente.",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q19", "text": "Analizza la complessità clinica del paziente basandoti sulle comorbidità e confronta con il database.",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q20", "text": "Quali pattern diagnostici secondari emergono dalle cartelle cliniche?",
     "category": "comorbidity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    # ─── PHARMACOLOGY (Q21-Q30): consigli farmacologici ───
    {"id": "Q21", "text": "Quali consigli farmacologici daresti per questo paziente?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q22", "text": "Qual è la terapia farmacologica ottimale basata sulla storia clinica?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q23", "text": "Consiglia dei farmaci appropriati per trattare le patologie di questo paziente.",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q24", "text": "Quali sono le raccomandazioni terapeutiche per questo specifico quadro clinico?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q25", "text": "Fornisci suggerimenti sull'uso dei farmaci per questo paziente.",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions"]},

    {"id": "Q26", "text": "Quale regime farmacologico sarebbe più indicato per questo caso?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q27", "text": "Ci sono interazioni farmacologiche da evitare per questo paziente?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions"]},

    {"id": "Q28", "text": "Indica una possibile strategia farmacologica per migliorare l'esito clinico.",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "outcome"]},

    {"id": "Q29", "text": "Quali farmaci dovrebbero essere considerati prioritari per questo paziente?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": False,
     "gt_has_attrs": ["prescriptions"]},

    {"id": "Q30", "text": "Sulla base delle diagnosi, quali sono i principali consigli terapeutici?",
     "category": "pharmacology",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": False,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    # ─── SIMILARITY / CONFRONTI (Q31-Q40): richiedono similarity = True ───
    {"id": "Q31", "text": "Confronta questo paziente con casi simili nel database per identificare pattern comuni.",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q32", "text": "Come si posiziona questo paziente rispetto a soggetti con diagnosi analoghe?",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q33", "text": "Analizza le differenze e le somiglianze tra questo paziente e i suoi simili.",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q34", "text": "Quali sono i risultati clinici tipici per pazienti simili a questo?",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["outcome", "diagnoses"]},

    {"id": "Q35", "text": "Confronta il decorso di questo paziente con quello di altri casi comparabili.",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["outcome"]},

    {"id": "Q36", "text": "Esistono pazienti nel database con un quadro clinico quasi identico?",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q37", "text": "Usa i dati dei pazienti simili per valutare la tipicità di questo caso.",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q38", "text": "In che modo questo paziente si discosta dai pattern standard di pazienti simili?",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    {"id": "Q39", "text": "Quali strategie terapeutiche hanno funzionato meglio in pazienti simili a questo?",
     "category": "similarity",
     "gt_topics": ["pattern_analysis", "consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q40", "text": "Esegui un'analisi comparativa tra questo paziente e la popolazione simile nel database.",
     "category": "similarity",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses"]},

    # ─── MIXED (Q41-Q50): combinazioni di categorie ───
    {"id": "Q41", "text": "Riassumi le terapie e le comorbidità di questo paziente.",
     "category": "mixed",
     "gt_topics": ["pattern_analysis", "consulto_farmacologico"], "gt_similarity": False,
     "gt_has_attrs": ["diagnoses", "prescriptions"]},

    {"id": "Q42", "text": "Cosa suggeriscono i dati dei pazienti simili riguardo al trattamento di questo caso?",
     "category": "mixed",
     "gt_topics": ["consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions"]},

    {"id": "Q43", "text": "Estrai info su farmaci e diagnosi e confrontale con la media del database.",
     "category": "mixed",
     "gt_topics": ["pattern_analysis", "consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["prescriptions", "diagnoses"]},

    {"id": "Q44", "text": "Quali sono i punti critici del quadro clinico e farmacologico di questo paziente?",
     "category": "mixed",
     "gt_topics": ["pattern_analysis", "consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses", "prescriptions"]},

    {"id": "Q45", "text": "Fornisci un riassunto approfondito di tutto ciò che sai sul paziente.",
     "category": "mixed",
     "gt_topics": ["pattern_analysis", "consulto_farmacologico"], "gt_similarity": False,
     "gt_has_attrs": ["demographics", "diagnoses", "vitals", "prescriptions"]},

    {"id": "Q46", "text": "Analizza le patologie del paziente e consiglia una terapia basata su casi analoghi.",
     "category": "mixed",
     "gt_topics": ["pattern_analysis", "consulto_farmacologico"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses", "prescriptions"]},

    {"id": "Q47", "text": "Quali sono i dati salienti del paziente e quali pattern di rischio emergono dal database?",
     "category": "mixed",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["demographics", "diagnoses", "outcome"]},

    {"id": "Q48", "text": "Analizza i parametri di laboratorio e i segni vitali critici di questo paziente confrontandoli con pazienti simili",
     "category": "mixed",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["vitals", "labs"]},

    {"id": "Q49", "text": "Confronta il profilo di rischio di questo paziente con casi clinici affini nel database.",
     "category": "mixed",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["diagnoses", "outcome"]},

    {"id": "Q50", "text": "Quali sono le informazioni chiave da considerare per la gestione complessiva di questo paziente?",
     "category": "mixed",
     "gt_topics": ["pattern_analysis"], "gt_similarity": True,
     "gt_has_attrs": ["demographics", "admission_details", "diagnoses", "outcome"]},
]

# ══════════════════════════════════════════════════════════════════════════════
#  PATIENTS TO SKIP — manually add already processed hadm_ids
#  Esempio: SKIP_HADM_IDS = [24742431, 22866548]
# ══════════════════════════════════════════════════════════════════════════════

SKIP_HADM_IDS: list[int] = [
]

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_ragas_record(
    result: dict,
    question_meta: dict,
) -> dict:
    """
    Extracts required fields for RAGAS from a pipeline result.

    Campi prodotti:
      - user_input         : original question
      - response           : synthesized final response
      - retrieved_contexts : list of contexts used (similar patients + target)
      - reference          : None (no ground truth disponibile → LLM-as-judge)
      - metadata           : additional info (hadm_id, question_id, timing, ...)
    """
    patient_id = result.get("patient_id")
    question   = result.get("question", "")
    routing    = result.get("routing", {})
    responses  = result.get("responses", {})
    final_report = result.get("final_report", "")

    # ── Final response (prefer synthesized report) ──
    if final_report and len(final_report.strip()) > 20:
        answer = final_report.strip()
    elif responses:
        # Concatenate specialist responses
        answer = "\n\n---\n\n".join(responses.values()).strip()
    else:
        answer = ""

    # ── Retrieved contexts (for faithfulness / context precision) ──
    contexts = []

    # Retrieve contexts from routing (if context was saved)
    raw_context = result.get("_context", {})
    target_patient = raw_context.get("target_patient", {})
    if target_patient:
        ctx_str = f"[TARGET PATIENT hadm_id={patient_id}] " + json.dumps(
            target_patient, ensure_ascii=False, default=str
        )
        contexts.append(ctx_str)

    for i, sim_pat in enumerate(raw_context.get("similar_patients", []), 1):
        ctx_str = (
            f"[SIMILAR PATIENT #{i} hadm_id={sim_pat.get('hadm_id', '?')} "
            f"score={sim_pat.get('similarity_score', '?')}] "
            + json.dumps(sim_pat, ensure_ascii=False, default=str)
        )
        contexts.append(ctx_str)

    # Diagnosis glossary as additional context
    glossary = raw_context.get("diagnosis_glossary", {})
    if glossary:
        contexts.append("[DIAGNOSIS GLOSSARY] " + json.dumps(glossary, ensure_ascii=False))

    # Extract and format specialist responses to use as 'reference' (ground truth for contradiction_rate)
    specialist_texts = []
    if responses:
        for role, text in responses.items():
            specialist_texts.append(f"[{role.upper()}]\n{text}")
    specialists_combined = "\n\n---\n\n".join(specialist_texts).strip()

    # ── Record RAGAS ──
    record = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": specialists_combined if specialists_combined else None,  # Used as base for ContradictionRate
        "metadata": {
            "hadm_id": patient_id,
            "question_id": question_meta["id"],
            "question_category": question_meta["category"],
            "topics_routed": routing.get("topics", []),
            "similarity_used": routing.get("similarity", False),
            "k_similar": routing.get("k_similar", 0),
            "n_contexts_retrieved": len(contexts),
            "timing_seconds": result.get("timing", {}).get("total_seconds", 0),
            "routing_reasoning": routing.get("reasoning", ""),
            "status": "success",
        },
    }
    return record


def save_results(
    ragas_records: list,
    raw_results: list,
    output_dir: Path,
    run_id: str,
    summary: dict,
):
    """Saves all output files in the run folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # ── Save raw results ──
    for raw in raw_results:
        hadm_id = raw.get("patient_id", "unknown")
        q_id    = raw.get("_question_id", "Q00")
        fname   = raw_dir / f"{hadm_id}_{q_id}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False, default=str)

    # ── Save RAGAS dataset (JSON) ──
    ragas_path = output_dir / "ragas_dataset.json"
    with open(ragas_path, "w", encoding="utf-8") as f:
        json.dump(ragas_records, f, indent=2, ensure_ascii=False, default=str)

    # ── Save RAGAS dataset (CSV) ──
    csv_path = output_dir / "ragas_dataset.csv"
    rows = []
    for rec in ragas_records:
        row = {
            "user_input": rec["user_input"],
            "response": rec["response"],
            "retrieved_contexts": " ||| ".join(rec["retrieved_contexts"]),
            "reference": rec.get("reference", ""),
        }
        row.update(rec.get("metadata", {}))
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── Save summary ──
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return ragas_path, csv_path, summary_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_generation(
    n_patients: int = 2,
    questions: list = None,
    seed: int = 42,
    no_similarity: bool = False,
    dry_run: bool = False,
    results_root: Path = None,
    delay_between_calls: float = 2.0,
):
    """
    Executes RAGAS dataset generation.

    Args:
        n_patients:            Number of random hadm_ids to sample
        questions:             List of questions (default: CLINICAL_QUESTIONS)
        seed:                  Seed for reproducibility
        no_similarity:         Disable similar patient search
        dry_run:               Only print combinations, do not execute pipeline
        results_root:          Root path for results (default: ./results)
        delay_between_calls:   Seconds to wait between consecutive API calls
    """
    random.seed(seed)
    questions = questions or CLINICAL_QUESTIONS
    results_root = results_root or (_PROJECT_ROOT / "results")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_root / f"run_{run_id}"

    # ── Create output directory immediately so it is visible from start ──
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)

    console.rule(f"[bold cyan]RAGAS Dataset Generator — Run {run_id}[/bold cyan]")
    console.print(f"  📁 Output → {output_dir}")
    console.print(f"  👥 Pazienti: {n_patients} | 🎯 Domande: {len(questions)}")
    console.print(f"  🔢 Seed: {seed} | 🔗 Similarity: {'OFF' if no_similarity else 'ON'}")
    console.print()

    # ── Settings ──
    settings = get_settings()
    if not settings.llm.api_key:
        console.print("[bold red]❌ GEMINI_API_KEY is not configured![/bold red]")
        sys.exit(1)

    # ── Initialize pipeline ──
    pipeline = AgenticPipeline(settings)
    try:
        pipeline.initialize(skip_milvus=no_similarity)
    except Exception as e:
        console.print(f"[bold red]Pipeline initialization error: {e}[/bold red]")
        sys.exit(1)

    # ── Select random hadm_ids from real dataset ──
    all_hadm_ids = list(pipeline.data_gatherer._hadm_to_idx.keys())
    total_patients_needed = n_patients * len(questions)

    if len(all_hadm_ids) < total_patients_needed:
        console.print(
            f"[yellow]⚠ Solo {len(all_hadm_ids)} patients available in database "
            f"(requested {total_patients_needed} unici). Some patients will be reused.[/yellow]"
        )
        selected_hadm_ids = random.choices(all_hadm_ids, k=total_patients_needed)
    else:
        selected_hadm_ids = random.sample(all_hadm_ids, total_patients_needed)

    console.print(f"[green]✓ Selected {len(set(selected_hadm_ids))} unique hadm_ids out of {total_patients_needed} requested[/green]")
    console.print()

    # Create (hadm_id, question) pairs
    pairs = []
    for q_idx, q in enumerate(questions):
        start_idx = q_idx * n_patients
        end_idx = start_idx + n_patients
        q_patients = selected_hadm_ids[start_idx:end_idx]
        for p in q_patients:
            pairs.append((p, q))

    if dry_run:
        console.rule("[yellow]DRY RUN — no API calls executed[/yellow]")
        console.print(f"Total combinations: {len(pairs)}")
        for hadm_id, q in pairs:
            console.print(f"  [{hadm_id}] {q['id']}: {q['text'][:70]}...")
        pipeline.close()
        return

    # ── Main loop ──
    total_combinations = len(pairs)
    ragas_records: list = []
    raw_results: list   = []
    errors: list        = []
    processed           = 0
    skipped             = 0

    _skip_set = set(SKIP_HADM_IDS)
    if _skip_set:
        console.print(
            f"[yellow]⏭  Patients to skip ({len(_skip_set)}): {sorted(_skip_set)}[/yellow]"
        )

    console.rule("[bold green]STARTING GENERATION[/bold green]")

    for idx, (hadm_id, question_meta) in enumerate(pairs, 1):
        # ── Salta pazienti in SKIP_HADM_IDS ──
        if hadm_id in _skip_set:
            console.print(
                f"\n[dim]⏭  Combination {idx}/{total_combinations} "
                f"(hadm_id={hadm_id}) — SKIPPED (in SKIP_HADM_IDS)[/dim]"
            )
            continue

        console.print(
            f"\n[bold cyan]━━━ Combination {idx}/{total_combinations} "
            f"(hadm_id={hadm_id} | Domanda={question_meta['id']}) ━━━[/bold cyan]"
        )
        console.print(
            f"  Domanda: {question_meta['text'][:70]}..."
        )

        try:
            # Execute pipeline
            t0 = time.perf_counter()
            result = pipeline.run(
                question=question_meta["text"],
                patient_id=hadm_id,
                force_no_similarity=no_similarity,
                save_context=False,
            )
            elapsed = time.perf_counter() - t0

            # Inject context into result for RAGAS extraction
            # (reconstruct context using data_gatherer)
            try:
                routing = result.get("routing", {})
                context_obj = pipeline.context_constructor.build_context(
                    routing=routing,
                    data_gatherer=pipeline.data_gatherer,
                    similarity_engine=pipeline.similarity_engine if not no_similarity else None,
                )
                result["_context"] = context_obj
            except Exception as ctx_err:
                logger.warning(f"Unable to re-extract context: {ctx_err}")
                result["_context"] = {}

            # Enrich result with metadata
            result["_question_id"]   = question_meta["id"]
            result["_question_meta"] = question_meta

            # Extract RAGAS record
            ragas_rec = extract_ragas_record(result, question_meta)
            ragas_records.append(ragas_rec)
            raw_results.append(result)
            processed += 1

            console.print(
                f"    [green]✓[/green] Completed in {elapsed:.1f}s | "
                f"Contexts: {len(ragas_rec['retrieved_contexts'])} | "
                f"Response: {len(ragas_rec['response'])} chars"
            )

        except Exception as e:
            skipped += 1
            err_info = {
                "hadm_id": hadm_id,
                "question_id": question_meta["id"],
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(err_info)
            console.print(f"    [red]✗ Errore: {e}[/red]")
            logger.error(f"Error for {hadm_id}/{question_meta['id']}: {e}")

        # Pause between API calls to avoid rate-limiting
        if idx < total_combinations:
            time.sleep(delay_between_calls)

        # Checkpoint every 5 combinations or at the end
        if idx % 5 == 0 or idx == total_combinations:
            console.print(
                f"\n[yellow]💾 Checkpoint after combination {idx}/{total_combinations}...[/yellow]"
            )
            _checkpoint_save(ragas_records, output_dir, run_id, idx)

    # ── Final summary ──
    pipeline.close()
    total_time = sum(
        r.get("timing", {}).get("total_seconds", 0)
        for r in raw_results
    )

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_patients_requested": n_patients,
            "n_questions": len(questions),
            "seed": seed,
            "no_similarity": no_similarity,
            "delay_between_calls": delay_between_calls,
        },
        "results": {
            "total_combinations": total_combinations,
            "successful": processed,
            "failed": skipped,
            "success_rate_pct": round(100 * processed / total_combinations, 1) if total_combinations > 0 else 0,
        },
        "timing": {
            "total_pipeline_time_seconds": round(total_time, 1),
            "avg_seconds_per_call": round(total_time / processed, 1) if processed > 0 else 0,
        },
        "hadm_ids_used": selected_hadm_ids,
        "questions_used": [{"id": q["id"], "text": q["text"], "category": q["category"]} for q in questions],
        "errors": errors,
    }

    # ── Save all files ──
    console.rule("[bold green]SALVATAGGIO RISULTATI[/bold green]")
    ragas_path, csv_path, summary_path = save_results(
        ragas_records, raw_results, output_dir, run_id, summary
    )

    # ── Final report ──
    console.rule("[bold cyan]GENERATION COMPLETED[/bold cyan]")
    console.print(f"  ✅ Successes:     {processed}/{total_combinations}")
    console.print(f"  ❌ Failures:      {skipped}/{total_combinations}")
    console.print(f"  ⏱  Total time: {total_time:.0f}s")
    console.print(f"  📊 Record RAGAS: {len(ragas_records)}")
    console.print()
    console.print(f"  📁 Output directory: [bold]{output_dir}[/bold]")
    console.print(f"  📄 RAGAS JSON:   {ragas_path}")
    console.print(f"  📋 RAGAS CSV:    {csv_path}")
    console.print(f"  📝 Run summary:  {summary_path}")

    return ragas_records, summary


def _checkpoint_save(ragas_records: list, output_dir: Path, run_id: str, n_patients_done: int):
    """Saving intermediate checkpoint to prevent data loss on crash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_{n_patients_done}patients.json"
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ragas_records, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Checkpoint saved: {ckpt_path} ({len(ragas_records)} record)")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🏥 RAGAS Dataset Generator — Agentic Clinical Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Esempi:
  python generate_ragas_dataset.py
  python generate_ragas_dataset.py --n-patients 20 --seed 42
  python generate_ragas_dataset.py --n-patients 5 --no-similarity
  python generate_ragas_dataset.py --n-patients 3 --dry-run
  python generate_ragas_dataset.py --n-patients 20 --questions Q01 Q03 Q05
        """,
    )

    parser.add_argument(
        "--n-patients",
        type=int,
        default=2,
        help="Number of random hadm_ids to sample per ciascuna domanda (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility of random selection (default: 42)",
    )
    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Disable similar patient search (faster, no Milvus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combinations without running the pipeline",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        metavar="Q_ID",
        help="Question IDs to use (e.g. Q01 Q03 Q05). Default: all",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Pause seconds between consecutive API calls (default: 2.0)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Root directory for results (default: ./results)",
    )

    args = parser.parse_args()

    # Filter questions if specified
    questions = CLINICAL_QUESTIONS
    if args.questions:
        questions = [q for q in CLINICAL_QUESTIONS if q["id"] in args.questions]
        if not questions:
            console.print(
                f"[red]No questions found for IDs: {args.questions}[/red]\n"
                f"Available: {[q['id'] for q in CLINICAL_QUESTIONS]}"
            )
            sys.exit(1)
        console.print(f"[cyan]Selected questions: {[q['id'] for q in questions]}[/cyan]")

    results_root = Path(args.results_dir) if args.results_dir else _PROJECT_ROOT / "results"

    run_generation(
        n_patients=args.n_patients,
        questions=questions,
        seed=args.seed,
        no_similarity=args.no_similarity,
        dry_run=args.dry_run,
        results_root=results_root,
        delay_between_calls=args.delay,
    )


if __name__ == "__main__":
    main()
