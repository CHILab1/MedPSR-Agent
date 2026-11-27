"""
evaluate_retrieval_and_routing.py
-------------------------------
Benchmark evaluation script for Orchestrator routing, multi-threshold Jaccard
sensitivity, and IDF-weighted FastText vs. TF-IDF retrieval comparison.

Reproduces the routing and retrieval benchmark datasets consumed by
statistical_analysis.py without embedded notebook outputs.

Usage:
    python utils/evaluate_retrieval_and_routing.py --mode fasttext-vs-tfidf
    python utils/evaluate_retrieval_and_routing.py --mode thresholds
    python utils/evaluate_retrieval_and_routing.py --mode integrated --n-patients 50 --seed 42
    python utils/evaluate_retrieval_and_routing.py --mode all
"""

import argparse
import ast
from collections import Counter
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent

# Prevent utils/logging.py from shadowing stdlib logging when invoked directly
sys.path = [p for p in sys.path if Path(p or ".").resolve() != UTILS_DIR]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.generate_ragas_query_dataset import CLINICAL_QUESTIONS

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def _parse_diag_codes(val: Any) -> List[str]:
    raw_set: Set[Any] = set()
    if isinstance(val, (list, np.ndarray)):
        raw_set = set(val)
    elif isinstance(val, str):
        try:
            raw_set = set(ast.literal_eval(val))
        except Exception:
            raw_set = set(val.split(","))
    return [str(c).strip() for c in raw_set if str(c).strip() not in ("nan", "None", "")]


def _load_diagnosis_gt(parquet_path: Path) -> Tuple[np.ndarray, Dict[int, Set[str]], List[str]]:
    df_diag = pd.read_parquet(parquet_path, columns=["hadm_id", "diagnosis_categories_list"])
    all_hadm_ids = df_diag["hadm_id"].to_numpy()
    diag_gt_3char: Dict[int, Set[str]] = {}
    diag_corpus_4char: List[str] = []

    for hid_val, val in zip(df_diag["hadm_id"].values, df_diag["diagnosis_categories_list"].values):
        hid = int(hid_val)
        codes_4char = _parse_diag_codes(val)
        diag_gt_3char[hid] = {c[:3] for c in codes_4char if c}
        diag_corpus_4char.append(" ".join(codes_4char))

    return all_hadm_ids, diag_gt_3char, diag_corpus_4char


def _compute_ndcg(j_list: List[float]) -> float:
    if not j_list:
        return 0.0
    dcg = sum(v / np.log2(r + 1) for r, v in enumerate(j_list, start=1))
    ideal = sorted(j_list, reverse=True)
    idcg = sum(v / np.log2(r + 1) for r, v in enumerate(ideal, start=1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def run_integrated_eval(
    checkpoint_dir: Path,
    output_csv: Path,
    n_patients: int = 50,
    seed: int = 42,
    jaccard_threshold: float = 0.2,
) -> None:
    """Run Orchestrator routing + SimilarityEngine evaluation over sampled admissions."""
    from agents.orchestrator import OrchestratorAgent
    from agents.similarity_engine import SimilarityEngine
    from config.settings import get_settings

    parquet_path = checkpoint_dir / "combined_data_slim.parquet"
    all_hadm_ids, diag_gt_3char, _ = _load_diagnosis_gt(parquet_path)
    hadm_to_idx = {int(hid): i for i, hid in enumerate(all_hadm_ids)}

    rng = np.random.default_rng(seed)
    sample_patients = rng.choice(all_hadm_ids, size=n_patients, replace=False).tolist()
    rng_q = np.random.default_rng(seed + 1)
    q_perm = rng_q.permutation(len(CLINICAL_QUESTIONS)).tolist()
    assigned_questions = [CLINICAL_QUESTIONS[q_perm[i % len(CLINICAL_QUESTIONS)]] for i in range(n_patients)]

    settings = get_settings()
    settings.checkpoint_path = str(checkpoint_dir)
    orchestrator = OrchestratorAgent(settings)
    engine = SimilarityEngine(settings)
    engine.load(hadm_to_idx)

    rows = []
    try:
        for idx, (pat_id, q) in enumerate(zip(sample_patients, assigned_questions), start=1):
            pat_int = int(pat_id)
            t0 = time.perf_counter()
            routing = orchestrator.route(q["text"], pat_int)
            orch_time = round(time.perf_counter() - t0, 2)

            pred_topics = set(routing.get("topics", []))
            gt_topics = set(q["gt_topics"])
            topics_ok = gt_topics.issubset(pred_topics)

            pred_sim = bool(routing.get("similarity", False))
            gt_sim = bool(q["gt_similarity"])
            sim_ok = pred_sim == gt_sim

            pred_attrs = set(routing.get("attributes_list", []))
            gt_attrs = set(q["gt_has_attrs"])
            attrs_ok = gt_attrs.issubset(pred_attrs)

            orch_score = sum([topics_ok, sim_ok, attrs_ok]) / 3.0
            row: Dict[str, Any] = {
                "patient_idx": idx,
                "hadm_id": pat_int,
                "q_id": q["id"],
                "q_category": q["category"],
                "q_text": q["text"],
                "gt_similarity": gt_sim,
                "gt_topics": str(q["gt_topics"]),
                "gt_has_attrs": str(q["gt_has_attrs"]),
                "pred_topics": str(list(pred_topics)),
                "pred_similarity": pred_sim,
                "pred_k_similar": routing.get("k_similar", 10),
                "pred_weights": str(routing.get("weights_similarity", [0.4, 0.5, 0.1])),
                "pred_attrs": str(list(pred_attrs)),
                "pred_reasoning": routing.get("reasoning", ""),
                "topics_ok": topics_ok,
                "sim_ok": sim_ok,
                "attrs_ok": attrs_ok,
                "orch_score": orch_score,
                "orch_time_s": orch_time,
                "orch_status": "success",
            }

            if pred_sim:
                k_sim = int(routing.get("k_similar", 10) or 10)
                w_sim = routing.get("weights_similarity", [0.4, 0.5, 0.1])
                retrieved = engine.find_similar(pat_int, k=k_sim, weights=w_sim)
                target_diags = diag_gt_3char.get(pat_int, set())
                j_scores = []
                for rec_hadm, _ in retrieved:
                    rec_diags = diag_gt_3char.get(int(rec_hadm), set())
                    inter = len(target_diags & rec_diags)
                    union = len(target_diags | rec_diags)
                    j_scores.append(inter / union if union > 0 else 0.0)

                hits = [j for j in j_scores if j >= jaccard_threshold]
                mrr = next((1.0 / r for r, j in enumerate(j_scores, start=1) if j >= jaccard_threshold), 0.0)
                row.update({
                    "sim_performed": True,
                    "sim_hit": len(hits) > 0,
                    "sim_mrr": mrr,
                    "sim_best_jaccard": max(j_scores) if j_scores else 0.0,
                    "sim_avg_jaccard": float(np.mean(j_scores)) if j_scores else 0.0,
                    "sim_precision": len(hits) / k_sim if k_sim > 0 else 0.0,
                    "sim_ndcg": _compute_ndcg(j_scores),
                    "sim_k_used": k_sim,
                    "sim_weights_used": str(w_sim),
                    "sim_status": "success",
                })
            else:
                row.update({
                    "sim_performed": False,
                    "sim_hit": None,
                    "sim_mrr": None,
                    "sim_best_jaccard": None,
                    "sim_avg_jaccard": None,
                    "sim_precision": None,
                    "sim_ndcg": None,
                    "sim_k_used": None,
                    "sim_weights_used": None,
                    "sim_status": "not requested",
                })
            rows.append(row)
    finally:
        engine.close()

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved integrated evaluation CSV: {output_csv}")


def run_threshold_sensitivity(
    checkpoint_dir: Path,
    eval_csv: Path,
    output_json: Path,
    pool_size: int = 200,
) -> None:
    """Compute exact-cosine multi-threshold Jaccard sensitivity on active similarity queries."""
    parquet_path = checkpoint_dir / "combined_data_slim.parquet"
    all_hadm_ids, diag_gt_3char, _ = _load_diagnosis_gt(parquet_path)
    hadm_to_idx = {int(hid): i for i, hid in enumerate(all_hadm_ids)}
    idx_to_hadm = {i: int(hid) for i, hid in enumerate(all_hadm_ids)}

    emb_diag = np.load(checkpoint_dir / "patient_embeddings_diagnosis_4digit.npy").astype(np.float32)
    diag_norms = np.linalg.norm(emb_diag, axis=1, keepdims=True)
    diag_norms[diag_norms == 0] = 1.0
    emb_diag = emb_diag / diag_norms

    emb_num = np.load(checkpoint_dir / "patient_embeddings_numeric.npy", mmap_mode="r")
    emb_cat = np.load(checkpoint_dir / "patient_embeddings_categorical.npy", mmap_mode="r")

    def _cos_vec(x_row: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
        dot = np.dot(x_row, y_mat.T)[0]
        nx = np.linalg.norm(x_row) or 1e-9
        ny = np.linalg.norm(y_mat, axis=1)
        ny[ny == 0] = 1e-9
        return dot / (nx * ny)

    df_eval = pd.read_csv(eval_csv)
    sim_cases = df_eval[df_eval["sim_status"] == "success"].copy()

    cases_out = []
    for case_idx, (_, row) in enumerate(sim_cases.iterrows(), start=1):
        pat_int = int(row["hadm_id"])
        k_sim = int(row["sim_k_used"])
        w_sim = ast.literal_eval(row["sim_weights_used"]) if isinstance(row["sim_weights_used"], str) else row["sim_weights_used"]
        target_idx = hadm_to_idx[pat_int]

        dots = np.dot(emb_diag, emb_diag[target_idx])
        dots[target_idx] = -999.0
        top_cands = np.argpartition(dots, -pool_size)[-pool_size:]
        top_cands = top_cands[np.argsort(-dots[top_cands])].tolist()

        sim_num = _cos_vec(emb_num[target_idx].reshape(1, -1), emb_num[top_cands])
        sim_cat = _cos_vec(emb_cat[target_idx].reshape(1, -1), emb_cat[top_cands])

        scored = [
            (cand_idx, float(dots[cand_idx]) * w_sim[0] + float(sim_num[i]) * w_sim[1] + float(sim_cat[i]) * w_sim[2])
            for i, cand_idx in enumerate(top_cands)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = [idx_to_hadm[c_idx] for c_idx, _ in scored[:k_sim]]

        target_diags = diag_gt_3char.get(pat_int, set())
        j_scores = []
        for rec_hadm in top_k:
            rec_diags = diag_gt_3char.get(rec_hadm, set())
            inter = len(target_diags & rec_diags)
            union = len(target_diags | rec_diags)
            j_scores.append(inter / union if union > 0 else 0.0)

        entry: Dict[str, Any] = {
            "case_id": case_idx,
            "hadm_id": pat_int,
            "q_id": str(row.get("q_id", "")),
            "k_used": k_sim,
            "weights_used": w_sim,
            "raw_jaccard_scores": [float(x) for x in j_scores],
        }
        for th in THRESHOLDS:
            entry[f"hit_{th}"] = int(any(j >= th for j in j_scores))
            entry[f"prec_{th}"] = round(float(sum(1 for j in j_scores if j >= th) / k_sim) if k_sim > 0 else 0.0, 4)
        cases_out.append(entry)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"n_cases": len(cases_out), "thresholds": THRESHOLDS, "cases": cases_out}, f, indent=2)
    print(f"Saved Jaccard thresholds JSON: {output_json} ({len(cases_out)} cases)")


def _build_tfidf_matrix(diag_corpus_4char: List[str], max_features: int = 100) -> sp.csr_matrix:
    """Build L2-normalized 100D TF-IDF matrix (uses sklearn if installed, else sparse replica)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
        return vectorizer.fit_transform(diag_corpus_4char)
    except ImportError:
        token_pattern = re.compile(r"(?u)\b\w+\b")
        doc_tokens = [token_pattern.findall(doc) for doc in diag_corpus_4char]
        corpus_counts: Counter = Counter()
        for toks in doc_tokens:
            for tok in toks:
                corpus_counts[tok] += 1
        top_terms = sorted(corpus_counts.keys(), key=lambda t: (-corpus_counts[t], t))[:max_features]
        vocab = {term: idx for idx, term in enumerate(sorted(top_terms))}

        n_docs = len(doc_tokens)
        rows, cols, data = [], [], []
        df_counts = np.zeros(len(vocab), dtype=np.float64)
        for row_idx, toks in enumerate(doc_tokens):
            for tok, cnt in Counter(toks).items():
                if tok in vocab:
                    col_idx = vocab[tok]
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(cnt)
                    df_counts[col_idx] += 1

        x_tf = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, len(vocab)), dtype=np.float32)
        idf = np.log((1.0 + n_docs) / (1.0 + df_counts)) + 1.0
        x_tfidf = x_tf.multiply(idf).tocsr()
        row_norms = sp.linalg.norm(x_tfidf, axis=1)
        row_norms[row_norms == 0] = 1.0
        return sp.diags(1.0 / row_norms) @ x_tfidf


def run_fasttext_vs_tfidf(
    checkpoint_dir: Path,
    output_json: Path,
    n_patients: int = 100,
    k_eval: int = 10,
    seed: int = 42,
) -> None:
    """Compare IDF-weighted FastText vs. 100D TF-IDF baseline using exact cosine similarity."""
    parquet_path = checkpoint_dir / "combined_data_slim.parquet"
    all_hadm_ids, diag_gt_3char, diag_corpus_4char = _load_diagnosis_gt(parquet_path)
    hadm_to_idx = {int(hid): i for i, hid in enumerate(all_hadm_ids)}
    idx_to_hadm = {i: int(hid) for i, hid in enumerate(all_hadm_ids)}

    x_tfidf = _build_tfidf_matrix(diag_corpus_4char, max_features=100)

    emb_ft = np.load(checkpoint_dir / "patient_embeddings_diagnosis_4digit.npy").astype(np.float32)
    ft_norms = np.linalg.norm(emb_ft, axis=1, keepdims=True)
    ft_norms[ft_norms == 0] = 1.0
    emb_ft = emb_ft / ft_norms

    rng = np.random.default_rng(seed)
    sample_patients = rng.choice(all_hadm_ids, size=n_patients, replace=False).tolist()

    def _patient_metrics(j_list: List[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "best_jaccard": float(max(j_list)) if j_list else 0.0,
            "avg_jaccard": float(np.mean(j_list)) if j_list else 0.0,
            "ndcg": _compute_ndcg(j_list),
        }
        for th in THRESHOLDS:
            out[f"hit_{th}"] = int(any(j >= th for j in j_list))
            out[f"prec_{th}"] = float(sum(1 for j in j_list if j >= th) / k_eval)
        return out

    patients_out = []
    for pat_id in sample_patients:
        pat_int = int(pat_id)
        t_idx = hadm_to_idx[pat_int]
        target_diags = diag_gt_3char[pat_int]

        # FastText exact cosine
        ft_scores = np.dot(emb_ft, emb_ft[t_idx])
        ft_scores[t_idx] = -999.0
        ft_top = np.argpartition(ft_scores, -k_eval)[-k_eval:]
        ft_top = ft_top[np.argsort(-ft_scores[ft_top])]
        j_ft = [
            len(target_diags & diag_gt_3char.get(idx_to_hadm[int(i)], set()))
            / max(1, len(target_diags | diag_gt_3char.get(idx_to_hadm[int(i)], set())))
            for i in ft_top
        ]

        # TF-IDF exact cosine
        tf_scores = x_tfidf.dot(x_tfidf[t_idx].T).toarray().ravel()
        tf_scores[t_idx] = -1.0
        tf_top = np.argpartition(tf_scores, -k_eval)[-k_eval:]
        tf_top = tf_top[np.argsort(-tf_scores[tf_top])]
        j_tf = [
            len(target_diags & diag_gt_3char.get(idx_to_hadm[int(i)], set()))
            / max(1, len(target_diags | diag_gt_3char.get(idx_to_hadm[int(i)], set())))
            for i in tf_top
        ]

        patients_out.append({
            "hadm_id": pat_int,
            "fasttext": _patient_metrics(j_ft),
            "tfidf": _patient_metrics(j_tf),
            "raw_jaccard_fasttext": [float(x) for x in j_ft],
            "raw_jaccard_tfidf": [float(x) for x in j_tf],
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {"n_patients": n_patients, "k_eval": k_eval, "thresholds": THRESHOLDS, "patients": patients_out},
            f,
            indent=2,
        )
    print(f"Saved FastText vs TF-IDF JSON: {output_json} ({len(patients_out)} patients)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator & Retrieval Benchmark Evaluation")
    parser.add_argument(
        "--mode",
        choices=["integrated", "thresholds", "fasttext-vs-tfidf", "all"],
        default="fasttext-vs-tfidf",
        help="Which benchmark evaluation to run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pipeline_checkpoints",
        help="Path to MIMIC-IV preprocessed checkpoints",
    )
    args = parser.parse_args()

    if args.mode in ("integrated", "all"):
        run_integrated_eval(args.checkpoint_dir, UTILS_DIR / "results_orchestrator_eval.csv")
    if args.mode in ("thresholds", "all"):
        run_threshold_sensitivity(
            args.checkpoint_dir,
            UTILS_DIR / "results_orchestrator_eval.csv",
            UTILS_DIR / "results_jaccard_thresholds.json",
        )
    if args.mode in ("fasttext-vs-tfidf", "all"):
        run_fasttext_vs_tfidf(args.checkpoint_dir, UTILS_DIR / "results_fasttext_vs_tfidf.json")


if __name__ == "__main__":
    main()
