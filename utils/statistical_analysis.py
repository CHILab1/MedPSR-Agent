"""
statistical_analysis.py
-----------------------
Portable statistical analysis script for the Clinical Decision Support System
evaluation benchmark. Computes means, standard errors (SE), 95% percentile
bootstrap confidence intervals (10,000 resamples), paired Wilcoxon signed-rank
tests, exact McNemar tests, and Holm-Bonferroni multiple-testing adjustments.

Inputs (located in the same utils/ directory):
- results_full_pipeline_ragas.json
- results_ragas_ablation_and_monolithic.json
- results_orchestrator_eval.csv
- results_jaccard_thresholds.json
- results_fasttext_vs_tfidf.json

Outputs:
- results_statistical_report.md
- results_statistical_analysis.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

UTILS_DIR = Path(__file__).resolve().parent
# Prevent utils/logging.py from shadowing stdlib logging when invoked directly
sys.path = [p for p in sys.path if Path(p or ".").resolve() != UTILS_DIR]

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

FULL_RAGAS_JSON_PATH = UTILS_DIR / "results_full_pipeline_ragas.json"
ABLATION_JSON_PATH = UTILS_DIR / "results_ragas_ablation_and_monolithic.json"
ORCHESTRATOR_CSV_PATH = UTILS_DIR / "results_orchestrator_eval.csv"
THRESHOLDS_JSON_PATH = UTILS_DIR / "results_jaccard_thresholds.json"
FASTTEXT_TFIDF_JSON_PATH = UTILS_DIR / "results_fasttext_vs_tfidf.json"

REPORT_MD_PATH = UTILS_DIR / "results_statistical_report.md"
REPORT_JSON_PATH = UTILS_DIR / "results_statistical_analysis.json"


# ==============================================================================
#  CORE STATISTICAL FUNCTIONS
# ==============================================================================

def summary_stats(x: Any, n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Compute Mean, SD, SE, and 95% percentile bootstrap confidence interval."""
    x = np.asarray(x, dtype=float)
    x_valid = x[~np.isnan(x)]
    n = len(x_valid)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    mean_val = float(np.mean(x_valid))
    std_val = float(np.std(x_valid, ddof=1)) if n > 1 else 0.0
    se_val = std_val / np.sqrt(n) if n > 1 else 0.0

    if std_val == 0.0 or n == 1:
        ci_low, ci_high = mean_val, mean_val
    else:
        rng = np.random.default_rng(seed)
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = np.mean(x_valid[boot_idx], axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "se": round(se_val, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "n": int(n),
    }


def paired_wilcoxon(a: Any, b: Any) -> float:
    """Two-sided paired Wilcoxon signed-rank test (zero_method='wilcox')."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a_clean = a[mask]
    b_clean = b[mask]

    diff = a_clean - b_clean
    if np.all(diff == 0) or len(diff) == 0:
        return 1.0
    try:
        _, p_val = wilcoxon(a_clean, b_clean, alternative="two-sided", zero_method="wilcox")
        return float(p_val)
    except Exception:
        return 1.0


def paired_mcnemar_exact(a_bin: Any, b_bin: Any) -> Tuple[float, Dict[str, int]]:
    """Exact two-sided McNemar test for paired binary indicators."""
    a_bin = np.asarray(a_bin, dtype=int)
    b_bin = np.asarray(b_bin, dtype=int)

    both = int(np.sum((a_bin == 1) & (b_bin == 1)))
    a_only = int(np.sum((a_bin == 1) & (b_bin == 0)))
    b_only = int(np.sum((a_bin == 0) & (b_bin == 1)))
    neither = int(np.sum((a_bin == 0) & (b_bin == 0)))

    n_disc = a_only + b_only
    if n_disc == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(min(a_only, b_only), n_disc, 0.5, alternative="two-sided").pvalue)

    counts = {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "n_discordant": n_disc,
    }
    return p_val, counts


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Sequential Holm-Bonferroni adjustment for family-wise error rate control."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


# ==============================================================================
#  DATA LOADING
# ==============================================================================

def load_data() -> Dict[str, Any]:
    """Load all benchmark evaluation artifacts from utils/."""
    required_files = [
        FULL_RAGAS_JSON_PATH,
        ABLATION_JSON_PATH,
        ORCHESTRATOR_CSV_PATH,
        THRESHOLDS_JSON_PATH,
        FASTTEXT_TFIDF_JSON_PATH,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation file not found: {path}")

    with open(FULL_RAGAS_JSON_PATH, "r", encoding="utf-8") as f:
        full_ragas = json.load(f)
    with open(ABLATION_JSON_PATH, "r", encoding="utf-8") as f:
        ablation = json.load(f)
    with open(THRESHOLDS_JSON_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    with open(FASTTEXT_TFIDF_JSON_PATH, "r", encoding="utf-8") as f:
        fasttext_tfidf = json.load(f)

    orch_df = pd.read_csv(ORCHESTRATOR_CSV_PATH)
    sim_df = orch_df[orch_df["sim_status"] == "success"].copy()

    return {
        "full_ragas": full_ragas,
        "ablation": ablation,
        "orchestrator_df": orch_df,
        "similarity_df": sim_df,
        "thresholds": thresholds,
        "fasttext_tfidf": fasttext_tfidf,
    }


# ==============================================================================
#  TABLE ANALYSES
# ==============================================================================

def analyze_ablation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Paired statistical analysis for the 3-way ablation study (N=100)."""
    full_recs = data["full_ragas"]["full_pipeline"]["results"]
    mono_recs = data["ablation"]["gemma31b_monolithic"]["results"]
    abl_recs = data["ablation"]["pipeline_no_similarity"]["results"]

    full_map = {(int(r["hadm_id"]), str(r["question_id"])): r for r in full_recs}
    mono_map = {(int(r["hadm_id"]), str(r["question_id"])): r for r in mono_recs}
    abl_map = {(int(r["hadm_id"]), str(r["question_id"])): r for r in abl_recs}

    common_keys = sorted(
        set(full_map.keys()) & set(mono_map.keys()) & set(abl_map.keys()),
        key=lambda k: full_map[k].get("index", 999),
    )

    metrics = {
        "Answer Relevancy": {
            "Full": [full_map[k]["answer_relevancy"] for k in common_keys],
            "Monolithic": [mono_map[k]["answer_relevancy"] for k in common_keys],
            "Ablated": [abl_map[k]["answer_relevancy"] for k in common_keys],
        },
        "Clinical Validity": {
            "Full": [full_map[k]["clinical_validity"] for k in common_keys],
            "Monolithic": [mono_map[k]["clinical_validity"] for k in common_keys],
            "Ablated": [abl_map[k]["clinical_validity"] for k in common_keys],
        },
        "Context Precision": {
            "Full": [full_map[k]["context_precision"] for k in common_keys],
            "Monolithic": [mono_map[k]["context_precision"] for k in common_keys],
            "Ablated": [abl_map[k]["context_precision"] for k in common_keys],
        },
    }

    descriptive: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m_name, configs in metrics.items():
        descriptive[m_name] = {
            cfg_name: summary_stats(arr) for cfg_name, arr in configs.items()
        }

    comparisons = []
    raw_pvals = []
    for m_name, configs in metrics.items():
        p_mono = paired_wilcoxon(configs["Full"], configs["Monolithic"])
        comparisons.append({
            "metric": m_name,
            "comparison": "Full Pipeline vs Monolithic",
            "p_raw": p_mono,
        })
        raw_pvals.append(p_mono)

        p_abl = paired_wilcoxon(configs["Full"], configs["Ablated"])
        comparisons.append({
            "metric": m_name,
            "comparison": "Full Pipeline vs Ablated",
            "p_raw": p_abl,
        })
        raw_pvals.append(p_abl)

    adj_pvals = holm_bonferroni(raw_pvals)
    for comp, p_adj in zip(comparisons, adj_pvals):
        comp["p_adj_holm"] = p_adj
        comp["significant"] = bool(p_adj < 0.05)
        comp["verdict"] = "Statistically Significant (p < 0.05)" if p_adj < 0.05 else "Observed Trend (n.s.)"

    return {
        "n_cases": len(common_keys),
        "descriptive": descriptive,
        "comparisons": comparisons,
    }


def analyze_overall(data: Dict[str, Any], ablation_res: Dict[str, Any]) -> Dict[str, Any]:
    """Summary statistics (Mean, SE, 95% Bootstrap CI) for component-wise metrics."""
    df_orch = data["orchestrator_df"]
    df_sim = data["similarity_df"]
    full_recs = data["full_ragas"]["full_pipeline"]["results"]

    orch_scores = df_orch["orch_score"].dropna().tolist()
    hit_arr = [1 if bool(h) else 0 for h in df_sim["sim_hit"]]
    prec_arr = df_sim["sim_precision"].dropna().tolist()
    ndcg_arr = df_sim["sim_ndcg"].dropna().tolist()

    cv_arr = [r["clinical_validity"] for r in full_recs if r.get("clinical_validity") is not None]
    cr_arr = [r["contradiction_rate"] for r in full_recs if r.get("contradiction_rate") is not None]
    cp_arr = [r["context_precision"] for r in full_recs if r.get("context_precision") is not None]

    return {
        "Orchestrator Accuracy": {
            "component": "Orchestrator",
            "focus": "Router classification correctness",
            "stats": summary_stats(orch_scores),
            "n_cases": len(orch_scores),
        },
        "Hit-Rate @ K": {
            "component": "Similarity Engine",
            "focus": "Utility of retrieved cohort (Jaccard >= 0.2)",
            "stats": summary_stats(hit_arr),
            "n_cases": len(hit_arr),
        },
        "Precision @ K": {
            "component": "Similarity Engine",
            "focus": "Yield of retrieved cohort (Jaccard >= 0.2)",
            "stats": summary_stats(prec_arr),
            "n_cases": len(prec_arr),
        },
        "nDCG @ K": {
            "component": "Similarity Engine",
            "focus": "Ranking similarity cohort",
            "stats": summary_stats(ndcg_arr),
            "n_cases": len(ndcg_arr),
        },
        "Clinical Validity": {
            "component": "Specialist & Synthesizer",
            "focus": "Automated medical plausibility",
            "stats": summary_stats(cv_arr),
            "n_cases": len(cv_arr),
        },
        "Contradiction Rate": {
            "component": "Report Synthesizer",
            "focus": "Inner-agent factual consistency",
            "stats": summary_stats(cr_arr),
            "n_cases": len(cr_arr),
        },
        "Answer Relevancy": {
            "component": "End-to-End",
            "focus": "Alignment of response to query",
            "stats": ablation_res["descriptive"]["Answer Relevancy"]["Full"],
            "n_cases": ablation_res["n_cases"],
        },
        "Context Precision": {
            "component": "Retrieval-to-Context",
            "focus": "Ranking quality of context",
            "stats": summary_stats(cp_arr),
            "n_cases": len(cp_arr),
        },
    }


def analyze_thresholds(data: Dict[str, Any]) -> Dict[str, Any]:
    """Jaccard threshold sensitivity analysis (N=70)."""
    cases = data["thresholds"]["cases"]
    results = {}
    for th in [0.2, 0.3, 0.4, 0.5]:
        h_arr = [c[f"hit_{th}"] for c in cases]
        p_arr = [c[f"prec_{th}"] for c in cases]
        results[th] = {
            "hit_rate": summary_stats(h_arr),
            "precision": summary_stats(p_arr),
            "n_cases": len(cases),
        }
    return results


def analyze_baseline_comparison(data: Dict[str, Any]) -> Dict[str, Any]:
    """Paired retrieval benchmark: IDF-weighted FastText vs TF-IDF baseline (N=100)."""
    patients = data["fasttext_tfidf"]["patients"]
    results: Dict[str, Any] = {
        "n_patients": len(patients),
        "thresholds": {},
        "global_metrics": {},
        "tests": [],
    }

    raw_pvals = []
    test_labels = []

    for th in [0.2, 0.3, 0.4, 0.5]:
        ft_hits = np.array([p["fasttext"][f"hit_{th}"] for p in patients])
        tf_hits = np.array([p["tfidf"][f"hit_{th}"] for p in patients])
        ft_prec = np.array([p["fasttext"][f"prec_{th}"] for p in patients])
        tf_prec = np.array([p["tfidf"][f"prec_{th}"] for p in patients])

        p_mcnemar, mcn_counts = paired_mcnemar_exact(ft_hits, tf_hits)
        raw_pvals.append(p_mcnemar)
        test_labels.append(f"Hit-Rate@10 (tau={th})")

        p_wilc = paired_wilcoxon(ft_prec, tf_prec)
        raw_pvals.append(p_wilc)
        test_labels.append(f"Precision@10 (tau={th})")

        results["thresholds"][th] = {
            "hit_rate": {
                "FastText": summary_stats(ft_hits),
                "TF-IDF": summary_stats(tf_hits),
                "p_mcnemar": p_mcnemar,
                "mcnemar_contingency": mcn_counts,
            },
            "precision": {
                "FastText": summary_stats(ft_prec),
                "TF-IDF": summary_stats(tf_prec),
                "p_wilcoxon": p_wilc,
            },
        }

    ft_ndcg = np.array([p["fasttext"]["ndcg"] for p in patients])
    tf_ndcg = np.array([p["tfidf"]["ndcg"] for p in patients])
    p_ndcg = paired_wilcoxon(ft_ndcg, tf_ndcg)
    raw_pvals.append(p_ndcg)
    test_labels.append("nDCG@10")

    ft_jacc = np.array([p["fasttext"]["best_jaccard"] for p in patients])
    tf_jacc = np.array([p["tfidf"]["best_jaccard"] for p in patients])
    p_jacc = paired_wilcoxon(ft_jacc, tf_jacc)
    raw_pvals.append(p_jacc)
    test_labels.append("Mean Best Jaccard")

    results["global_metrics"]["nDCG@10"] = {
        "FastText": summary_stats(ft_ndcg),
        "TF-IDF": summary_stats(tf_ndcg),
        "p_wilcoxon": p_ndcg,
    }
    results["global_metrics"]["Mean Best Jaccard"] = {
        "FastText": summary_stats(ft_jacc),
        "TF-IDF": summary_stats(tf_jacc),
        "p_wilcoxon": p_jacc,
    }

    adj_pvals = holm_bonferroni(raw_pvals)
    for lbl, p_raw, p_adj in zip(test_labels, raw_pvals, adj_pvals):
        results["tests"].append({
            "comparison": lbl,
            "p_raw": round(p_raw, 5),
            "p_adj_holm": round(p_adj, 5),
            "significant": bool(p_adj < 0.05),
            "verdict": "Statistically Significant (p < 0.05)" if p_adj < 0.05 else "Observed Trend (n.s.)",
        })

    return results


# ==============================================================================
#  REPORT GENERATION
# ==============================================================================

def _fmt_p3(val: float) -> str:
    return "<0.001" if val < 0.001 else f"{val:.3f}"


def generate_latex_tables(
    t1_res: Dict[str, Any],
    t2_res: Dict[str, Any],
    t3_res: Dict[str, Any],
    t4_res: Dict[str, Any],
) -> str:
    """Generate publication-ready LaTeX tables matching the paper layout."""
    latex_out = []

    # Table 1: Overall Component-Wise Metrics
    t1_lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Overall Framework and Component-Wise Metrics. Note: End-to-end and routing metrics are calculated over the full benchmark ($N=100$, 50 clinical queries evaluated across 2 distinct patients), while Similarity Engine metrics are calculated over the subset of queries where retrieval was dynamically activated ($N=70$), evaluated at the query-specific $K$ assigned by the Orchestrator. Reported as Score $\pm$ SE [95\% Bootstrap CI].}",
        r"\label{tab:metrics_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lllc}",
        r"\toprule",
        r"\textbf{Component} & \textbf{Metric} & \textbf{Evaluation Focus} & \makecell{\textbf{Score $\pm$ SE} \\ \textbf{[95\% CI]}} \\",
        r"\midrule",
    ]
    rows_t1 = [
        ("Orchestrator", "Orchestrator Accuracy", "Router classification correctness", t1_res["Orchestrator Accuracy"]["stats"]),
        ("Similarity Engine", "Hit-Rate @ $K$", "Utility of retrieved cohort (Jaccard $\\ge 0.2$)", t1_res["Hit-Rate @ K"]["stats"]),
        ("Similarity Engine", "Precision @ $K$", "Yield of retrieved cohort (Jaccard $\\ge 0.2$)", t1_res["Precision @ K"]["stats"]),
        ("Similarity Engine", "nDCG @ $K$", "Ranking similarity cohort", t1_res["nDCG @ K"]["stats"]),
        ("Specialist \\& Synthesizer", "Clinical Validity", "Automated medical plausibility", t1_res["Clinical Validity"]["stats"]),
        ("Report Synthesizer", "Contradiction Rate", "Inner-agent factual consistency", t1_res["Contradiction Rate"]["stats"]),
        ("End-to-End", "Answer Relevancy", "Alignment of response to query", t1_res["Answer Relevancy"]["stats"]),
        ("Retrieval-to-Context", "Context Precision", "Ranking quality of context", t1_res["Context Precision"]["stats"]),
    ]
    for i, (comp, metric, focus, s) in enumerate(rows_t1):
        t1_lines.append(
            f"{comp} & {metric} & {focus} & \\makecell{{\\textbf{{{s['mean']:.3f}}} $\\pm {s['se']:.3f}$ \\\\ {{[{s['ci_low']:.3f}, {s['ci_high']:.3f}]}}}} \\\\"
        )
        if i < len(rows_t1) - 1:
            t1_lines.append(r"\addlinespace")
    t1_lines.extend([r"\botrule", r"\end{tabular*}", r"\end{table}"])
    latex_out.append("\n".join(t1_lines))

    # Table 2: Sensitivity to Jaccard Thresholds
    hr_cells = " & ".join(
        f"\\makecell{{\\textbf{{{t2_res[th]['hit_rate']['mean']:.3f}}} $\\pm {t2_res[th]['hit_rate']['se']:.3f}$ \\\\ {{[{t2_res[th]['hit_rate']['ci_low']:.2f}, {t2_res[th]['hit_rate']['ci_high']:.2f}]}}}}"
        for th in [0.2, 0.3, 0.4, 0.5]
    )
    pr_cells = " & ".join(
        f"\\makecell{{\\textbf{{{t2_res[th]['precision']['mean']:.3f}}} $\\pm {t2_res[th]['precision']['se']:.3f}$ \\\\ {{[{t2_res[th]['precision']['ci_low']:.2f}, {t2_res[th]['precision']['ci_high']:.2f}]}}}}"
        for th in [0.2, 0.3, 0.4, 0.5]
    )
    t2_latex = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Retrieval Performance Under Varying Jaccard Similarity Thresholds ($N=70$, evaluated at the dynamically assigned $K$, reported as Mean $\pm$ SE [95\% Bootstrap CI]).}",
        r"\label{tab:thresholds}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}",
        r"\toprule",
        r"\textbf{Metric / Jaccard Threshold} & \textbf{0.2} & \textbf{0.3} & \textbf{0.4} & \textbf{0.5} \\",
        r"\midrule",
        f"Hit-Rate @ $K$ & {hr_cells} \\\\",
        r"\addlinespace",
        f"Precision @ $K$ & {pr_cells} \\\\",
        r"\botrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    latex_out.append(t2_latex)

    # Table 3: Comparative Retrieval Performance
    t3_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparative Retrieval Performance on the 100 benchmark patients: FastText+IDF vs. TF-IDF Baseline across Multiple Jaccard Thresholds (Mean $\pm$ SE [95\% Bootstrap CI], with exact Paired McNemar tests for binary Hit-Rate and Paired Wilcoxon signed-rank tests for Precision, nDCG, and Best Jaccard; $p_{\text{adj}}$ corrected via Holm--Bonferroni).}",
        r"\label{tab:baseline_comparison}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}clccc}",
        r"\toprule",
        r"\textbf{Threshold} & \textbf{Method} & \textbf{Hit-Rate@10} & \textbf{Precision@10} & \makecell{\textbf{Significance} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} \\",
        r"\midrule",
    ]
    for th in [0.2, 0.3, 0.4, 0.5]:
        ft_h = t3_res["thresholds"][th]["hit_rate"]["FastText"]
        tf_h = t3_res["thresholds"][th]["hit_rate"]["TF-IDF"]
        ft_p = t3_res["thresholds"][th]["precision"]["FastText"]
        tf_p = t3_res["thresholds"][th]["precision"]["TF-IDF"]
        t_h = next(t for t in t3_res["tests"] if t["comparison"] == f"Hit-Rate@10 (tau={th})")
        t_p = next(t for t in t3_res["tests"] if t["comparison"] == f"Precision@10 (tau={th})")

        prec_ft_mean = f"{ft_p['mean']:.3f}" if th == 0.2 else f"\\textbf{{{ft_p['mean']:.3f}}}"
        prec_tf_mean = f"\\textbf{{{tf_p['mean']:.3f}}}" if th == 0.2 else f"{tf_p['mean']:.3f}"

        t3_lines.append(
            f"\\multirow{{3}}{{*}}{{{th}}} & \\textbf{{FastText+IDF}} & "
            f"\\makecell{{\\textbf{{{ft_h['mean']:.3f}}} $\\pm {ft_h['se']:.3f}$ \\\\ {{[{ft_h['ci_low']:.2f}, {ft_h['ci_high']:.2f}]}}}} & "
            f"\\makecell{{{prec_ft_mean} $\\pm {ft_p['se']:.3f}$ \\\\ {{[{ft_p['ci_low']:.2f}, {ft_p['ci_high']:.2f}]}}}} & "
            f"\\multirow{{3}}{{*}}{{\\makecell[l]{{Hit: $p={t_h['p_raw']:.3f} \\,/\\, {t_h['p_adj_holm']:.3f}$ \\\\ Prec: $p={t_p['p_raw']:.3f} \\,/\\, {t_p['p_adj_holm']:.3f}$}}}} \\\\"
        )
        t3_lines.append(
            f" & TF-IDF & "
            f"\\makecell{{{tf_h['mean']:.3f} $\\pm {tf_h['se']:.3f}$ \\\\ {{[{tf_h['ci_low']:.2f}, {tf_h['ci_high']:.2f}]}}}} & "
            f"\\makecell{{{prec_tf_mean} $\\pm {tf_p['se']:.3f}$ \\\\ {{[{tf_p['ci_low']:.2f}, {tf_p['ci_high']:.2f}]}}}} & \\\\"
        )
        t3_lines.append(r"\midrule")

    ndcg_ft = t3_res["global_metrics"]["nDCG@10"]["FastText"]
    ndcg_tf = t3_res["global_metrics"]["nDCG@10"]["TF-IDF"]
    t_ndcg = next(t for t in t3_res["tests"] if t["comparison"] == "nDCG@10")
    jacc_ft = t3_res["global_metrics"]["Mean Best Jaccard"]["FastText"]
    jacc_tf = t3_res["global_metrics"]["Mean Best Jaccard"]["TF-IDF"]
    t_jacc = next(t for t in t3_res["tests"] if t["comparison"] == "Mean Best Jaccard")

    t3_lines.extend([
        r"\midrule",
        r"& & \textbf{nDCG@10} & \makecell{\textbf{Mean Best} \\ \textbf{Jaccard}} & \makecell{\textbf{Significance} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} \\",
        r"\midrule",
        (
            f"& \\textbf{{FastText+IDF}} & "
            f"\\makecell{{\\textbf{{{ndcg_ft['mean']:.3f}}} $\\pm {ndcg_ft['se']:.3f}$ \\\\ {{[{ndcg_ft['ci_low']:.2f}, {ndcg_ft['ci_high']:.2f}]}}}} & "
            f"\\makecell{{\\textbf{{{jacc_ft['mean']:.3f}}} $\\pm {jacc_ft['se']:.3f}$ \\\\ {{[{jacc_ft['ci_low']:.2f}, {jacc_ft['ci_high']:.2f}]}}}} & "
            f"\\multirow{{3}}{{*}}{{\\makecell[l]{{nDCG: $p={t_ndcg['p_raw']:.3f} \\,/\\, {t_ndcg['p_adj_holm']:.3f}$ \\\\ Jacc: $p={t_jacc['p_raw']:.3f} \\,/\\, {t_jacc['p_adj_holm']:.3f}$}}}} \\\\"
        ),
        (
            f"& TF-IDF & "
            f"\\makecell{{{ndcg_tf['mean']:.3f} $\\pm {ndcg_tf['se']:.3f}$ \\\\ {{[{ndcg_tf['ci_low']:.2f}, {ndcg_tf['ci_high']:.2f}]}}}} & "
            f"\\makecell{{{jacc_tf['mean']:.3f} $\\pm {jacc_tf['se']:.3f}$ \\\\ {{[{jacc_tf['ci_low']:.2f}, {jacc_tf['ci_high']:.2f}]}}}} & \\\\"
        ),
        r"\botrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    latex_out.append("\n".join(t3_lines))

    # Table 4: Ablation Study
    t4_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation Study Results: Comparison of the Full Pipeline against Architectural Baselines across the Benchmark Dataset ($N=100$, Reported as Mean $\pm$ SE [95\% Bootstrap CI]). Significance is reported as unadjusted $p_{\text{raw}} \,/\, \text{Holm--Bonferroni adjusted } p_{\text{adj}}$ using paired Wilcoxon signed-rank tests.}",
        r"\label{tab:ablation_metrics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccc}",
        r"\toprule",
        r"\textbf{Metric} & \makecell{\textbf{Full} \\ \textbf{Pipeline}} & \textbf{Monolithic} & \makecell{\textbf{Ablated} \\ \textbf{Pipeline}} & \makecell{\textbf{Full vs. Mono} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} & \makecell{\textbf{Full vs. Ablated} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} \\",
        r"\midrule",
    ]
    metrics_t4 = ["Answer Relevancy", "Clinical Validity", "Context Precision"]
    for idx, m_name in enumerate(metrics_t4):
        desc = t4_res["descriptive"][m_name]
        f_s, m_s, a_s = desc["Full"], desc["Monolithic"], desc["Ablated"]
        c_mono = next(c for c in t4_res["comparisons"] if c["metric"] == m_name and "Monolithic" in c["comparison"])
        c_abl = next(c for c in t4_res["comparisons"] if c["metric"] == m_name and "Ablated" in c["comparison"])

        t4_lines.append(
            f"{m_name} & "
            f"\\makecell{{\\textbf{{{f_s['mean']:.3f}}} $\\pm {f_s['se']:.3f}$ \\\\ {{[{f_s['ci_low']:.3f}, {f_s['ci_high']:.3f}]}}}} & "
            f"\\makecell{{{m_s['mean']:.3f} $\\pm {m_s['se']:.3f}$ \\\\ {{[{m_s['ci_low']:.3f}, {m_s['ci_high']:.3f}]}}}} & "
            f"\\makecell{{{a_s['mean']:.3f} $\\pm {a_s['se']:.3f}$ \\\\ {{[{a_s['ci_low']:.3f}, {a_s['ci_high']:.3f}]}}}} & "
            f"${_fmt_p3(c_mono['p_raw'])} \\,/\\, {_fmt_p3(c_mono['p_adj_holm'])}$ & "
            f"${_fmt_p3(c_abl['p_raw'])} \\,/\\, {_fmt_p3(c_abl['p_adj_holm'])}$ \\\\"
        )
        if idx < len(metrics_t4) - 1:
            t4_lines.append(r"\addlinespace")
    t4_lines.extend([r"\botrule", r"\end{tabular*}", r"\end{table}"])
    latex_out.append("\n".join(t4_lines))

    return "\n\n".join(latex_out)


def generate_markdown_report(
    t1_res: Dict[str, Any],
    t2_res: Dict[str, Any],
    t3_res: Dict[str, Any],
    t4_res: Dict[str, Any],
    latex_code: str,
) -> str:
    """Generate human-readable Markdown summary report."""
    md = [
        "# Statistical Analysis Report",
        f"\n**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n**Methodology:** 10,000-resample percentile bootstrap 95% CIs, Standard Errors (SE), two-sided paired Wilcoxon signed-rank tests, exact McNemar tests, and sequential Holm-Bonferroni FWER adjustment.",
        "\n---",
        "\n## 1. Overall Framework and Component-Wise Metrics (N=100 / N=70)",
        "| Component | Metric | Mean | SE | 95% Bootstrap CI | Sample ($N$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for name, item in t1_res.items():
        s = item["stats"]
        md.append(
            f"| {item['component']} | {name} | **{s['mean']:.3f}** | ± {s['se']:.3f} | [{s['ci_low']:.3f}, {s['ci_high']:.3f}] | {item['n_cases']} |"
        )

    md.extend([
        "\n---",
        "\n## 2. Jaccard Threshold Sensitivity (N=70)",
        "| Jaccard Threshold | Hit-Rate @ K (Mean ± SE [95% CI]) | Precision @ K (Mean ± SE [95% CI]) |",
        "| :---: | :---: | :---: |",
    ])
    for th, vals in t2_res.items():
        hr, pr = vals["hit_rate"], vals["precision"]
        md.append(
            f"| $\\tau = {th}$ | **{hr['mean']:.3f}** ± {hr['se']:.3f} [{hr['ci_low']:.2f}, {hr['ci_high']:.2f}] | **{pr['mean']:.3f}** ± {pr['se']:.3f} [{pr['ci_low']:.2f}, {pr['ci_high']:.2f}] |"
        )

    md.extend([
        "\n---",
        "\n## 3. Retrieval Baseline Comparison: FastText+IDF vs TF-IDF (N=100)",
        "| Metric / Threshold | FastText+IDF (Mean ± SE [95% CI]) | TF-IDF Baseline (Mean ± SE [95% CI]) | Paired Test | $p_{\\text{raw}}$ | $p_{\\text{adj}}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for th in [0.2, 0.3, 0.4, 0.5]:
        ft_h = t3_res["thresholds"][th]["hit_rate"]["FastText"]
        tf_h = t3_res["thresholds"][th]["hit_rate"]["TF-IDF"]
        mcn = t3_res["thresholds"][th]["hit_rate"]["mcnemar_contingency"]
        t_h = next(t for t in t3_res["tests"] if t["comparison"] == f"Hit-Rate@10 (tau={th})")
        md.append(
            f"| Hit-Rate@10 ($\\tau={th}$) | **{ft_h['mean']:.3f}** ± {ft_h['se']:.3f} [{ft_h['ci_low']:.2f}, {ft_h['ci_high']:.2f}] | {tf_h['mean']:.3f} ± {tf_h['se']:.3f} [{tf_h['ci_low']:.2f}, {tf_h['ci_high']:.2f}] | McNemar (FT_only={mcn['a_only']}, TF_only={mcn['b_only']}) | `{t_h['p_raw']:.4g}` | `{t_h['p_adj_holm']:.4g}` |"
        )

        ft_p = t3_res["thresholds"][th]["precision"]["FastText"]
        tf_p = t3_res["thresholds"][th]["precision"]["TF-IDF"]
        t_p = next(t for t in t3_res["tests"] if t["comparison"] == f"Precision@10 (tau={th})")
        ft_p_str = f"**{ft_p['mean']:.3f}**" if th > 0.2 else f"{ft_p['mean']:.3f}"
        tf_p_str = f"**{tf_p['mean']:.3f}**" if th == 0.2 else f"{tf_p['mean']:.3f}"
        md.append(
            f"| Precision@10 ($\\tau={th}$) | {ft_p_str} ± {ft_p['se']:.3f} [{ft_p['ci_low']:.2f}, {ft_p['ci_high']:.2f}] | {tf_p_str} ± {tf_p['se']:.3f} [{tf_p['ci_low']:.2f}, {tf_p['ci_high']:.2f}] | Paired Wilcoxon | `{t_p['p_raw']:.4g}` | `{t_p['p_adj_holm']:.4g}` |"
        )

    for gm_name in ["nDCG@10", "Mean Best Jaccard"]:
        ft_g = t3_res["global_metrics"][gm_name]["FastText"]
        tf_g = t3_res["global_metrics"][gm_name]["TF-IDF"]
        t_g = next(t for t in t3_res["tests"] if t["comparison"] == gm_name)
        md.append(
            f"| **{gm_name}** | **{ft_g['mean']:.3f}** ± {ft_g['se']:.3f} [{ft_g['ci_low']:.2f}, {ft_g['ci_high']:.2f}] | {tf_g['mean']:.3f} ± {tf_g['se']:.3f} [{tf_g['ci_low']:.2f}, {tf_g['ci_high']:.2f}] | Paired Wilcoxon | `{t_g['p_raw']:.4g}` | `{t_g['p_adj_holm']:.4g}` |"
        )

    md.extend([
        "\n---",
        "\n## 4. Architectural Ablation Study (N=100)",
        "| Metric | Full Pipeline (Mean ± SE [95% CI]) | Monolithic (Mean ± SE [95% CI]) | Ablated (Mean ± SE [95% CI]) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for m in ["Answer Relevancy", "Clinical Validity", "Context Precision"]:
        d = t4_res["descriptive"][m]
        f_s, m_s, a_s = d["Full"], d["Monolithic"], d["Ablated"]
        md.append(
            f"| **{m}** | **{f_s['mean']:.3f}** ± {f_s['se']:.3f} [{f_s['ci_low']:.3f}, {f_s['ci_high']:.3f}] | {m_s['mean']:.3f} ± {m_s['se']:.3f} [{m_s['ci_low']:.3f}, {m_s['ci_high']:.3f}] | {a_s['mean']:.3f} ± {a_s['se']:.3f} [{a_s['ci_low']:.3f}, {a_s['ci_high']:.3f}] |"
        )

    md.extend([
        "\n### Paired Wilcoxon Signed-Rank Tests (Holm-Bonferroni Adjusted)",
        "| Metric | Comparison | $p_{\\text{raw}}$ | $p_{\\text{adj}}$ | Verdict |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ])
    for c in t4_res["comparisons"]:
        p_adj_str = "< 0.0001" if c["p_adj_holm"] < 0.0001 else f"{c['p_adj_holm']:.4g}"
        md.append(
            f"| {c['metric']} | {c['comparison']} | `{c['p_raw']:.4g}` | `{p_adj_str}` | {c['verdict']} |"
        )

    md.extend([
        "\n---",
        "\n## 5. LaTeX Tables",
        "```latex",
        latex_code,
        "```",
    ])
    return "\n".join(md) + "\n"


def main() -> None:
    print("=" * 80)
    print("  STATISTICAL ANALYSIS — CLINICAL AGENTIC RAG BENCHMARK")
    print("=" * 80)

    raw_data = load_data()
    t4_res = analyze_ablation(raw_data)
    t1_res = analyze_overall(raw_data, t4_res)
    t2_res = analyze_thresholds(raw_data)
    t3_res = analyze_baseline_comparison(raw_data)

    latex_code = generate_latex_tables(t1_res, t2_res, t3_res, t4_res)
    md_content = generate_markdown_report(t1_res, t2_res, t3_res, t4_res, latex_code)

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    all_results = {
        "metadata": {
            "title": "Statistical Analysis Results (N=100 / N=70 Benchmark)",
            "timestamp": datetime.now().isoformat(),
            "methodology": "10,000-resample percentile bootstrap 95% CIs, SE, paired Wilcoxon, exact McNemar, Holm-Bonferroni FWER",
        },
        "table_1_overall": t1_res,
        "table_2_thresholds": t2_res,
        "table_3_baseline_comparison": t3_res,
        "table_4_ablation": t4_res,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved Markdown report: {REPORT_MD_PATH}")
    print(f"Saved JSON results:    {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
