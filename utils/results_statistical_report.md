# Statistical Analysis Report

**Generated at:** 2026-11-27 11:47:06

**Methodology:** 10,000-resample percentile bootstrap 95% CIs, Standard Errors (SE), two-sided paired Wilcoxon signed-rank tests, exact McNemar tests, and sequential Holm-Bonferroni FWER adjustment.

---

## 1. Overall Framework and Component-Wise Metrics (N=100 / N=70)
| Component | Metric | Mean | SE | 95% Bootstrap CI | Sample ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Orchestrator | Orchestrator Accuracy | **0.960** | ± 0.013 | [0.933, 0.983] | 100 |
| Similarity Engine | Hit-Rate @ K | **0.886** | ± 0.038 | [0.800, 0.957] | 70 |
| Similarity Engine | Precision @ K | **0.491** | ± 0.042 | [0.411, 0.574] | 70 |
| Similarity Engine | nDCG @ K | **0.875** | ± 0.011 | [0.854, 0.896] | 70 |
| Specialist & Synthesizer | Clinical Validity | **0.876** | ± 0.020 | [0.835, 0.913] | 100 |
| Report Synthesizer | Contradiction Rate | **0.000** | ± 0.000 | [0.000, 0.000] | 100 |
| End-to-End | Answer Relevancy | **0.763** | ± 0.004 | [0.755, 0.772] | 100 |
| Retrieval-to-Context | Context Precision | **0.831** | ± 0.019 | [0.794, 0.866] | 100 |

---

## 2. Jaccard Threshold Sensitivity (N=70)
| Jaccard Threshold | Hit-Rate @ K (Mean ± SE [95% CI]) | Precision @ K (Mean ± SE [95% CI]) |
| :---: | :---: | :---: |
| $\tau = 0.2$ | **0.886** ± 0.038 [0.800, 0.957] | **0.491** ± 0.042 [0.411, 0.574] |
| $\tau = 0.3$ | **0.714** ± 0.054 [0.614, 0.814] | **0.266** ± 0.041 [0.190, 0.349] |
| $\tau = 0.4$ | **0.429** ± 0.060 [0.314, 0.543] | **0.183** ± 0.038 [0.114, 0.263] |
| $\tau = 0.5$ | **0.286** ± 0.054 [0.186, 0.400] | **0.140** ± 0.038 [0.074, 0.219] |

---

## 3. Retrieval Baseline Comparison: FastText+IDF vs TF-IDF (N=100)
| Metric / Threshold | FastText+IDF (Mean ± SE [95% CI]) | TF-IDF Baseline (Mean ± SE [95% CI]) | Paired Test | $p_{\text{raw}}$ | $p_{\text{adj}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hit-Rate@10 ($\tau=0.2$) | **0.980** ± 0.014 [0.950, 1.000] | 0.930 ± 0.026 [0.880, 0.970] | McNemar (FT_only=7, TF_only=2) | `0.1797` | `0.8948` |
| Precision@10 ($\tau=0.2$) | 0.620 ± 0.034 [0.554, 0.686] | **0.668** ± 0.034 [0.601, 0.732] | Paired Wilcoxon | `0.1789` | `0.8948` |
| Hit-Rate@10 ($\tau=0.3$) | **0.760** ± 0.043 [0.680, 0.840] | 0.720 ± 0.045 [0.630, 0.810] | McNemar (FT_only=16, TF_only=12) | `0.5716` | `1` |
| Precision@10 ($\tau=0.3$) | **0.385** ± 0.035 [0.318, 0.454] | 0.332 ± 0.034 [0.269, 0.399] | Paired Wilcoxon | `0.2606` | `0.8948` |
| Hit-Rate@10 ($\tau=0.4$) | **0.550** ± 0.050 [0.450, 0.650] | 0.430 ± 0.050 [0.330, 0.530] | McNemar (FT_only=20, TF_only=8) | `0.0357` | `0.2499` |
| Precision@10 ($\tau=0.4$) | **0.211** ± 0.029 [0.155, 0.271] | 0.125 ± 0.020 [0.088, 0.166] | Paired Wilcoxon | `0.00534` | `0.05338` |
| Hit-Rate@10 ($\tau=0.5$) | **0.400** ± 0.049 [0.300, 0.500] | 0.310 ± 0.046 [0.220, 0.400] | McNemar (FT_only=18, TF_only=9) | `0.1221` | `0.7325` |
| Precision@10 ($\tau=0.5$) | **0.141** ± 0.026 [0.093, 0.195] | 0.068 ± 0.013 [0.044, 0.096] | Paired Wilcoxon | `0.00872` | `0.0785` |
| **nDCG@10** | **0.918** ± 0.006 [0.907, 0.930] | 0.870 ± 0.022 [0.825, 0.909] | Paired Wilcoxon | `0.9781` | `1` |
| **Mean Best Jaccard** | **0.458** ± 0.019 [0.421, 0.496] | 0.393 ± 0.019 [0.357, 0.429] | Paired Wilcoxon | `0.0134` | `0.1072` |

---

## 4. Architectural Ablation Study (N=100)
| Metric | Full Pipeline (Mean ± SE [95% CI]) | Monolithic (Mean ± SE [95% CI]) | Ablated (Mean ± SE [95% CI]) |
| :--- | :---: | :---: | :---: |
| **Answer Relevancy** | **0.763** ± 0.004 [0.755, 0.772] | 0.734 ± 0.006 [0.723, 0.745] | 0.727 ± 0.017 [0.690, 0.757] |
| **Clinical Validity** | **0.876** ± 0.020 [0.835, 0.913] | 0.853 ± 0.026 [0.800, 0.899] | 0.853 ± 0.022 [0.809, 0.893] |
| **Context Precision** | **0.831** ± 0.019 [0.794, 0.866] | 0.747 ± 0.028 [0.692, 0.799] | 0.762 ± 0.037 [0.690, 0.831] |

### Paired Wilcoxon Signed-Rank Tests (Holm-Bonferroni Adjusted)
| Metric | Comparison | $p_{\text{raw}}$ | $p_{\text{adj}}$ | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| Answer Relevancy | Full Pipeline vs Monolithic | `1.547e-06` | `< 0.0001` | Statistically Significant (p < 0.05) |
| Answer Relevancy | Full Pipeline vs Ablated | `0.8314` | `1` | Observed Trend (n.s.) |
| Clinical Validity | Full Pipeline vs Monolithic | `0.6882` | `1` | Observed Trend (n.s.) |
| Clinical Validity | Full Pipeline vs Ablated | `0.4025` | `1` | Observed Trend (n.s.) |
| Context Precision | Full Pipeline vs Monolithic | `0.0327` | `0.1635` | Observed Trend (n.s.) |
| Context Precision | Full Pipeline vs Ablated | `0.2305` | `0.9222` | Observed Trend (n.s.) |

---

## 5. LaTeX Tables
```latex
\begin{table}[!ht]
\centering
\caption{Overall Framework and Component-Wise Metrics. Note: End-to-end and routing metrics are calculated over the full benchmark ($N=100$, 50 clinical queries evaluated across 2 distinct patients), while Similarity Engine metrics are calculated over the subset of queries where retrieval was dynamically activated ($N=70$), evaluated at the query-specific $K$ assigned by the Orchestrator. Reported as Score $\pm$ SE [95\% Bootstrap CI].}
\label{tab:metrics_summary}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lllc}
\toprule
\textbf{Component} & \textbf{Metric} & \textbf{Evaluation Focus} & \makecell{\textbf{Score $\pm$ SE} \\ \textbf{[95\% CI]}} \\
\midrule
Orchestrator & Orchestrator Accuracy & Router classification correctness & \makecell{\textbf{0.960} $\pm 0.013$ \\ {[0.933, 0.983]}} \\
\addlinespace
Similarity Engine & Hit-Rate @ $K$ & Utility of retrieved cohort (Jaccard $\ge 0.2$) & \makecell{\textbf{0.886} $\pm 0.038$ \\ {[0.800, 0.957]}} \\
\addlinespace
Similarity Engine & Precision @ $K$ & Yield of retrieved cohort (Jaccard $\ge 0.2$) & \makecell{\textbf{0.491} $\pm 0.042$ \\ {[0.411, 0.574]}} \\
\addlinespace
Similarity Engine & nDCG @ $K$ & Ranking similarity cohort & \makecell{\textbf{0.875} $\pm 0.011$ \\ {[0.854, 0.896]}} \\
\addlinespace
Specialist \& Synthesizer & Clinical Validity & Automated medical plausibility & \makecell{\textbf{0.876} $\pm 0.020$ \\ {[0.835, 0.913]}} \\
\addlinespace
Report Synthesizer & Contradiction Rate & Inner-agent factual consistency & \makecell{\textbf{0.000} $\pm 0.000$ \\ {[0.000, 0.000]}} \\
\addlinespace
End-to-End & Answer Relevancy & Alignment of response to query & \makecell{\textbf{0.763} $\pm 0.004$ \\ {[0.755, 0.772]}} \\
\addlinespace
Retrieval-to-Context & Context Precision & Ranking quality of context & \makecell{\textbf{0.831} $\pm 0.019$ \\ {[0.794, 0.866]}} \\
\botrule
\end{tabular*}
\end{table}

\begin{table}[ht]
\centering
\caption{Retrieval Performance Under Varying Jaccard Similarity Thresholds ($N=70$, evaluated at the dynamically assigned $K$, reported as Mean $\pm$ SE [95\% Bootstrap CI]).}
\label{tab:thresholds}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccc}
\toprule
\textbf{Metric / Jaccard Threshold} & \textbf{0.2} & \textbf{0.3} & \textbf{0.4} & \textbf{0.5} \\
\midrule
Hit-Rate @ $K$ & \makecell{\textbf{0.886} $\pm 0.038$ \\ {[0.800, 0.957]}} & \makecell{\textbf{0.714} $\pm 0.054$ \\ {[0.614, 0.814]}} & \makecell{\textbf{0.429} $\pm 0.060$ \\ {[0.314, 0.543]}} & \makecell{\textbf{0.286} $\pm 0.054$ \\ {[0.186, 0.400]}} \\
\addlinespace
Precision @ $K$ & \makecell{\textbf{0.491} $\pm 0.042$ \\ {[0.411, 0.574]}} & \makecell{\textbf{0.266} $\pm 0.041$ \\ {[0.190, 0.349]}} & \makecell{\textbf{0.183} $\pm 0.038$ \\ {[0.114, 0.263]}} & \makecell{\textbf{0.140} $\pm 0.038$ \\ {[0.074, 0.219]}} \\
\botrule
\end{tabular*}
\end{table}

\begin{table}[ht]
\centering
\caption{Comparative Retrieval Performance on the 100 benchmark patients: FastText+IDF vs. TF-IDF Baseline across Multiple Jaccard Thresholds (Mean $\pm$ SE [95\% Bootstrap CI], with exact Paired McNemar tests for binary Hit-Rate and Paired Wilcoxon signed-rank tests for Precision, nDCG, and Best Jaccard; $p_{\text{adj}}$ corrected via Holm--Bonferroni).}
\label{tab:baseline_comparison}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}clccc}
\toprule
\textbf{Threshold} & \textbf{Method} & \textbf{Hit-Rate@10} & \textbf{Precision@10} & \makecell{\textbf{Significance} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} \\
\midrule
\multirow{3}{*}{0.2} & \textbf{FastText+IDF} & \makecell{\textbf{0.980} $\pm 0.014$ \\ {[0.950, 1.000]}} & \makecell{0.620 $\pm 0.034$ \\ {[0.554, 0.686]}} & \multirow{3}{*}{\makecell[l]{Hit: $p=0.180 \,/\, 0.895$ \\ Prec: $p=0.179 \,/\, 0.895$}} \\
 & TF-IDF & \makecell{0.930 $\pm 0.026$ \\ {[0.880, 0.970]}} & \makecell{\textbf{0.668} $\pm 0.034$ \\ {[0.601, 0.732]}} & \\
\midrule
\multirow{3}{*}{0.3} & \textbf{FastText+IDF} & \makecell{\textbf{0.760} $\pm 0.043$ \\ {[0.680, 0.840]}} & \makecell{\textbf{0.385} $\pm 0.035$ \\ {[0.318, 0.454]}} & \multirow{3}{*}{\makecell[l]{Hit: $p=0.572 \,/\, 1.000$ \\ Prec: $p=0.261 \,/\, 0.895$}} \\
 & TF-IDF & \makecell{0.720 $\pm 0.045$ \\ {[0.630, 0.810]}} & \makecell{0.332 $\pm 0.034$ \\ {[0.269, 0.399]}} & \\
\midrule
\multirow{3}{*}{0.4} & \textbf{FastText+IDF} & \makecell{\textbf{0.550} $\pm 0.050$ \\ {[0.450, 0.650]}} & \makecell{\textbf{0.211} $\pm 0.029$ \\ {[0.155, 0.271]}} & \multirow{3}{*}{\makecell[l]{Hit: $p=0.036 \,/\, 0.250$ \\ Prec: $p=0.005 \,/\, 0.053$}} \\
 & TF-IDF & \makecell{0.430 $\pm 0.050$ \\ {[0.330, 0.530]}} & \makecell{0.125 $\pm 0.020$ \\ {[0.088, 0.166]}} & \\
\midrule
\multirow{3}{*}{0.5} & \textbf{FastText+IDF} & \makecell{\textbf{0.400} $\pm 0.049$ \\ {[0.300, 0.500]}} & \makecell{\textbf{0.141} $\pm 0.026$ \\ {[0.093, 0.195]}} & \multirow{3}{*}{\makecell[l]{Hit: $p=0.122 \,/\, 0.732$ \\ Prec: $p=0.009 \,/\, 0.079$}} \\
 & TF-IDF & \makecell{0.310 $\pm 0.046$ \\ {[0.220, 0.400]}} & \makecell{0.068 $\pm 0.013$ \\ {[0.044, 0.096]}} & \\
\midrule
\midrule
& & \textbf{nDCG@10} & \makecell{\textbf{Mean Best} \\ \textbf{Jaccard}} & \makecell{\textbf{Significance} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} \\
\midrule
& \textbf{FastText+IDF} & \makecell{\textbf{0.918} $\pm 0.006$ \\ {[0.907, 0.930]}} & \makecell{\textbf{0.458} $\pm 0.019$ \\ {[0.421, 0.496]}} & \multirow{3}{*}{\makecell[l]{nDCG: $p=0.978 \,/\, 1.000$ \\ Jacc: $p=0.013 \,/\, 0.107$}} \\
& TF-IDF & \makecell{0.870 $\pm 0.022$ \\ {[0.825, 0.909]}} & \makecell{0.393 $\pm 0.019$ \\ {[0.357, 0.429]}} & \\
\botrule
\end{tabular*}
\end{table}

\begin{table}[ht]
\centering
\caption{Ablation Study Results: Comparison of the Full Pipeline against Architectural Baselines across the Benchmark Dataset ($N=100$, Reported as Mean $\pm$ SE [95\% Bootstrap CI]). Significance is reported as unadjusted $p_{\text{raw}} \,/\, \text{Holm--Bonferroni adjusted } p_{\text{adj}}$ using paired Wilcoxon signed-rank tests.}
\label{tab:ablation_metrics}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccc}
\toprule
\textbf{Metric} & \makecell{\textbf{Full} \\ \textbf{Pipeline}} & \textbf{Monolithic} & \makecell{\textbf{Ablated} \\ \textbf{Pipeline}} & \makecell{\textbf{Full vs. Mono} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} & \makecell{\textbf{Full vs. Ablated} \\ ($p_{\text{raw}} \,/\, p_{\text{adj}}$)} \\
\midrule
Answer Relevancy & \makecell{\textbf{0.763} $\pm 0.004$ \\ {[0.755, 0.772]}} & \makecell{0.734 $\pm 0.006$ \\ {[0.723, 0.745]}} & \makecell{0.727 $\pm 0.017$ \\ {[0.690, 0.757]}} & $<0.001 \,/\, <0.001$ & $0.831 \,/\, 1.000$ \\
\addlinespace
Clinical Validity & \makecell{\textbf{0.876} $\pm 0.020$ \\ {[0.835, 0.913]}} & \makecell{0.853 $\pm 0.026$ \\ {[0.800, 0.899]}} & \makecell{0.853 $\pm 0.022$ \\ {[0.809, 0.893]}} & $0.688 \,/\, 1.000$ & $0.403 \,/\, 1.000$ \\
\addlinespace
Context Precision & \makecell{\textbf{0.831} $\pm 0.019$ \\ {[0.794, 0.866]}} & \makecell{0.747 $\pm 0.028$ \\ {[0.692, 0.799]}} & \makecell{0.762 $\pm 0.037$ \\ {[0.690, 0.831]}} & $0.033 \,/\, 0.163$ & $0.231 \,/\, 0.922$ \\
\botrule
\end{tabular*}
\end{table}
```
