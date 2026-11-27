"""
agents/specialist_llm.py
─────────────────────────
Agent 5 — Clinical Specialist LLM.

Clinical reasoning engine of the system. Receives structured context and
physician question, performs advanced clinical reasoning, and produces analysis.
"""

import json
import time
import concurrent.futures
from typing import Any, Dict, List

from google import genai
from google.genai import types

from config.settings import Settings
from utils.logging import get_logger, LogTimer

logger = get_logger("specialist_llm")

# ── Specialized prompts by topic ──

SPECIALIST_PROMPTS = {
    "pattern_analysis": """\
Sei un sistema avanzato di Clinical Decision Support. Analizza il 'target_patient' \
confrontandolo con la coorte di 'similar_patients'.
Il tuo compito NON è riassumere la cartella, ma rispondere direttamente e in modo focalizzato alla 'DOMANDA DEL MEDICO'.

REGOLE CLINICO-TEMPORALI TASSATIVE:
La tua analisi deve essere temporale, un innalzamento di valori nelle prime giornate \
dall'intervento o fase acuta è la NORMALE risposta fisiologica. NON segnalarlo come \
rischio imminente a meno che non ci si trovi oltre la terza giornata. Puoi capire il \
numero delle giornate dalle medications che sono elencate per giorno. Considera la prima \
data come "Giorno 0" e le successive con il numero ad aumentare. Nel paziente target va \
data priorità all'ultima giornata elencata. Non sovrastimare la gravità di parametri che \
possono essere influenzati da farmaci se sono accettabili per il contesto acuto. Inoltre \
non prestare troppa attenzione ai dosaggi perchè spesso rappresentano il totale preparato \
non quello effettivamente somministrato.

REGOLA FONDAMENTALE SULLA STRUTTURA DEL REPORT:
1. SE LA DOMANDA DEL MEDICO E' SPECIFICA (es. chiede di specifiche comorbilità, sintomi o dettagli da cercare):
   - IGNORA la struttura standard a paragrafi.
   - RISPONDI ESATTAMENTE a ciò che viene chiesto, in modo iper-focalizzato e dettagliato.
   - Usa il confronto con la coorte SOLO per supportare la risposta a quella specifica domanda. NON aggiungere dettagli generici (come gravità complessiva se non richiesta).
   - Approfondisci al massimo l'aspetto specifico richiesto.

2. SE LA DOMANDA DEL MEDICO E' GENERICA (es. "Fai un'analisi", "Cosa ne pensi?"):
   - Struttura la risposta in tre paragrafi discorsivi ma densi:
     a) **Valutazione della Gravità Relativa:** Il paziente target è più o meno grave della media dei simili?
     b) **Pattern di Complicazioni Attese:** Quale complicazione dei simili non è ancora presente nel target?
     c) **Discrepanze Cliniche:** Quale parametro del target stona rispetto ai simili?

Sii diretto, professionale e predittivo.

DATI JSON: {context_string}
""",

    "consulto_farmacologico": """\
Agisci come un Farmacologo Clinico Esperto. Analizza la terapia del 'target_patient' \
in relazione alla sua problematica clinica e al confronto con i 'similar_patients'.
Il tuo compito principale è rispondere direttamente e in modo focalizzato alla 'DOMANDA DEL MEDICO'.

REGOLE CLINICO-TEMPORALI TASSATIVE:
La tua analisi deve essere temporale, un innalzamento di valori nelle prime giornate \
dall'intervento o fase acuta è la NORMALE risposta fisiologica. NON segnalarlo come \
rischio imminente a meno che non ci si trovi oltre la terza giornata. Puoi capire il \
numero delle giornate dalle medications che sono elencate per giorno. Considera la prima \
data come "Giorno 0" e le successive con il numero ad aumentare. Nel paziente target va \
data priorità all'ultima giornata elencata. Non sovrastimare la gravità di parametri che \
possono essere influenzati da farmaci se sono accettabili per il contesto acuto. Inoltre \
non prestare troppa attenzione ai dosaggi perchè spesso rappresentano il totale preparato \
non quello effettivamente somministrato.

REGOLA FONDAMENTALE SULLA STRUTTURA DEL REPORT:
1. SE LA DOMANDA DEL MEDICO E' SPECIFICA (es. su un farmaco, su una classe di farmaci, o su sintomi per cui chiede modifiche terapeutiche):
   - IGNORA la struttura standard.
   - RISPONDI ESATTAMENTE alla richiesta in modo dettagliato, tralasciando tutto ciò che non riguarda il problema specifico posto dal medico.
   - Usa i 'similar_patients' SOLO per confermare o smentire la strategia associata a quella specifica richiesta.
   - Non elencare deprescribing o gap terapeutici fuori contesto se la domanda verte su tutt'altro.

2. SE LA DOMANDA DEL MEDICO E' GENERICA (es. "Analizza la terapia", "Che ne pensi dei farmaci?"):
   - Fornisci un parere critico diviso in tre punti logici:
     a) **Analisi di Appropriatezza & Deprescribing:** Identifica farmaci pericolosi/inutili e suggerisci sospensione.
     b) **Gap Terapeutici:** Suggerisci farmaci presenti nei simili ma assenti nel target.
     c) **Monitoraggio Farmacocinetico:** Suggerisci aggiustamenti di dosaggio per i farmaci correnti.
   - Non elencare i farmaci che vanno bene, ma solo cosa togliere e cosa aggiungere.

DATI JSON: {context_string}
""",
}


class SpecialistLLM:
    """
    Agent 5: Specialist LLM for advanced clinical reasoning.
    Executes topic-specific analysis based on Orchestrator assignment.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.llm.specialist_model

        if not settings.llm.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Set it in .env file or environment variables."
            )

        self.client = genai.Client(
            api_key=settings.llm.api_key,
            http_options=types.HttpOptions(api_version=settings.llm.api_version),
        )

    def analyze(
        self,
        context: Dict[str, Any],
        question: str,
        topics: List[str],
    ) -> Dict[str, str]:
        """
        Executes requested clinical analyses in parallel.

        Args:
            context: Complete context JSON from ContextConstructor
            question: Original physician question
            topics: List of topics to analyze

        Returns:
            Dict[topic_name, analysis_text]
        """
        results = {}
        valid_topics = []

        for topic in topics:
            if topic not in SPECIALIST_PROMPTS:
                logger.warning(f"Topic '{topic}' has no specialized prompt. Skipped.")
            else:
                valid_topics.append(topic)

        if not valid_topics:
            return results

        def _do_analyze(t: str) -> tuple[str, str | None]:
            with LogTimer(f"SpecialistLLM — {t}", logger):
                return t, self._run_analysis(context, question, t)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_topics), 5)) as executor:
            future_to_topic = {executor.submit(_do_analyze, t): t for t in valid_topics}
            for future in concurrent.futures.as_completed(future_to_topic):
                t = future_to_topic[future]
                try:
                    topic, response = future.result()
                    if response:
                        results[topic] = response
                except Exception as e:
                    logger.error(f"Unexpected error in parallel execution for topic '{t}': {e}")

        return results

    def _run_analysis(
        self,
        context: Dict[str, Any],
        question: str,
        topic: str,
    ) -> str | None:
        """
        Executes a single analysis for a specific topic.
        Retries indefinitely on rate-limits or transient errors
        using exponential backoff (2s -> 4s -> 8s -> ... max 60s).
        """
        logger.info(f"[agent]SpecialistLLM[/agent] — Analysis: {topic}")

        # Prepare JSON context string
        context_string = json.dumps(context, indent=2, ensure_ascii=False)

        # Format specialized prompt
        prompt_template = SPECIALIST_PROMPTS[topic]
        prompt_final = prompt_template.format(context_string=context_string)

        # Add original physician question as context
        full_prompt = (
            f"DOMANDA DEL MEDICO: {question}\n\n"
            f"{prompt_final}"
        )

        # ── Count tokens once ──
        try:
            token_count = self.client.models.count_tokens(
                model=self.model_id,
                contents=full_prompt,
            )
            logger.info(f"  Estimated input tokens: {token_count.total_tokens}")
        except Exception:
            pass  # Do not block if count_tokens fails

        # ── Retry with exponential backoff ──
        attempt = 0
        wait = 2  # initial seconds
        while True:
            attempt += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.settings.llm.specialist_temperature,
                        top_p=self.settings.llm.specialist_top_p,
                        top_k=self.settings.llm.specialist_top_k,
                    ),
                )

                result_text = response.text
                if result_text:
                    logger.info(
                        f"[success]Response received:[/success] "
                        f"{len(result_text)} characters"
                        + (f" (attempt {attempt})" if attempt > 1 else "")
                    )
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        meta = response.usage_metadata
                        logger.info(
                            f"  Tokens — input: {getattr(meta, 'prompt_token_count', 'N/A')}, "
                            f"output: {getattr(meta, 'candidates_token_count', 'N/A')}"
                        )
                    return result_text.strip()
                else:
                    logger.warning(
                        f"Empty response for topic '{topic}' (attempt {attempt}). "
                        f"Retrying in {wait}s..."
                    )

            except Exception as e:
                err_str = str(e).lower()
                # Transient errors: rate limit (429), server error (5xx), timeout
                is_transient = any(kw in err_str for kw in (
                    "429", "quota", "rate", "too many", "resource_exhausted",
                    "500", "503", "timeout", "deadline", "unavailable",
                ))
                if is_transient:
                    logger.warning(
                        f"  [{topic}] Attempt {attempt} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                else:
                    # Non-recoverable error (e.g. invalid prompt, auth)
                    logger.error(
                        f"  [ERROR] [{topic}] Non-transient error at attempt {attempt}: {e}"
                    )
                    return None

            time.sleep(wait)
            wait = min(wait * 2, 60)  # exponential backoff, max 60s

