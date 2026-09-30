"""
agents/report_synthesizer.py
──────────────────────────────
Agent 6 — Final Report Synthesizer.

Receives specialist analyses and combines them into a single coherent
narrative report, aligned with the physician's original question.
"""

import os
from typing import Dict, List, Optional
from google import genai
from google.genai import types

from config.settings import Settings
from utils.logging import get_logger, LogTimer

logger = get_logger("report_synthesizer")

SYNTHESIS_PROMPT = """\
Agisci come un Medical Editor e Senior Physician. Il tuo compito è unificare \
diversi topic di analisi clinica in un unico report coerente e professionale.

## INPUT DISPONIBILI
1. DOMANDA ORIGINALE: "{question}"
2. ANALISI SPECIALISTICHE:
{specialist_outputs}

## REGOLE DI SINTESI E FOCALIZZAZIONE
- **Rispettare la Domanda Originale come Priorità Assoluta**: Se la "DOMANDA ORIGINALE" pone un quesito specifico (es. ricerca di comorbilità nascoste, consigli terapeutici mirati), IL REPORT DEVE RISPONDERE IN MANIERA ESPLICITA E CENTRALE A QUELLA DOMANDA. Taglia senza pietà i dettagli generali provenienti dagli specialisti se oscurano il focus sulla domanda.
- **Integrazione Totale**: Non limitarti a riassumere i topic separatamente o fare un taglia e incolla. Crea un discorso unico. Se un pattern clinico spiega un suggerimento farmacologico relativo alla domanda, uniscili nello stesso capoverso.
- **Adattamento Dinamico della Struttura**:
  - Se la domanda è iper-specifica (es. "aiuto la paziente sta male, trova comorbilità e migliora farmaci"): Struttura il report in modo da rispondere a quei due esatti punti: 1) Possibili comorbilità nascoste nei simili; 2) Miglioramenti terapeutici. Evita analisi superflue (es. gravità generica, se non funzionale al problema narrato).
  - Se la domanda è esplorativa/generica: usa un'analisi strutturata in paragrafi tematici.
- **Tone**: Professionale, da medico a medico, asciutto, privo di disclaimer etici inutili (es. "sono solo un'IA...").
- **Zero Ridondanza**: Elimina i preamboli e vai dritto al nucleo clinico.
- **Fiducia**: Non inventare nuovi fatti clinici. Limitati alle prove fornite dagli specialisti.

Rispondi sempre nel modo più aderente possibile all'esigenza specifica espressa dal medico, valorizzando i dati degli specialisti che rispondono al suo dubbio.
"""

class ReportSynthesizer:
    """
    Agent 6: Final clinical report synthesis.
    Unifies outputs from specialist agents.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.llm.synthesizer_model

        if not settings.llm.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=settings.llm.api_key,
            http_options=types.HttpOptions(api_version=settings.llm.api_version),
        )

    def synthesize(self, question: str, specialist_results: Dict[str, str]) -> str:
        """
        Generates the unified final report.
        
        Args:
            question: Original physician question
            specialist_results: Map {topic: analysis_text}
            
        Returns:
            Synthesized final report text.
        """
        if not specialist_results:
            return "No analysis available for synthesis."

        # Prepare string with specialist outputs
        specialist_outputs = ""
        for topic, text in specialist_results.items():
            specialist_outputs += f"### TOPIC: {topic.upper()}\n{text}\n\n"

        prompt = SYNTHESIS_PROMPT.format(
            question=question,
            specialist_outputs=specialist_outputs
        )

        logger.info(f"[agent]ReportSynthesizer[/agent] — Generating final report with {self.model_id}")

        with LogTimer("ReportSynthesizer — Final synthesis", logger):
            try:
                if ":" in self.model_id:
                    import urllib.request
                    import json as _json
                    
                    logger.info(f"Using local Ollama model: {self.model_id}")
                    data = _json.dumps({
                        "model": self.model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3
                        }
                    }).encode('utf-8')
                    
                    req = urllib.request.Request("http://localhost:11434/api/generate", data=data, headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req) as resp:
                        result = _json.loads(resp.read().decode('utf-8'))
                        report = result.get("response", "").strip()
                else:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,  # Low temperature for fidelity
                        ),
                    )
                    
                    report = response.text
                if report:
                    logger.info(f"[success]Final report generated ({len(report)} characters)[/success]")
                    return report.strip()
                else:
                    logger.warning("Empty synthesis output received.")
                    return self._fallback_join(specialist_results)
                    
            except Exception as e:
                logger.error(f"Error in final synthesis: {e}")
                return self._fallback_join(specialist_results)

    def _fallback_join(self, results: Dict[str, str]) -> str:
        """Simple text concatenation on LLM synthesis failure."""
        output = "--- CLINICAL REPORT (Automatic Merge) ---\n\n"
        for topic, text in results.items():
            output += f"【 {topic.upper()} 】\n{text}\n\n"
        return output

