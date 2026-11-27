"""
agents/orchestrator.py
──────────────────────
Agent 1 — LLM-based decision router.

Receives physician question in natural language and produces a structured
routing JSON that guides the rest of the pipeline.
"""

import json

from google import genai
from google.genai import types

from config.settings import Settings
from utils.logging import get_logger

logger = get_logger("orchestrator")

# ── System prompt per l'orchestratore ──
ROUTING_SYSTEM_PROMPT = """\
Sei un sistema di routing intelligente per una piattaforma medica.
Il tuo unico compito è analizzare la domanda del medico e restituire in output ESCLUSIVAMENTE un JSON di configurazione.

REGOLE PER I CAMPI DEL JSON:

1. "topics": Lista. Inserisci "pattern_analysis" (per diagnosi, rischi, confronti tra pazienti) e/o "consulto_farmacologico" (per terapie, cure, farmaci).
2. "similarity": Booleano. Metti SEMPRE 'true' se la domanda chiede di guardare al "database", "comorbidità", fare "confronti" o cercare "pazienti simili". Metti 'false' ESCLUSIVAMENTE se chiedono di fare un mero riassunto di chi è il paziente senza cercare connessioni esterne, oppure una domanda guarda al singolo paziente.
3. "k_similar": Intero che indica il numero di pazienti simili da recuperare.
   - PRIORITÀ ASSOLUTA: Se la domanda richiede esplicitamente un numero (es: "confronta 50 pazienti", "trova 30 casi simili"), USA ESATTAMENTE IL NUMERO SPECIFICATO (es. 50, 30), ignorando i default sottostanti.
   - DEFAULT SE NESSUN NUMERO È SPECIFICATO:
     - Usa 5 per richieste banalissime o veloci.
     - Usa 10 COME DEFAULT ASSOLUTO per la quasi totalità delle ricerche mediche.
     - Usa 15 se la richiesta raggiunge livelli estremi di rarità o complessità clinica.
4. "weights_similarity": Array di 3 numeri decimali (somma = 1.0). L'ORDINE DEVE ESSERE ESATTAMENTE QUESTO:
   - Valore 1: Peso Diagnosi
   - Valore 2: Peso Parametri Clinici
   - Valore 3: Peso Dati Demografici
5. "attributes_list": Lista di stringhe.
   - Inserisci SEMPRE: "demographics", "admission_details", "diagnoses", "outcome".
   - SE si chiedono "analisi approfondite"/"complete", OPPURE si menzionano esplicitamente "parametri clinici" o "esami": includi TASSATIVAMENTE anche "vitals", "labs", "previous_diagnoses".
   - SE si affronta il tema di terapie, cure mediche o farmaci: includi "prescriptions".

ESEMPIO STRUTTURALE DI RISPOSTA (da adattare):

{
    "reasoning": "Breve motivazione logica sul perché hai scelto questi parametri.",
    "topics": ["pattern_analysis"],
    "similarity": true,
    "k_similar": 10,
    "weights_similarity": [0.2, 0.2, 0.6],
    "attributes_list": ["demographics", "admission_details", "diagnoses", "outcome", "prescriptions", "vitals", "labs", "previous_diagnoses"]
}
"""

class Orchestrator:
    """
    Agent 1: LLM decision router.
    Analyzes physician question and produces execution plan.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.llm.orchestrator_model

        if not settings.llm.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Set it in .env file or environment variables."
            )

        self.client = genai.Client(
            api_key=settings.llm.api_key,
            http_options=types.HttpOptions(api_version=settings.llm.api_version),
        )

    def route(self, question: str, patient_id: int) -> dict:
        """
        Analyzes question and produces routing JSON.

        Args:
            question: Physician natural language question
            patient_id: Target patient hadm_id

        Returns:
            dict with keys: topics, similarity, k_similar,
                  weights_similarity, attributes_list, reasoning
        """
        logger.info(f"[agent]Orchestrator[/agent] — Routing question for patient {patient_id}")

        user_prompt = (
            f"Domanda del medico: \"{question}\"\n\n"
            f"Paziente target ID (hadm_id): {patient_id}\n\n"
            "Analizza la domanda e genera il JSON di routing."
        )

        try:
            # Integrate system prompt into user instructions (maximum compatibility)
            full_user_prompt = f"{ROUTING_SYSTEM_PROMPT}\n\n{user_prompt}"

            # Support for local Ollama models
            if ":" in self.model_id:
                import urllib.request
                import json as _json
                
                logger.info(f"Using local Ollama model: {self.model_id}")
                data = _json.dumps({
                    "model": self.model_id,
                    "prompt": full_user_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.3,
                        "top_k": 5
                    }
                }).encode('utf-8')
                
                req = urllib.request.Request("http://localhost:11434/api/generate", data=data, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req) as resp:
                    result = _json.loads(resp.read().decode('utf-8'))
                    raw_text = result.get("response", "").strip()
            else:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_user_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        top_p=0.3,
                        top_k=5,
                    ),
                )
    
                raw_text = response.text.strip()
            
            # Clean markdown json block if present
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
            routing = json.loads(raw_text)

            # Validation and sanitization
            routing = self._validate_routing(routing, patient_id, question)

            logger.info(
                f"[success]Routing:[/success] topics={routing['topics']}, "
                f"similarity={routing['similarity']}, k={routing.get('k_similar', 'N/A')}"
            )
            logger.info(f"  Reasoning: {routing.get('reasoning', 'N/A')}")

            return routing

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from orchestrator: {e}. Using fallback routing.")
            return self._fallback_routing(patient_id, question)
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}. Using fallback routing.")
            return self._fallback_routing(patient_id, question)

    def _validate_routing(self, routing: dict, patient_id: int, question: str) -> dict:
        """Validates and sanitizes routing JSON."""
        valid_topics = set(self.settings.available_topics)
        valid_attrs = set(self.settings.available_attributes)

        # Topics
        topics = routing.get("topics", [])
        topics = [t for t in topics if t in valid_topics]
        if not topics:
            topics = list(valid_topics)
        routing["topics"] = topics

        # Similarity (robust interpretation)
        sim_val = routing.get("similarity", True)
        if isinstance(sim_val, str):
            sim_enabled = sim_val.lower() in ("true", "1", "yes", "t", "on")
        else:
            sim_enabled = bool(sim_val)
        routing["similarity"] = sim_enabled

        # K similar
        k = routing.get("k_similar", self.settings.similarity.k_similar)
        if not sim_enabled:
            routing["k_similar"] = 0
        else:
            routing["k_similar"] = int(k or 10)

        # Weights
        weights = routing.get("weights_similarity")
        if not sim_enabled:
            routing["weights_similarity"] = None
        elif isinstance(weights, list) and len(weights) == 3:
            total = sum(weights)
            if total > 0:
                routing["weights_similarity"] = [w / total for w in weights]
            else:
                routing["weights_similarity"] = self.settings.similarity.weights
        else:
            routing["weights_similarity"] = self.settings.similarity.weights

        # Attributes
        attrs = routing.get("attributes_list", [])
        attrs = [a for a in attrs if a in valid_attrs]
        if not attrs:
            attrs = list(valid_attrs)
        routing["attributes_list"] = attrs

        # Metadata
        routing["patient_id"] = patient_id
        routing["prompt"] = question

        return routing

    def _fallback_routing(self, patient_id: int, question: str) -> dict:
        """Conservative fallback routing — includes everything."""
        return {
            "reasoning": "Fallback: complete routing for safety",
            "prompt": question,
            "patient_id": patient_id,
            "topics": list(self.settings.available_topics),
            "similarity": True,
            "k_similar": self.settings.similarity.k_similar,
            "weights_similarity": self.settings.similarity.weights,
            "attributes_list": list(self.settings.available_attributes),
        }

