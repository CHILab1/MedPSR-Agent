"""
core/pipeline.py
────────────────
Main agentic pipeline coordinator.

Orchestrates full flow:
  Question → Orchestrator → DataGatherer + SimilarityEngine
          → ContextConstructor → SpecialistLLM → ReportSynthesizer → Response
"""

import json
import time
from typing import Any, Dict, Optional

from agents.orchestrator import Orchestrator
from agents.data_gatherer import DataGatherer
from agents.similarity_engine import SimilarityEngine
from agents.context_constructor import ContextConstructor
from agents.specialist_llm import SpecialistLLM
from agents.report_synthesizer import ReportSynthesizer
from config.settings import Settings
from utils.logging import get_logger, LogTimer, console

logger = get_logger("pipeline")


class AgenticPipeline:
    """
    Multi-agent system coordinator.
    Manages initialization, data loading, and execution
    of the entire agent chain.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

        # ── Agents ──
        self.orchestrator: Optional[Orchestrator] = None
        self.data_gatherer: Optional[DataGatherer] = None
        self.similarity_engine: Optional[SimilarityEngine] = None
        self.context_constructor: Optional[ContextConstructor] = None
        self.specialist: Optional[SpecialistLLM] = None
        self.synthesizer: Optional[ReportSynthesizer] = None

        self._initialized = False

    def initialize(self, skip_milvus: bool = False):
        """
        Initializes all agents and loads data.

        Args:
            skip_milvus: If True, skips SimilarityEngine loading (useful for testing without Milvus)
        """
        if self._initialized:
            return

        with LogTimer("Agentic pipeline initialization", logger):
            console.rule("[bold cyan]AGENTIC CLINICAL DECISION SUPPORT[/bold cyan]")

            # 1. Orchestrator (LLM Router)
            logger.info("Initializing Agent 1: [agent]Orchestrator[/agent]")
            self.orchestrator = Orchestrator(self.settings)

            # 2. Data Gatherer
            logger.info("Initializing Agent 3: [agent]DataGatherer[/agent]")
            self.data_gatherer = DataGatherer(self.settings)
            self.data_gatherer.load()

            # 3. Similarity Engine (optional)
            if not skip_milvus:
                logger.info("Initializing Agent 4: [agent]SimilarityEngine[/agent]")
                self.similarity_engine = SimilarityEngine(self.settings)
                self.similarity_engine.load(
                    hadm_to_idx=self.data_gatherer._hadm_to_idx
                )
            else:
                logger.info("SimilarityEngine skipped (--no-similarity)")

            # 4. Context Constructor
            logger.info("Initializing Agent 2: [agent]ContextConstructor[/agent]")
            self.context_constructor = ContextConstructor(self.settings)

            # 5. Specialist LLM
            logger.info("Initializing Agent 5: [agent]SpecialistLLM[/agent]")
            self.specialist = SpecialistLLM(self.settings)

            # 6. Report Synthesizer
            logger.info("Initializing Agent 6: [agent]ReportSynthesizer[/agent]")
            self.synthesizer = ReportSynthesizer(self.settings)

            self._initialized = True
            console.rule("[bold green]SYSTEM READY[/bold green]")

    def run(
        self,
        question: str,
        patient_id: int,
        k_similar: Optional[int] = None,
        force_no_similarity: bool = False,
        save_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete agentic pipeline.

        Args:
            question: Physician question in natural language
            patient_id: Target patient hadm_id
            k_similar: Override for number of similar patients
            force_no_similarity: Force disable similarity
            save_context: If True, saves JSON context to file

        Returns:
            Dict containing routing, context, and responses
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Patient validation
        if not self.data_gatherer.patient_exists(patient_id):
            available = self.data_gatherer.get_available_hadm_ids(5)
            raise ValueError(
                f"Patient {patient_id} not found in dataset.\n"
                f"Example available hadm_id: {available}"
            )

        pipeline_start = time.perf_counter()
        console.rule(f"[bold cyan]PATIENT ANALYSIS {patient_id}[/bold cyan]")

        # ═══════════════════════════════════════════════════════
        # STEP 1: ORCHESTRATOR — Decision routing
        # ═══════════════════════════════════════════════════════
        with LogTimer("STEP 1: Orchestrator", logger):
            routing = self.orchestrator.route(question, patient_id)

        # Override from CLI
        if force_no_similarity:
            routing["similarity"] = False
        if k_similar is not None:
            routing["k_similar"] = k_similar

        # Log complete routing
        logger.info(f"Final routing:\n{json.dumps(routing, indent=2, ensure_ascii=False)}")

        # ═══════════════════════════════════════════════════════
        # STEP 2: CONTEXT CONSTRUCTOR → DataGatherer + Similarity
        # ═══════════════════════════════════════════════════════
        with LogTimer("STEP 2: Context Construction", logger):
            context = self.context_constructor.build_context(
                routing=routing,
                data_gatherer=self.data_gatherer,
                similarity_engine=self.similarity_engine,
            )

        # Optional context saving
        if save_context:
            ctx_path = f"context_{patient_id}_{int(time.time())}.json"
            with open(ctx_path, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=4, ensure_ascii=False)
            logger.info(f"Context saved to: {ctx_path}")

        # ═══════════════════════════════════════════════════════
        # STEP 3: SPECIALIST LLM — Clinical reasoning
        # ═══════════════════════════════════════════════════════
        with LogTimer("STEP 3: Specialist LLM", logger):
            responses = self.specialist.analyze(
                context=context,
                question=question,
                topics=routing["topics"],
            )

        # ═══════════════════════════════════════════════════════
        # STEP 4: REPORT SYNTHESIZER — Final synthesis
        # ═══════════════════════════════════════════════════════
        final_report = ""
        if responses:
            with LogTimer("STEP 4: Report Synthesizer", logger):
                final_report = self.synthesizer.synthesize(
                    question=question,
                    specialist_results=responses
                )

        # ═══════════════════════════════════════════════════════
        # FINAL RESULT
        # ═══════════════════════════════════════════════════════
        total_time = time.perf_counter() - pipeline_start
        console.rule(f"[bold green]ANALYSIS COMPLETED in {total_time:.2f}s[/bold green]")

        result = {
            "patient_id": patient_id,
            "question": question,
            "routing": routing,
            "responses": responses,
            "final_report": final_report,
            "timing": {
                "total_seconds": round(total_time, 2),
            },
        }

        # Display responses
        self._display_responses(responses, final_report)

        return result

    def _display_responses(self, responses: Dict[str, str], final_report: str = ""):
        """Displays formatted responses in console."""
        for topic, text in responses.items():
            console.print()
            console.rule(f"[bold magenta]{_topic_title(topic)}[/bold magenta]")
            console.print(text)
            console.print()

        # Unified report
        if final_report:
            console.print()
            console.rule("[bold green]✧ UNIFIED CLINICAL REPORT ✧[/bold green]")
            console.print(final_report)
            console.print()

    def close(self):
        """Close resources (Milvus connections, etc.)."""
        if self.similarity_engine:
            self.similarity_engine.close()


def _topic_title(topic: str) -> str:
    """Converts topic name into readable title."""
    titles = {
        "pattern_analysis": "📊 Diagnostic Pattern Analysis",
        "consulto_farmacologico": "💊 Pharmacological Consultation",
    }
    return titles.get(topic, topic.replace("_", " ").title())

