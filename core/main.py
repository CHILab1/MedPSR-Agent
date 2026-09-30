"""
main.py — Agentic Clinical Decision Support System
═══════════════════════════════════════════════════
CLI Entry point for the MIMIC-IV multi-agent system.

Usage:
    python main.py --patient_id 20742541 --question "Analyze diagnostic patterns"
    python main.py --patient_id 20742541 --question "Pharmacological consultation" --k_similar 5
    python main.py --patient_id 20742541 --question "Patient data" --no-similarity
    python main.py --patient_id 20742541 --question "..." --save-context --verbose
"""

import argparse
import json
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from core.pipeline import AgenticPipeline
from utils.logging import get_logger, console

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(
        description="🏥 Agentic Clinical Decision Support — MIMIC-IV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
    Examples:
    python main.py --patient_id 20742541 --question "Analyze patient diagnostic patterns"
    python main.py --patient_id 26133978 --question "Pharmacological consultation" --k_similar 5
    python main.py --patient_id 20742541 --question "Assess risk" --no-similarity
            """,
    )

    parser.add_argument(
        "--patient_id",
        type=int,
        required=True,
        help="Target admission ID (hadm_id)",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Physician question in natural language",
    )
    parser.add_argument(
        "--k_similar",
        type=int,
        default=None,
        help="Override: number of similar patients (default: decided by orchestrator)",
    )
    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Disable similar patient search",
    )
    parser.add_argument(
        "--save-context",
        action="store_true",
        help="Save generated JSON context to file",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save final report to text file",
    )
    parser.add_argument(
        "--mimic-path",
        type=str,
        default=None,
        help="Override: path to original MIMIC-IV folder",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detailed debug output",
    )

    args = parser.parse_args()

    # ── Settings ──
    overrides = {}
    if args.mimic_path:
        overrides["mimic_data_path"] = args.mimic_path

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    settings = get_settings(**overrides)

    # Check API key
    if not settings.llm.api_key:
        console.print(
            "[bold red]❌ GEMINI_API_KEY is not configured![/bold red]\n"
            "Create a .env file in the project folder with:\n"
            "  GEMINI_API_KEY=your-api-key\n"
            "Or export environment variable.",
            highlight=False,
        )
        sys.exit(1)

    # ── Pipeline ──
    pipeline = AgenticPipeline(settings)

    try:
        # Initialization
        pipeline.initialize(skip_milvus=args.no_similarity)

        # Execution
        result = pipeline.run(
            question=args.question,
            patient_id=args.patient_id,
            k_similar=args.k_similar,
            force_no_similarity=args.no_similarity,
            save_context=args.save_context,
        )

        # Optional report saving
        if args.save_report:
            _save_report(result, args.save_report)
            console.print(f"\n[bold green]Report saved to: {args.save_report}[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        pipeline.close()


def _save_report(result: dict, filepath: str):
    """Saves final report in text format."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  AGENTIC CLINICAL DECISION SUPPORT — REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Patient (hadm_id): {result['patient_id']}\n")
        f.write(f"Question: {result['question']}\n")
        f.write(f"Total time: {result['timing']['total_seconds']}s\n\n")

        routing = result.get("routing", {})
        f.write(f"Analyzed topics: {', '.join(routing.get('topics', []))}\n")
        f.write(f"Similarity: {'Yes' if routing.get('similarity') else 'No'}\n")
        if routing.get("similarity"):
            f.write(f"K similar: {routing.get('k_similar')}\n")
            f.write(f"Weights: {routing.get('weights_similarity')}\n")
        f.write(f"Reasoning: {routing.get('reasoning', 'N/A')}\n")
        f.write("\n" + "=" * 70 + "\n")
        
        # 1. UNIFIED REPORT (Agent 6)
        if result.get("final_report"):
            f.write("  ✧ UNIFIED CLINICAL REPORT ✧\n")
            f.write("=" * 70 + "\n\n")
            f.write(result["final_report"] + "\n\n")
            f.write("=" * 70 + "\n")
        
        # 2. DETAILED ANALYSES (Specialists)
        f.write("  DETAILED COMPONENT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        for topic, text in result.get("responses", {}).items():
            topic_map = {
                "pattern_analysis": "DIAGNOSTIC PATTERN ANALYSIS",
                "consulto_farmacologico": "PHARMACOLOGICAL CONSULTATION",
            }
            title = topic_map.get(topic, topic.upper())
            f.write(f"  {title}\n")
            f.write("-" * 70 + "\n")
            f.write(text + "\n\n")

        f.write("=" * 70 + "\n")


if __name__ == "__main__":
    main()