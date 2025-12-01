"""
CLI entry point for the idea-first evolution loop.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .engine import IdeaProgramEngine


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the idea/program evolution loop.")
    parser.add_argument("initial_program", type=Path, help="Path to the initial program file.")
    parser.add_argument("evaluation_file", type=Path, help="Path to evaluation file (evaluate function).")
    parser.add_argument("--config", type=Path, default=None, help="OpenEvolve-style YAML config.")
    parser.add_argument("--iterations", type=int, default=3, help="Idea iterations to run.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Model used for idea gen, program gen/repair, and eval Q&A (unless overridden).",
    )
    def _parse_optional_float(value: str):
        if value.lower() in ("none", "null"):
            return None
        return float(value)

    parser.add_argument(
        "--temperature",
        type=_parse_optional_float,
        default=0.8,
        help="Temperature for program gen/repair/reasoning and eval Q&A (if supported by the model). Use 'none' to omit.",
    )
    parser.add_argument(
        "--programs-per-idea",
        type=int,
        default=3,
        help="Number of program candidates to generate per idea.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results_idea_loop"),
        help="Directory to store idea/program artifacts.",
    )
    parser.add_argument(
        "--eval-qa-iterations",
        type=int,
        default=3,
        help="Number of iterations for the evaluation Q&A loop.",
    )
    parser.add_argument(
        "--eval-qa-prompt",
        type=Path,
        default=None,
        help="Optional custom prompt payload for the evaluation Q&A loop.",
    )
    parser.add_argument(
        "--eval-qa-temperature",
        type=_parse_optional_float,
        default=None,
        help="Temperature to use for the evaluation Q&A loop (defaults to --temperature). Use 'none' to omit.",
    )
    parser.add_argument(
        "--eval-qa-model",
        type=str,
        default=None,
        help="Override the model for the evaluation Q&A loop (defaults to the main model).",
    )
    parser.add_argument(
        "--idea-prompt",
        type=Path,
        default=None,
        help="Optional custom prompt payload for the idea agent.",
    )
    parser.add_argument(
        "--eval-thinking",
        type=str,
        default="high",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort for eval Q&A and eval reasoning.",
    )
    args = parser.parse_args()

    engine = IdeaProgramEngine(
        args.initial_program,
        args.evaluation_file,
        config_path=args.config,
        results_root=args.results,
        model=args.model,
        temperature=args.temperature,
        programs_per_idea=args.programs_per_idea,
        eval_prompt=args.eval_qa_prompt,
        eval_iterations=args.eval_qa_iterations,
        eval_temperature=args.eval_qa_temperature,
        eval_model=args.eval_qa_model,
        idea_prompt=args.idea_prompt,
        eval_reasoning_effort=args.eval_thinking,
    )
    engine.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
