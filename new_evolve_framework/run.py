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
    args = parser.parse_args()

    engine = IdeaProgramEngine(
        args.initial_program,
        args.evaluation_file,
        config_path=args.config,
        results_root=args.results,
        programs_per_idea=args.programs_per_idea,
    )
    engine.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
