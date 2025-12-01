"""
Idea-first evolution orchestrator.

Self-contained loop that does:
- bootstrap eval of initial program
- generate idea with idea_agent
- generate programs sequentially with prior attempt context
- evaluate (with repair retries) and run reasoning
- snapshot artifacts

Does not use MAP-Elites or the main OpenEvolve controller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import Config, load_config
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler

from rag_codebase.eval_agent.idea_agent import (
    run_idea_refinement,
    IDEA_OUTPUT_PATH,
    EVAL_OUTPUT_PATH,
)
from rag_codebase.eval_agent.evaluation_agent import (
    make_llm_call,
    _message_content_to_text,
)

from .models import IdeaNode, ProgramRecord
from .similarity import IdeaSimilarity

logger = logging.getLogger(__name__)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _format_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _score_from_metrics(metrics: Dict[str, Any]) -> float:
    if "combined_score" in metrics and isinstance(metrics["combined_score"], (int, float)):
        return float(metrics["combined_score"])
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    return float(sum(numeric) / max(1, len(numeric))) if numeric else 0.0


def _extract_code_block(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        without_ticks = stripped[3:]
        newline_idx = without_ticks.find("\n")
        if newline_idx != -1:
            without_ticks = without_ticks[newline_idx + 1 :]
        if without_ticks.endswith("```"):
            without_ticks = without_ticks[: -3]
        return without_ticks.strip()
    return stripped


def _extract_idea_summary(idea_payload: Dict[str, Any]) -> str:
    keys = ("Summary", "summary", "idea", "design", "proposal", "idea_summary")
    for key in keys:
        val = idea_payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return json.dumps(idea_payload, ensure_ascii=True)[:2000]


class IdeaProgramEngine:
    """Idea/program loop manager."""

    def __init__(
        self,
        initial_program_path: Path,
        evaluation_file: Path,
        config_path: Optional[Path] = None,
        *,
        results_root: Path = Path("results_idea_loop"),
        program_gen_prompt: Optional[Path] = None,
        compile_fix_prompt: Optional[Path] = None,
        reasoning_prompt: Optional[Path] = None,
        model: str = "gpt-5.1",
        idea_iterations: int = 2,
        programs_per_idea: int = 3,
        compile_fix_attempts: int = 2,
    ) -> None:
        self.initial_program_path = Path(initial_program_path)
        self.evaluation_file = Path(evaluation_file)
        self.config: Config = load_config(str(config_path) if config_path else None)
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)

        suffix = self.initial_program_path.suffix or ".py"
        self.language = self.config.language or "python"

        # Evaluator setup
        llm_eval_ensemble = LLMEnsemble(self.config.llm.evaluator_models)
        evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        evaluator_prompt_sampler.set_templates("evaluator_system_message")
        self.evaluator = Evaluator(
            self.config.evaluator,
            str(self.evaluation_file),
            llm_eval_ensemble,
            evaluator_prompt_sampler,
            database=None,
            suffix=suffix,
        )

        # LLM prompt assets
        base_prompt_dir = Path(__file__).parent / "prompts"
        self.program_prompt_path = program_gen_prompt or (base_prompt_dir / "program_generation.txt")
        self.compile_fix_prompt_path = compile_fix_prompt or (base_prompt_dir / "compile_fix.txt")
        self.reasoning_prompt_path = reasoning_prompt or (base_prompt_dir / "reasoning.txt")

        # Loop config
        self.model = model
        self.idea_iterations = idea_iterations
        self.programs_per_idea = programs_per_idea
        self.compile_fix_attempts = compile_fix_attempts

        # State
        self.nodes: Dict[str, IdeaNode] = {}
        self.programs: Dict[str, ProgramRecord] = {}
        self.iteration_counter = 0
        embed_model = getattr(self.config.database, "embedding_model", None) or "text-embedding-3-small"
        threshold = getattr(self.config.database, "similarity_threshold", 0.92)
        self.similarity = IdeaSimilarity(model_name=embed_model, threshold=threshold)

    def _save_snapshot(self, iteration_dir: Path, node: IdeaNode, new_programs: List[ProgramRecord]) -> None:
        iteration_dir.mkdir(parents=True, exist_ok=True)
        idea_path = iteration_dir / "idea.json"
        programs_path = iteration_dir / "programs.jsonl"

        idea_path.write_text(json.dumps(asdict(node), indent=2), encoding="utf-8")
        with programs_path.open("w", encoding="utf-8") as outf:
            for p in new_programs:
                outf.write(json.dumps(asdict(p)) + "\n")

        # top-level index for resume
        index_path = self.results_root / "index.json"
        index_payload = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "programs": [asdict(p) for p in self.programs.values()],
            "last_iteration": self.iteration_counter,
        }
        index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    def _run_reasoning(self, metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> str:
        prompt_template = _read_text(self.reasoning_prompt_path)
        prompt = prompt_template.format(
            language=self.language,
            metrics=json.dumps(metrics, ensure_ascii=True),
            summary=artifacts.get("summary") or artifacts.get("Summary") or "",
            runs=artifacts.get("runs") or "",
        )
        resp = make_llm_call(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a concise evaluator for code quality."},
                    {"role": "user", "content": prompt},
                ],
                "reasoning_effort": "low",
            }
        )
        text = _message_content_to_text(resp.choices[0].message.content)
        return text.strip()

    def _format_artifact_context(self, artifacts: Dict[str, Any]) -> str:
        """Extract the most relevant artifact text for repair prompts."""
        keys_of_interest = (
            "build_output",
            "run_exp_stdout",
            "run_exp_stderr",
            "run_exp_output",
            "stderr",
            "traceback",
            "error",
        )
        lines: List[str] = []
        for key in keys_of_interest:
            val = artifacts.get(key)
            if isinstance(val, str) and val.strip():
                trimmed = val.strip()
                if len(trimmed) > 2000:
                    trimmed = trimmed[:2000] + "\n...[truncated]..."
                lines.append(f"{key}: {trimmed}")
        if not lines:
            try:
                return json.dumps(artifacts, ensure_ascii=True)[:2000]
            except Exception:  # pragma: no cover
                return ""
        return "\n".join(lines)

    def _fix_compile(self, code: str, error_text: str, artifacts: Dict[str, Any]) -> str:
        prompt_template = _read_text(self.compile_fix_prompt_path)
        user_prompt = prompt_template.format(
            language=self.language,
            suffix=self.initial_program_path.suffix or ".py",
            code=code,
            error=error_text,
            artifacts=self._format_artifact_context(artifacts),
        )
        resp = make_llm_call(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You fix code so it compiles and runs."},
                    {"role": "user", "content": user_prompt},
                ],
                "reasoning_effort": "medium",
            }
        )
        text = _message_content_to_text(resp.choices[0].message.content)
        return _extract_code_block(text)

    def _prepare_artifacts(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize evaluator artifacts similar to process_parallel expectations."""
        if not isinstance(artifacts, dict):
            return {}

        normalized = dict(artifacts)

        summary = normalized.pop("summary", None) or normalized.get("Summary")
        if summary is not None:
            normalized["summary"] = summary

        runs = normalized.pop("runs", None)
        if runs is not None:
            normalized["runs"] = runs

        debug_path = normalized.get("debug_path")
        if isinstance(debug_path, str) and debug_path.strip():
            try:
                base_debug_dir = Path(debug_path).expanduser().resolve()
                debug_dir = base_debug_dir / "evolve_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                normalized["debug_path"] = str(debug_dir)
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to prepare debug_dir for %s: %s", debug_path, exc)
        return normalized

    def _generate_program_code(
        self,
        node: IdeaNode,
        best_program: ProgramRecord,
        prior_attempts: List[ProgramRecord],
    ) -> str:
        prompt_template = _read_text(self.program_prompt_path)
        snippets = node.idea_payload.get("implementation_snippets") if node.idea_payload else []
        snippet_text = json.dumps(snippets, ensure_ascii=True) if snippets else "[]"

        attempts_fmt: List[str] = []
        for rec in reversed(prior_attempts[-3:]):
            note = rec.reasoning or rec.error_text or "n/a"
            attempts_fmt.append(
                f"- status: {rec.status} | metrics: {_format_metrics(rec.metrics)} | note: {note}"
            )
        prior_attempts_text = "\n".join(attempts_fmt) if attempts_fmt else "none"

        prompt = prompt_template.format(
            language=self.language,
            suffix=self.initial_program_path.suffix or ".py",
            idea_summary=node.idea_summary,
            snippets=snippet_text,
            best_metrics=_format_metrics(best_program.metrics),
            best_reasoning_present=bool(best_program.reasoning),
            prior_attempts=prior_attempts_text,
        )

        resp = make_llm_call(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You write full, working programs."},
                    {"role": "user", "content": prompt},
                ],
                "reasoning_effort": "medium",
            }
        )
        text = _message_content_to_text(resp.choices[0].message.content)
        return _extract_code_block(text)

    async def _evaluate_with_repair(
        self,
        code: str,
        parent_program_id: Optional[str],
        generation_order: int,
    ) -> ProgramRecord:
        working_code = code
        last_metrics: Optional[Dict[str, Any]] = None
        last_artifacts: Dict[str, Any] = {}
        last_error_text = ""

        for attempt in range(self.compile_fix_attempts + 1):
            program_id = os.urandom(8).hex()
            artifacts: Dict[str, Any] = {}
            try:
                metrics = await self.evaluator.evaluate_program(working_code, program_id)
                raw_artifacts = self.evaluator.get_pending_artifacts(program_id) or {}
                artifacts = self._prepare_artifacts(raw_artifacts)
                last_metrics = metrics
                last_artifacts = artifacts
                success = metrics.get("run_success") == 1 or metrics.get("success") == 1 or not metrics.get("error")
                if success:
                    reasoning = self._run_reasoning(metrics, artifacts)
                    return ProgramRecord.create(
                        status="success",
                        code=working_code,
                        metrics=metrics,
                        reasoning=reasoning,
                        eval_artifacts=artifacts,
                        generation_order=generation_order,
                        parent_program_id=parent_program_id,
                    )
                error_text = (
                    artifacts.get("stderr")
                    or artifacts.get("build_output")
                    or artifacts.get("run_exp_stderr")
                    or json.dumps(metrics, ensure_ascii=True)
                )
                last_error_text = error_text
            except Exception as exc:  # pragma: no cover
                last_error_text = str(exc)
                last_artifacts = artifacts

            if attempt >= self.compile_fix_attempts:
                logger.warning("Compile/run failed after retries: %s", last_error_text)
                fail_metrics = (
                    last_metrics
                    if last_metrics is not None
                    else {"combined_score": float("-inf"), "run_success": 0}
                )
                return ProgramRecord.create(
                    status="failed",
                    code=working_code,
                    metrics=fail_metrics,
                    error_text=last_error_text,
                    eval_artifacts=last_artifacts,
                    generation_order=generation_order,
                    parent_program_id=parent_program_id,
                )

            working_code = self._fix_compile(working_code, last_error_text, last_artifacts)

        # Fallback (should not hit)
        return ProgramRecord.create(
            status="failed",
            code=working_code,
            metrics={"combined_score": float("-inf"), "run_success": 0},
            error_text=last_error_text or "Unknown failure",
            eval_artifacts=last_artifacts,
            generation_order=generation_order,
            parent_program_id=parent_program_id,
        )

    def _bootstrap_first_node(self) -> IdeaNode:
        base_code = self.initial_program_path.read_text(encoding="utf-8")
        logger.info("Bootstrapping with initial program at %s", self.initial_program_path)
        record = asyncio.run(self._evaluate_with_repair(base_code, None, generation_order=0))
        if record is None:
            raise RuntimeError("Unable to evaluate initial program successfully.")

        self.programs[record.id] = record
        node = IdeaNode.create(idea_payload=None, idea_summary="Bootstrap node")
        node.programs.append(record.id)
        node.update_best(record.id, _score_from_metrics(record.metrics))
        self.nodes[node.id] = node
        return node

    def _generate_new_idea(self, parent_node: IdeaNode) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not parent_node.best_program_id:
            raise RuntimeError("Parent node has no best program.")
        best_program = self.programs[parent_node.best_program_id]
        perf_summary = f"Metrics: {_format_metrics(best_program.metrics)}\nReasoning: {best_program.reasoning or 'n/a'}"
        idea_iters = run_idea_refinement(
            perf_summary,
            perf_summary,
            iteration_count=self.idea_iterations,
        )
        final_payload = idea_iters[-1]["idea_payload"]
        return final_payload, idea_iters

    def _similar_to_existing(self, embedding: List[float]) -> bool:
        for node in self.nodes.values():
            if node.embedding and self.similarity.is_too_similar(embedding, node.embedding):
                return True
        return False

    def _select_parent_node(self) -> IdeaNode:
        return max(
            self.nodes.values(),
            key=lambda n: n.best_score or float("-inf"),
        )

    def _build_node_from_idea(
        self, idea_payload: Dict[str, Any], idea_embedding: List[float], idea_iters: List[Dict[str, Any]]
    ) -> IdeaNode:
        idea_summary = _extract_idea_summary(idea_payload)
        new_node = IdeaNode.create(
            idea_payload=idea_payload,
            idea_summary=idea_summary,
            artifacts_dir=str(self.results_root / f"iteration_{self.iteration_counter}"),
        )
        new_node.embedding = idea_embedding
        new_node.idea_iterations = idea_iters
        return new_node

    def _generate_programs_for_node(
        self,
        node: IdeaNode,
        parent_best: ProgramRecord,
    ) -> List[ProgramRecord]:
        prior_attempts: List[ProgramRecord] = []
        generated: List[ProgramRecord] = []
        for idx in range(self.programs_per_idea):
            code = self._generate_program_code(node, parent_best, prior_attempts or [parent_best])
            record = asyncio.run(
                self._evaluate_with_repair(code, parent_best.id if parent_best else None, idx)
            )
            if record:
                self.programs[record.id] = record
                node.programs.append(record.id)
                prior_attempts.append(record)
                generated.append(record)
        return generated

    def _copy_exp_markers(self, generated: List[ProgramRecord], iteration_dir: Path) -> None:
        for rec in generated:
            dbg = rec.eval_artifacts.get("debug_path") if isinstance(rec.eval_artifacts, dict) else None
            if not dbg:
                continue
            dbg_path = Path(dbg)
            try:
                exp_dir = None
                for ancestor in dbg_path.parents:
                    if "exp" in ancestor.name.lower():
                        exp_dir = ancestor
                        break
                if exp_dir and exp_dir.exists():
                    target = iteration_dir / exp_dir.name
                    if exp_dir.is_dir():
                        if not target.exists():
                            shutil.copytree(exp_dir, target, dirs_exist_ok=True)
                    else:
                        if not target.exists():
                            shutil.copy2(exp_dir, target)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to copy exp marker from %s: %s", dbg, exc)

    def _copy_rag_outputs(self, iteration_dir: Path) -> None:
        """Copy idea_agent/evaluation_agent outputs into iteration folder if present."""
        for source in (IDEA_OUTPUT_PATH, EVAL_OUTPUT_PATH):
            try:
                if source.exists():
                    target = iteration_dir / source.name
                    shutil.copy2(source, target)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to copy %s: %s", source, exc)

    def run(self, iterations: int = 3) -> None:
        if not self.nodes:
            self._bootstrap_first_node()

        for _ in range(iterations):
            self.iteration_counter += 1
            parent_node = self._select_parent_node()
            parent_best = self.programs[parent_node.best_program_id] if parent_node.best_program_id else None
            if parent_best is None:
                raise RuntimeError("Best program missing for parent node.")

            idea_payload, idea_iters = self._generate_new_idea(parent_node)
            idea_summary = _extract_idea_summary(idea_payload)
            idea_embedding = self.similarity.embed(idea_summary)
            if self._similar_to_existing(idea_embedding):
                logger.info("New idea is too similar; reusing summary for logging but continuing anyway.")

            new_node = self._build_node_from_idea(idea_payload, idea_embedding, idea_iters)

            generated = self._generate_programs_for_node(new_node, parent_best)

            successful = [rec for rec in generated if rec.status == "success"]
            if successful:
                best = max(successful, key=lambda r: _score_from_metrics(r.metrics))
                new_node.update_best(best.id, _score_from_metrics(best.metrics))
            else:
                new_node.programs.append(parent_best.id)
                new_node.update_best(parent_best.id, _score_from_metrics(parent_best.metrics))

            self.nodes[new_node.id] = new_node

            iteration_dir = self.results_root / f"iteration_{self.iteration_counter}"
            self._save_snapshot(iteration_dir, new_node, generated)
            self._copy_exp_markers(generated, iteration_dir)
            self._copy_rag_outputs(iteration_dir)

            logger.info(
                "Iteration %s complete. Programs: %s, best score=%s",
                self.iteration_counter,
                len(generated),
                new_node.best_score,
            )
