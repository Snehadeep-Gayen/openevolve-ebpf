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

import yaml
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
    run_evaluation_qa_loop,
)

from .models import IdeaNode, ProgramRecord
from .similarity import IdeaSimilarity

logger = logging.getLogger(__name__)


class LiteralStr(str):
    """YAML literal block string."""


def _repr_literal_str(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


# Ensure safe_dump uses literal blocks for LiteralStr values.
yaml.add_representer(LiteralStr, _repr_literal_str)
yaml.add_representer(LiteralStr, _repr_literal_str, Dumper=yaml.SafeDumper)


def _literal(value: Optional[str]) -> Optional[LiteralStr]:
    return LiteralStr(value) if value is not None else None


def _literalize(obj: Any) -> Any:
    """Recursively wrap strings as LiteralStr for block-style YAML output."""
    if isinstance(obj, str):
        return LiteralStr(obj)
    if isinstance(obj, list):
        return [_literalize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _literalize(v) for k, v in obj.items()}
    return obj


def _supports_reasoning(model_name: str) -> bool:
    """Return True if model likely supports reasoning_effort."""

    if not model_name:
        return False
    lowered = model_name.lower()
    return lowered.startswith(("o3", "o1", "gpt-5"))


def _supports_temperature(model_name: str) -> bool:
    """Return True if model likely supports custom temperature."""

    if not model_name:
        return False
    lowered = model_name.lower()
    # Known non-temperature models: some o3 variants require default only; gpt-5.1 treated similarly here
    if lowered.startswith("o3") or lowered.startswith("gpt-5.1"):
        return False
    return True


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
        generation_feedback_prompt: Optional[Path] = None,
        model: str = "gpt-5.1",
        temperature: Optional[float] = 0.8,
        idea_iterations: int = 2,
        programs_per_idea: int = 3,
        compile_fix_attempts: int = 2,
        eval_prompt: Optional[Path] = None,
        eval_iterations: int = 3,
        eval_temperature: Optional[float] = 0.8,
        eval_model: Optional[str] = None,
        idea_prompt: Optional[Path] = None,
        eval_reasoning_effort: str = "high",
    ) -> None:
        self.initial_program_path = Path(initial_program_path)
        self.evaluation_file = Path(evaluation_file)
        self.config: Config = load_config(str(config_path) if config_path else None)
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)

        suffix = self.initial_program_path.suffix or ".c"
        self.language = self.config.language or "c"

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
        self.generation_feedback_prompt_path = (
            generation_feedback_prompt or (base_prompt_dir / "generation_feedback.txt")
        )
        self.eval_prompt_base_path = Path(eval_prompt) if eval_prompt else (base_prompt_dir / "eval_agent_prompt_base.txt")
        self.idea_prompt_base_path = Path(idea_prompt) if idea_prompt else (base_prompt_dir / "idea_agent_prompt_base.txt")
        self.eval_prompt_base_text = _read_text(self.eval_prompt_base_path)
        self.idea_prompt_base_text = _read_text(self.idea_prompt_base_path)

        # Loop config
        self.model = model
        self.temperature = temperature if _supports_temperature(model) else None
        self.model_supports_reasoning = _supports_reasoning(model)
        self.idea_iterations = idea_iterations
        self.programs_per_idea = programs_per_idea
        self.compile_fix_attempts = compile_fix_attempts
        self.eval_iterations = eval_iterations
        self.eval_temperature = eval_temperature if eval_temperature is not None else temperature
        self.eval_model = eval_model or model
        self.eval_model_supports_reasoning = _supports_reasoning(self.eval_model)
        self.idea_prompt_path: Optional[Path] = None
        self.eval_reasoning_effort = eval_reasoning_effort
        self.rejected_ideas: List[str] = []

        # State
        self.nodes: Dict[str, IdeaNode] = {}
        self.programs: Dict[str, ProgramRecord] = {}
        self.iteration_counter = 0
        self.eval_context: Optional[Dict[str, Any]] = None
        self.eval_output_path: Optional[Path] = None
        embed_model = getattr(self.config.database, "embedding_model", None) or "text-embedding-3-small"
        threshold = getattr(self.config.database, "similarity_threshold", 0.92)
        self.similarity = IdeaSimilarity(model_name=embed_model, threshold=threshold)

    def _save_snapshot(self, iteration_dir: Path, node: IdeaNode, new_programs: List[ProgramRecord]) -> None:
        iteration_dir.mkdir(parents=True, exist_ok=True)
        idea_path = iteration_dir / "idea.json"
        programs_path = iteration_dir / "programs.jsonl"

        node_dict = asdict(node)
        node_dict.pop("embedding", None)
        idea_path.write_text(json.dumps(node_dict, indent=2), encoding="utf-8")
        with programs_path.open("w", encoding="utf-8") as outf:
            for p in new_programs:
                outf.write(json.dumps(asdict(p)) + "\n")
                # Also write per-generation program record into its gen directory if available
                gen_dir = None
                if isinstance(p.eval_artifacts, dict):
                    gen_dir = p.eval_artifacts.get("program_generation_dir")
                if gen_dir:
                    try:
                        gen_dir_path = Path(gen_dir)
                        gen_dir_path.mkdir(parents=True, exist_ok=True)
                        gen_record_path = gen_dir_path / "program_record.json"
                        gen_record_path.write_text(json.dumps(asdict(p), indent=2), encoding="utf-8")
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Failed to write per-gen program record to %s: %s", gen_dir, exc)

        # top-level index for resume
        index_path = self.results_root / "index.yaml"
        index_payload = {
            "nodes": [
                {k: v for k, v in asdict(n).items() if k != "embedding"} for n in self.nodes.values()
            ],
            "programs": [asdict(p) for p in self.programs.values()],
            "last_iteration": self.iteration_counter,
        }
        index_path.write_text(
            yaml.safe_dump(index_payload, sort_keys=False, width=120, default_flow_style=False),
            encoding="utf-8",
        )

        # compact view of ideas + best programs
        clean_nodes: List[Dict[str, Any]] = []
        for n in self.nodes.values():
            best_program = self.programs.get(n.best_program_id) if n.best_program_id else None
            clean_nodes.append(
                {
                    "id": n.id,
                    "idea_summary": n.idea_summary,
                    "idea_payload": n.idea_payload,
                    "parent_id": n.parent_id,
                    "eval_summary": n.eval_summary,
                    "eval_hypothesis": n.eval_hypothesis,
                    "eval_score": n.eval_score,
                    "eval_output_path": n.eval_output_path,
                    "best_program": None
                    if not best_program
                    else {
                        "id": best_program.id,
                        "metrics": best_program.metrics,
                        "reasoning": best_program.reasoning,
                        "code": best_program.code,
                    },
                }
            )
        clean_nodes_path = self.results_root / "nodes_clean.yaml"
        clean_nodes_payload = {"nodes": clean_nodes, "last_iteration": self.iteration_counter}
        clean_nodes_path.write_text(
            yaml.safe_dump(clean_nodes_payload, sort_keys=False, width=120, default_flow_style=False),
            encoding="utf-8",
        )

    def _build_prompt_context(
        self,
        best_program: ProgramRecord,
        eval_context: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        context = {
            "best_metrics": best_program.metrics,
            "best_reasoning": best_program.reasoning,
            "eval_summary": (eval_context or {}).get("summary"),
            "eval_hypothesis": (eval_context or {}).get("hypothesis"),
            "eval_score": (eval_context or {}).get("score"),
        }

        lines = []
        if context["eval_summary"]:
            lines.append(f"Evaluation summary: {context['eval_summary']}")
        if context["eval_hypothesis"]:
            lines.append(f"Evaluation hypothesis: {context['eval_hypothesis']}")
        if context["best_metrics"]:
            lines.append(f"Best program metrics: {_format_metrics(context['best_metrics'])}")
        if context["best_reasoning"]:
            lines.append(f"Best program reasoning: {context['best_reasoning']}")
        if not lines:
            lines.append("No prior evaluation context available.")
        return context, "\n".join(lines)

    def _write_eval_prompt(
        self,
        iteration_dir: Path,
        best_program: ProgramRecord,
        eval_context: Optional[Dict[str, Any]],
    ) -> Path:
        """Create per-run prompt payload for the evaluation Q&A agent."""

        context_dict, context_text = self._build_prompt_context(best_program, eval_context)
        code_summary_block = getattr(self.config, "code_summary", None)
        runs_text = ""
        if isinstance(best_program.eval_artifacts, dict) and best_program.eval_artifacts.get("runs"):
            runs_text = json.dumps(best_program.eval_artifacts.get("runs"), indent=2)
        eval_prompt_text = (
            f"{self.eval_prompt_base_text.rstrip()}\n\n"
            f"Run context:\n{context_text}\n\n"
            f"Codebase summary:\n{code_summary_block or 'N/A'}\n\n"
            f"Run artifacts (runs):\n{runs_text or 'N/A'}\n\n"
            f"Best program code:\n```{self.initial_program_path.suffix or '.py'}\n{best_program.code}\n```"
        )
        eval_payload = {
            "eval_agent_prompt": _literal(eval_prompt_text),
            "run_context": context_dict,
            "best_program_code": _literal(best_program.code),
        }
        eval_prompt_path = iteration_dir / "eval_prompt.yaml"
        eval_prompt_path.write_text(
            yaml.safe_dump(
                _literalize(eval_payload),
                sort_keys=False,
                width=120,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        return eval_prompt_path

    def _write_idea_prompt(
        self,
        iteration_dir: Path,
        best_program: ProgramRecord,
        eval_context: Optional[Dict[str, Any]],
    ) -> Path:
        """Create per-run prompt payload for the idea agent."""

        context_dict, context_text = self._build_prompt_context(best_program, eval_context)
        code_summary_block = getattr(self.config, "code_summary", None)
        forbidden_block = "\n".join([f"- {s}" for s in self.rejected_ideas]) if self.rejected_ideas else "None"
        idea_prompt_text = (
            f"{self.idea_prompt_base_text.rstrip()}\n\n"
            f"Run context:\n{context_text}\n\n"
            f"Codebase summary:\n{code_summary_block or 'N/A'}\n\n"
            f"Forbidden/rejected ideas so far:\n{forbidden_block}\n\n"
            f"Best program code:\n```{self.initial_program_path.suffix or '.py'}\n{best_program.code}\n```"
        )
        idea_payload = {
            "idea_agent_prompt": _literal(idea_prompt_text),
            "run_context": context_dict,
            "best_program_code": _literal(best_program.code),
        }
        idea_prompt_path = iteration_dir / "idea_prompt.yaml"
        idea_prompt_path.write_text(
            yaml.safe_dump(
                _literalize(idea_payload),
                sort_keys=False,
                width=120,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        return idea_prompt_path

    def _program_reasoning(self) -> Optional[str]:
        """Attach concise reasoning from eval Q&A context if available."""
        if not self.eval_context:
            return None
        summary = self.eval_context.get("summary") or ""
        hypothesis = self.eval_context.get("hypothesis") or ""
        parts = []
        if summary:
            parts.append(f"Eval summary: {summary}")
        if hypothesis:
            parts.append(f"Hypothesis: {hypothesis}")
        return " | ".join(parts) if parts else None

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
            "runs",
        )
        lines: List[str] = []
        for key in keys_of_interest:
            val = artifacts.get(key)
            if isinstance(val, str) and val.strip():
                trimmed = val.strip()
                if len(trimmed) > 2000:
                    trimmed = trimmed[:2000] + "\n...[truncated]..."
                lines.append(f"{key}: {trimmed}")
            elif key == "runs" and val is not None:
                try:
                    blob = json.dumps(val, indent=2)[:2000]
                    lines.append(f"{key}: {blob}")
                except Exception:
                    continue
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You fix code so it compiles and runs."},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None and _supports_temperature(self.model):
            payload["temperature"] = self.temperature
        if self.model_supports_reasoning:
            payload["reasoning_effort"] = "medium"
        resp = make_llm_call(payload)
        text = _message_content_to_text(resp.choices[0].message.content)
        return _extract_code_block(text)

    def _prepare_artifacts(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize evaluator artifacts similar to process_parallel expectations."""
        if not isinstance(artifacts, dict):
            return {}

        normalized = dict(artifacts)
        raw_debug_path = normalized.get("debug_path")

        summary = normalized.pop("summary", None) or normalized.get("Summary")
        if summary is not None:
            normalized["summary"] = summary

        runs = normalized.pop("runs", None)
        if runs is not None:
            normalized["runs"] = runs

        debug_path = normalized.get("debug_path")
        if isinstance(debug_path, str) and debug_path.strip():
            try:
                normalized["raw_debug_path"] = debug_path
                base_debug_dir = Path(debug_path).expanduser().resolve()
                debug_dir = base_debug_dir / "evolve_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                normalized["debug_path"] = str(debug_dir)
                logger.info("Prepared debug dir: raw=%s -> evolve_debug=%s", base_debug_dir, debug_dir)
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to prepare debug_dir for %s: %s", debug_path, exc)
        return normalized

    def _log_program_generation(
        self,
        iteration_dir: Path,
        generation_order: int,
        payload: Dict[str, Any],
        response_text: str,
        extracted_code: str,
        gen_dir: Path,
    ) -> None:
        """Persist the program-generation prompt/response and extracted code for debugging."""
        out_dir = gen_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "gen.yaml"
        code_path = out_dir / f"program{self.initial_program_path.suffix or '.txt'}"

        messages = []
        for msg in payload.get("messages", []):
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                new_msg = dict(msg)
                new_msg["content"] = _literal(msg["content"])
                messages.append(new_msg)
            else:
                messages.append(msg)
        log_payload = {
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "messages": messages,
            "response_text": _literal(response_text),
            "extracted_code": _literal(extracted_code),
        }
        log_path.write_text(
            yaml.safe_dump(
                _literalize(log_payload),
                sort_keys=False,
                width=120,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        code_path.write_text(extracted_code, encoding="utf-8")

    def _log_generation_feedback(
        self,
        gen_dir: Path,
        payload: Dict[str, Any],
        response_text: str,
    ) -> None:
        """Persist the per-attempt feedback prompt/response."""

        gen_dir.mkdir(parents=True, exist_ok=True)
        log_path = gen_dir / "feedback.yaml"
        messages = []
        for msg in payload.get("messages", []):
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                new_msg = dict(msg)
                new_msg["content"] = _literal(msg["content"])
                messages.append(new_msg)
            else:
                messages.append(msg)
        log_payload = {
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "messages": messages,
            "response_text": _literal(response_text),
        }
        log_path.write_text(
            yaml.safe_dump(
                _literalize(log_payload),
                sort_keys=False,
                width=120,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

    def _generate_program_code(
        self,
        node: IdeaNode,
        best_program: ProgramRecord,
        prior_attempts: List[ProgramRecord],
        iteration_dir: Path,
        generation_order: int,
        feedback_history: Optional[List[str]] = None,
    ) -> Tuple[str, Path]:
        prompt_template = _read_text(self.program_prompt_path)
        code_summary_block = getattr(self.config, "code_summary", None) or "N/A"
        snippets = node.idea_payload.get("implementation_snippets") if node.idea_payload else []
        snippet_text = json.dumps(snippets, ensure_ascii=True) if snippets else "[]"
        feedback_history = feedback_history or []
        feedback_fmt = []
        for idx, fb in enumerate(reversed(feedback_history[-3:]), start=1):
            feedback_fmt.append(f"- Feedback {idx} (latest first): {fb}")
        feedback_text = "\n".join(feedback_fmt) if feedback_fmt else "none"

        attempts_fmt: List[str] = []
        for rec in reversed(prior_attempts[-3:]):
            note = rec.reasoning or rec.error_text or "n/a"
            attempts_fmt.append(
                "\n".join(
                    [
                        f"- status: {rec.status} | metrics: {_format_metrics(rec.metrics)} | note: {note}",
                        "  code:",
                        f"  ```{self.language}",
                        *[f"  {line}" for line in rec.code.splitlines()],
                        "  ```",
                    ]
                )
            )
        prior_attempts_text = "\n".join(attempts_fmt) if attempts_fmt else "none"

        best_program_block = ""
        if generation_order == 0 and best_program.code:
            best_program_block_lines = [
                f"```{self.language}",
                *best_program.code.splitlines(),
                "```",
            ]
            best_program_block = "\n".join(best_program_block_lines)
        else:
            best_program_block = "Omitted for this attempt (best program code shown on first generation only)."

        prompt = prompt_template.format(
            language=self.language,
            suffix=self.initial_program_path.suffix or ".py",
            idea_summary=node.idea_summary,
            snippets=snippet_text,
            best_metrics=_format_metrics(best_program.metrics),
            best_reasoning_present=bool(best_program.reasoning),
            prior_attempts=prior_attempts_text,
            best_program_code=best_program_block,
            code_summary=code_summary_block,
            feedback_history=feedback_text,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You write full, working programs."},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.model_supports_reasoning:
            payload["reasoning_effort"] = "medium"
        resp = make_llm_call(payload)
        text = _message_content_to_text(resp.choices[0].message.content)
        code = _extract_code_block(text)
        out_dir = iteration_dir / "program_generations" / f"gen_{generation_order}"
        self._log_program_generation(iteration_dir, generation_order, payload, text, code, out_dir)
        return code, out_dir

    def _run_generation_feedback(
        self,
        node: IdeaNode,
        program_record: ProgramRecord,
        eval_context: Optional[Dict[str, Any]],
        feedback_history: List[str],
        gen_dir: Path,
    ) -> Optional[str]:
        """Run a one-shot feedback LLM to guide the next generation."""

        prompt_template = _read_text(self.generation_feedback_prompt_path)
        metrics_text = _format_metrics(program_record.metrics) if program_record.metrics else "n/a"
        artifacts = program_record.eval_artifacts if isinstance(program_record.eval_artifacts, dict) else {}
        if "runs" not in artifacts:
            logger.warning("Generation feedback: 'runs' missing in artifacts for gen %s", program_record.generation_order)
        artifacts_text = self._format_artifact_context(artifacts)
        prior_fb = "\n".join([f"- {fb}" for fb in feedback_history[-3:]]) if feedback_history else "none"
        code_summary_block = getattr(self.config, "code_summary", None) or "N/A"
        parent_eval_output = ""
        if node.parent_id:
            parent_node = self.nodes.get(node.parent_id)
            if parent_node and parent_node.eval_output_path:
                try:
                    parent_eval_output = Path(parent_node.eval_output_path).read_text(encoding="utf-8")
                except Exception:
                    parent_eval_output = ""
        prompt = prompt_template.format(
            idea_summary=node.idea_summary,
            eval_summary=(eval_context or {}).get("summary") or "n/a",
            eval_hypothesis=(eval_context or {}).get("hypothesis") or "n/a",
            eval_score=(eval_context or {}).get("score") if eval_context else "n/a",
            parent_eval_output=parent_eval_output or "n/a",
            metrics=metrics_text or "n/a",
            artifacts=artifacts_text or "n/a",
            prior_feedback=prior_fb,
            code=program_record.code,
            suffix=self.initial_program_path.suffix or ".py",
            code_summary=code_summary_block,
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You review eBPF attempts and suggest the next fix."},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None and _supports_temperature(self.model):
            payload["temperature"] = self.temperature
        if self.model_supports_reasoning:
            payload["reasoning_effort"] = "medium"

        try:
            resp = make_llm_call(payload)
            text = _message_content_to_text(resp.choices[0].message.content)
            self._log_generation_feedback(gen_dir, payload, text)
            return text
        except Exception as exc:  # pragma: no cover
            logger.warning("Generation feedback call failed; continuing without it: %s", exc)
            return None

    async def _evaluate_with_repair(
        self,
        code: str,
        parent_program_id: Optional[str],
        generation_order: int,
        gen_dir: Optional[Path] = None,
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
                if gen_dir:
                    artifacts["program_generation_dir"] = str(gen_dir)
                last_metrics = metrics
                last_artifacts = artifacts
                success = metrics.get("run_success") == 1 or metrics.get("success") == 1 or not metrics.get("error")
                if success:
                    logger.info("Program generation %s succeeded on attempt %s", generation_order, attempt + 1)
                    reasoning = None
                    return ProgramRecord.create(
                        status="success",
                        code=working_code,
                        metrics=metrics,
                        reasoning=reasoning,
                        eval_artifacts=artifacts,
                        generation_order=generation_order,
                        parent_program_id=parent_program_id,
                        compile_attempts=attempt + 1,
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

            logger.info("Program generation %s failed attempt %s; entering repair", generation_order, attempt + 1)
            if attempt >= self.compile_fix_attempts:
                logger.warning("Compile/run failed after retries: %s", last_error_text)
                fail_metrics = (
                    last_metrics
                    if last_metrics is not None
                    else {"combined_score": float("-inf"), "run_success": 0}
                )
                if gen_dir:
                    last_artifacts["program_generation_dir"] = str(gen_dir)
                return ProgramRecord.create(
                    status="failed",
                    code=working_code,
                    metrics=fail_metrics,
                    error_text=last_error_text,
                    eval_artifacts=last_artifacts,
                    generation_order=generation_order,
                    parent_program_id=parent_program_id,
                    compile_attempts=attempt + 1,
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
        record = asyncio.run(self._evaluate_with_repair(base_code, None, generation_order=0, gen_dir=None))
        if record is None:
            raise RuntimeError("Unable to evaluate initial program successfully.")

        self.programs[record.id] = record
        node = IdeaNode.create(
            idea_payload=None,
            idea_summary="Bootstrap node",
            parent_id=None,
            artifacts_dir=self.results_root / "bootstrap",
        )
        node.programs.append(record.id)
        node.update_best(record.id, _score_from_metrics(record.metrics))
        self.nodes[node.id] = node

        # Persist bootstrap artifacts similar to an iteration
        bootstrap_dir = self.results_root / "bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        # Write programs.jsonl with the bootstrap record
        programs_path = bootstrap_dir / "programs.jsonl"
        with programs_path.open("w", encoding="utf-8") as outf:
            outf.write(json.dumps(asdict(record)) + "\n")
        # Copy any exp-* artifacts from the bootstrap evaluation
        self._copy_exp_markers([record], bootstrap_dir)

        return node

    def _run_eval_qa(
        self,
        prompt_path: Path,
        parent_node: IdeaNode,
        output_dir: Path,
        mirror_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the RAG Q&A loop for the given parent node and persist results on the node."""

        output_dir.mkdir(parents=True, exist_ok=True)
        # If this node already has eval context, reuse it and copy artifacts forward.
        if parent_node.eval_summary or parent_node.eval_hypothesis or parent_node.eval_score is not None:
            ctx = {
                "summary": parent_node.eval_summary,
                "hypothesis": parent_node.eval_hypothesis,
                "score": parent_node.eval_score,
            }
            if parent_node.eval_output_path:
                try:
                    src = Path(parent_node.eval_output_path)
                    if src.exists():
                        dst = output_dir / "eval_output.json"
                        if dst.resolve() != src.resolve():
                            shutil.copy2(src, dst)
                        self.eval_output_path = dst
                        # Also mirror to iteration folder if requested
                        if mirror_dir:
                            mirror_dir.mkdir(parents=True, exist_ok=True)
                            mirror_dst = mirror_dir / "eval_output.json"
                            if mirror_dst.resolve() != dst.resolve():
                                shutil.copy2(dst, mirror_dst)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to copy cached eval_output.json: %s", exc)
            self.eval_context = ctx
            return ctx

        output_path = output_dir / "eval_output.json"
        try:
            result = run_evaluation_qa_loop(
                iteration_count=self.eval_iterations,
                model=self.eval_model,
                temperature=self.eval_temperature if _supports_temperature(self.eval_model) else None,
                prompt_path=prompt_path,
                output_path=output_path,
                reasoning_effort=self.eval_reasoning_effort if self.eval_model_supports_reasoning else None,
            )
            self.eval_output_path = Path(result.get("output_path") or output_path)
            ctx = {
                "summary": result.get("final_summary"),
                "hypothesis": result.get("final_hypothesis"),
                "score": result.get("final_hypothesis_score"),
            }
            parent_node.eval_summary = ctx["summary"]
            parent_node.eval_hypothesis = ctx["hypothesis"]
            parent_node.eval_score = ctx["score"]
            parent_node.eval_output_path = str(self.eval_output_path) if self.eval_output_path else None
            try:
                if mirror_dir:
                    mirror_dir.mkdir(parents=True, exist_ok=True)
                    mirror_dst = mirror_dir / output_path.name
                    if mirror_dst.resolve() != output_path.resolve():
                        shutil.copy2(output_path, mirror_dst)
                if (
                    self.eval_output_path
                    and self.eval_output_path.exists()
                    and self.eval_output_path.resolve() != EVAL_OUTPUT_PATH.resolve()
                ):
                    shutil.copy2(self.eval_output_path, EVAL_OUTPUT_PATH)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to mirror eval_output.json to default location: %s", exc)
            self.eval_context = ctx
            return ctx
        except Exception as exc:  # pragma: no cover
            logger.warning("Evaluation QA loop failed; continuing without it: %s", exc)
            self.eval_context = None
            return None

    def _generate_new_idea(
        self,
        parent_node: IdeaNode,
        eval_context: Optional[Dict[str, Any]] = None,
        prompt_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not parent_node.best_program_id:
            raise RuntimeError("Parent node has no best program.")
        best_program = self.programs[parent_node.best_program_id]
        perf_summary = f"Metrics: {_format_metrics(best_program.metrics)}\nReasoning: {best_program.reasoning or 'n/a'}"
        idea_iters = run_idea_refinement(
            (eval_context or {}).get("hypothesis") or perf_summary,
            (eval_context or {}).get("summary") or perf_summary,
            iteration_count=self.idea_iterations,
            prompt_payload_path=prompt_path,
            model=self.model,
            reasoning_effort=self.eval_reasoning_effort if self.model_supports_reasoning else None,
            temperature=self.temperature,
        )
        final_payload = idea_iters[-1]["idea_payload"]
        return final_payload, idea_iters

    def _similar_to_existing(self, embedding: List[float]) -> bool:
        for node in self.nodes.values():
            if node.embedding and self.similarity.is_too_similar(embedding, node.embedding):
                logger.info("Similarity check: candidate is too close to node %s", node.id)
                return True
        logger.info("Similarity check: candidate accepted (no close matches)")
        return False

    def _is_novel_llm(self, idea_summary: str) -> Tuple[bool, Optional[str]]:
        """LLM-based novelty check against all prior idea summaries; returns (is_novel, explanation)."""

        if not self.nodes:
            return True, None

        prior_summaries = [n.idea_summary for n in self.nodes.values() if n.idea_summary]
        if not prior_summaries:
            return True, None

        existing_block = "\n".join([f"{idx+1}. {s}" for idx, s in enumerate(prior_summaries)])
        prompt = (
            "You are a strict novelty judge. Given a candidate idea and a list of prior ideas, "
            "decide if the candidate is genuinely novel (not overlapping or restating prior ideas). Focus on the architecture of the idea itself for the novelty, not the goals. A novel idea should lead to different system behavior."
            "First state 'novel' or 'not novel', then a one-line explanation why.\n\n"
            f"Prior ideas:\n{existing_block}\n\n"
            f"Candidate idea:\n{idea_summary}\n\n"
            "Answer with a single line: '<novel|not novel> — <reason>'."
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You judge novelty of ideas."},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None and _supports_temperature(self.model):
            payload["temperature"] = self.temperature
        if self.model_supports_reasoning:
            payload["reasoning_effort"] = "low"

        try:
            resp = make_llm_call(payload)
            raw = _message_content_to_text(resp.choices[0].message.content).strip()
            text = raw.lower()
            is_novel = text.startswith("novel")
            logger.info(
                "Novelty check via LLM: %s (response=%s)",
                "novel" if is_novel else "not novel",
                raw,
            )
            return is_novel, raw
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM novelty check failed; treating as novel. Error: %s", exc)
            return True, None

    def _select_parent_node(self) -> IdeaNode:
        return max(
            self.nodes.values(),
            key=lambda n: n.best_score or float("-inf"),
        )

    def _build_node_from_idea(
        self,
        idea_payload: Dict[str, Any],
        idea_embedding: List[float],
        idea_iters: List[Dict[str, Any]],
        parent_id: Optional[str],
    ) -> IdeaNode:
        idea_summary = _extract_idea_summary(idea_payload)
        new_node = IdeaNode.create(
            idea_payload=idea_payload,
            idea_summary=idea_summary,
            parent_id=parent_id,
            artifacts_dir=str(self.results_root / f"iteration_{self.iteration_counter}"),
        )
        new_node.embedding = idea_embedding
        new_node.idea_iterations = idea_iters
        return new_node

    def _generate_programs_for_node(
        self,
        node: IdeaNode,
        parent_best: ProgramRecord,
        iteration_dir: Path,
        eval_context: Optional[Dict[str, Any]],
    ) -> List[ProgramRecord]:
        prior_attempts: List[ProgramRecord] = []
        generated: List[ProgramRecord] = []
        feedback_history: List[str] = []
        for idx in range(self.programs_per_idea):
            code, gen_dir = self._generate_program_code(
                node,
                parent_best,
                prior_attempts or [parent_best],
                iteration_dir,
                idx,
                feedback_history,
            )
            record = asyncio.run(
                self._evaluate_with_repair(code, parent_best.id if parent_best else None, idx, gen_dir=gen_dir)
            )
            if record:
                self.programs[record.id] = record
                node.programs.append(record.id)
                prior_attempts.append(record)
                generated.append(record)
                # Only run feedback if there will be another generation in this iteration.
                if idx < self.programs_per_idea - 1:
                    feedback_text = self._run_generation_feedback(
                        node=node,
                        program_record=record,
                        eval_context=eval_context,
                        feedback_history=feedback_history,
                        gen_dir=gen_dir,
                    )
                    if feedback_text:
                        feedback_history.append(feedback_text)
                        record.generation_feedback = feedback_text
                        if isinstance(record.eval_artifacts, dict):
                            record.eval_artifacts["generation_feedback"] = feedback_text
        return generated

    def _copy_exp_markers(self, generated: List[ProgramRecord], iteration_dir: Path) -> None:
        for rec in generated:
            if not isinstance(rec.eval_artifacts, dict):
                continue
            candidate_paths = [
                rec.eval_artifacts.get("raw_debug_path"),
                rec.eval_artifacts.get("debug_path"),
            ]
            target_base = rec.eval_artifacts.get("program_generation_dir")
            target_base_path = Path(target_base) if target_base else iteration_dir
            for dbg in candidate_paths:
                if not dbg:
                    continue
                dbg_path = Path(dbg)
                try:
                    logger.info("Attempting exp copy from debug path: %s", dbg_path)
                    exp_dir = None
                    if "exp" in dbg_path.name.lower():
                        exp_dir = dbg_path
                    else:
                        for ancestor in dbg_path.parents:
                            if "exp" in ancestor.name.lower():
                                exp_dir = ancestor
                                break
                    if exp_dir and exp_dir.exists():
                        target = target_base_path / exp_dir.name
                        if exp_dir.is_dir():
                            if not target.exists():
                                shutil.copytree(exp_dir, target, dirs_exist_ok=True)
                                logger.info("Copied exp dir %s -> %s", exp_dir, target)
                        else:
                            if not target.exists():
                                shutil.copy2(exp_dir, target)
                                logger.info("Copied exp file %s -> %s", exp_dir, target)
                        break
                    else:
                        logger.info("No exp-* directory found for %s (checked %s)", dbg, dbg_path)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to copy exp marker from %s: %s", dbg, exc)

    def _copy_rag_outputs(self, iteration_dir: Path) -> None:
        """Copy idea_agent/evaluation_agent outputs into iteration folder if present."""
        sources: List[Path] = [IDEA_OUTPUT_PATH, EVAL_OUTPUT_PATH]
        if self.eval_output_path:
            sources.append(self.eval_output_path)

        for source in sources:
            try:
                source_path = Path(source)
                if not source_path.exists():
                    continue
                target = iteration_dir / source_path.name
                if target.resolve() == source_path.resolve():
                    continue
                shutil.copy2(source_path, target)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to copy %s: %s", source, exc)

    def _materialize_best_generation(self) -> None:
        """Create a best_generation directory with the top program/idea/eval artifacts."""

        if not self.nodes:
            logger.info("No nodes to materialize best generation.")
            return

        best_node = max(
            self.nodes.values(),
            key=lambda n: n.best_score or float("-inf"),
        )
        if not best_node.best_program_id or best_node.best_program_id not in self.programs:
            logger.info("Best node missing program; skipping best_generation export.")
            return

        best_program = self.programs[best_node.best_program_id]
        target_dir = self.results_root / "best_generation"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save program code
        code_path = target_dir / f"best_program{self.initial_program_path.suffix or '.txt'}"
        code_path.write_text(best_program.code, encoding="utf-8")

        # Save program record
        (target_dir / "program_record.json").write_text(
            json.dumps(asdict(best_program), indent=2), encoding="utf-8"
        )

        # Save idea payload/summary
        idea_payload_path = target_dir / "idea.yaml"
        idea_payload = {
            "idea_summary": best_node.idea_summary,
            "idea_payload": best_node.idea_payload,
            "parent_id": best_node.parent_id,
            "eval_summary": best_node.eval_summary,
            "eval_hypothesis": best_node.eval_hypothesis,
            "eval_score": best_node.eval_score,
        }
        idea_payload_path.write_text(
            yaml.safe_dump(_literalize(idea_payload), sort_keys=False, width=120, default_flow_style=False),
            encoding="utf-8",
        )

        # Copy eval output if available
        if best_node.eval_output_path:
            try:
                src = Path(best_node.eval_output_path)
                if src.exists():
                    shutil.copy2(src, target_dir / src.name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to copy eval output to best_generation: %s", exc)

        # Copy exp-* artifacts if present in eval_artifacts paths
        if isinstance(best_program.eval_artifacts, dict):
            candidate_paths = [
                best_program.eval_artifacts.get("raw_debug_path"),
                best_program.eval_artifacts.get("debug_path"),
            ]
            for dbg in candidate_paths:
                if not dbg:
                    continue
                dbg_path = Path(dbg)
                try:
                    exp_dir = None
                    if "exp" in dbg_path.name.lower():
                        exp_dir = dbg_path
                    else:
                        for ancestor in dbg_path.parents:
                            if "exp" in ancestor.name.lower():
                                exp_dir = ancestor
                                break
                    if exp_dir and exp_dir.exists():
                        target = target_dir / exp_dir.name
                        if exp_dir.is_dir():
                            shutil.copytree(exp_dir, target, dirs_exist_ok=True)
                            logger.info("Best generation: copied exp dir %s -> %s", exp_dir, target)
                        else:
                            shutil.copy2(exp_dir, target)
                            logger.info("Best generation: copied exp file %s -> %s", exp_dir, target)
                        break
                except Exception as exc:  # pragma: no cover
                    logger.warning("Best generation: failed to copy exp marker from %s: %s", dbg, exc)

    def run(self, iterations: int = 3) -> None:
        if not self.nodes:
            self._bootstrap_first_node()

        for _ in range(iterations):
            self.iteration_counter += 1
            parent_node = self._select_parent_node()
            parent_best = self.programs[parent_node.best_program_id] if parent_node.best_program_id else None
            if parent_best is None:
                raise RuntimeError("Best program missing for parent node.")

            iteration_dir = self.results_root / f"iteration_{self.iteration_counter}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            parent_eval_ctx = {
                "summary": parent_node.eval_summary,
                "hypothesis": parent_node.eval_hypothesis,
                "score": parent_node.eval_score,
            }
            eval_prompt_path = self._write_eval_prompt(iteration_dir, parent_best, parent_eval_ctx)
            parent_output_dir = Path(parent_node.artifacts_dir) if parent_node.artifacts_dir else iteration_dir
            eval_context = self._run_eval_qa(eval_prompt_path, parent_node, parent_output_dir, mirror_dir=iteration_dir)
            idea_prompt_path = self._write_idea_prompt(iteration_dir, parent_best, eval_context)

            max_novel_retries = 3
            attempt = 0
            while True:
                idea_payload, idea_iters = self._generate_new_idea(
                    parent_node, eval_context, idea_prompt_path
                )
                idea_summary = _extract_idea_summary(idea_payload)
                idea_embedding = self.similarity.embed(idea_summary)
                is_novel, novelty_reason = self._is_novel_llm(idea_summary)
                if is_novel:
                    logger.info(
                        "Novelty check passed for idea: %s | reason: %s",
                        idea_summary,
                        novelty_reason or "n/a",
                    )
                    break
                logger.info(
                    "Idea judged not novel; regenerating with forbidden list. Summary: %s | reason: %s",
                    idea_summary,
                    novelty_reason or "n/a",
                )
                self.rejected_ideas.append(idea_summary)
                attempt += 1
                if attempt >= max_novel_retries:
                    logger.warning("Exceeded novelty retries; proceeding with last idea despite similarity.")
                    break
                idea_prompt_path = self._write_idea_prompt(iteration_dir, parent_best, eval_context)

            new_node = self._build_node_from_idea(idea_payload, idea_embedding, idea_iters, parent_node.id)

            generated = self._generate_programs_for_node(new_node, parent_best, iteration_dir, eval_context)

            successful = [rec for rec in generated if rec.status == "success"]
            if successful:
                best = max(successful, key=lambda r: _score_from_metrics(r.metrics))
                new_node.update_best(best.id, _score_from_metrics(best.metrics))
            else:
                new_node.programs.append(parent_best.id)
                new_node.update_best(parent_best.id, _score_from_metrics(parent_best.metrics))

            self.nodes[new_node.id] = new_node

            self._save_snapshot(iteration_dir, new_node, generated)
            self._copy_exp_markers(generated, iteration_dir)
            self._copy_rag_outputs(iteration_dir)

            logger.info(
                "Iteration %s complete. Programs: %s, best score=%s",
                self.iteration_counter,
                len(generated),
                new_node.best_score,
            )

        # Materialize best overall artifacts at the end of the run.
        self._materialize_best_generation()
