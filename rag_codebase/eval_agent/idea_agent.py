"""Idea generation agent that iterates with RAG context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # Allow running as module or script.
    from .evaluation_agent import (
        _extract_json_payload,
        _message_content_to_text,
        answer_questions_with_rag,
        make_llm_call,
    )
except ImportError:  # pragma: no cover
    from evaluation_agent import (
        _extract_json_payload,
        _message_content_to_text,
        answer_questions_with_rag,
        make_llm_call,
    )

import yaml

BASE_DIR = Path(__file__).resolve().parent
IDEA_OUTPUT_PATH = BASE_DIR / "idea_output.json"
EVAL_OUTPUT_PATH = BASE_DIR / "eval_output.json"
DEFAULT_PROMPT_PATH = BASE_DIR / "payloads" / "codex_payload_idea.yaml"
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_REASONING = "high"
PROMPT_KEYS = ("idea_agent_prompt", "eval_agent_prompt", "prompt", "system_prompt")
QUERY_KEYS = ("queries", "questions", "rag_queries")


def _supports_reasoning(model_name: str) -> bool:
    if not isinstance(model_name, str):
        return False
    lowered = model_name.lower()
    return lowered.startswith(("o3", "o1", "gpt-5"))


def _load_prompt_payload(prompt_path: Optional[Path] = None) -> str:
    """Return the textual system prompt stored in *prompt_path*."""

    payload_path = Path(prompt_path or DEFAULT_PROMPT_PATH)
    if not payload_path.exists():
        raise FileNotFoundError(f"Idea agent prompt not found at {payload_path}")

    raw_text = payload_path.read_text(encoding="utf-8")
    parsed: Any = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError:
            parsed = None

    if isinstance(parsed, dict):
        for key in PROMPT_KEYS:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
    elif isinstance(parsed, str) and parsed.strip():
        return parsed

    return raw_text


def _write_iteration_outputs(iteration_records: list[Dict[str, Any]]) -> None:
    """Persist the current iteration state to idea_output.json and a YAML sidecar."""

    class LiteralStr(str):
        """YAML literal block string."""

    def _repr_literal_str(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

    yaml.add_representer(LiteralStr, _repr_literal_str)
    yaml.add_representer(LiteralStr, _repr_literal_str, Dumper=yaml.SafeDumper)

    def _literalize(obj: Any) -> Any:
        if isinstance(obj, str):
            return LiteralStr(obj)
        if isinstance(obj, list):
            return [_literalize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _literalize(v) for k, v in obj.items()}
        return obj

    simplified: list[Dict[str, Any]] = []
    for record in iteration_records:
        simplified.append(
            {
                "iteration": record.get("iteration"),
                "idea_payload": record.get("idea_payload"),
                "rag_answers": record.get("rag_answers"),
            }
        )

    payload = {"iterations": simplified}
    # JSON for compatibility
    IDEA_OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # YAML with literal blocks for readability
    yaml_payload = _literalize(payload)
    yaml_path = IDEA_OUTPUT_PATH.with_suffix(".yaml")
    yaml_path.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, width=120, default_flow_style=False),
        encoding="utf-8",
    )


def _load_final_evaluation_fields(
    eval_path: Optional[Path] = None,
) -> tuple[str, str]:
    """Read eval_output.json and return (summary, hypothesis_text)."""

    path = Path(eval_path or EVAL_OUTPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation output not found at {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation output at {path} is not valid JSON") from exc

    iterations: list[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        maybe = parsed.get("iterations")
        if isinstance(maybe, list):
            iterations.extend(maybe)
    elif isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, dict):
                maybe = entry.get("iterations")
                if isinstance(maybe, list):
                    iterations.extend(maybe)

    if not iterations:
        raise ValueError(f"No iteration records found in {path}")

    final_iteration = iterations[-1]
    summary_value = final_iteration.get("Summary") or final_iteration.get("summary", "")
    hypothesis_obj = final_iteration.get("hypothesis")
    if isinstance(hypothesis_obj, dict):
        hypothesis_value = hypothesis_obj.get("text", "")
    elif isinstance(hypothesis_obj, str):
        hypothesis_value = hypothesis_obj
    else:
        hypothesis_value = ""

    summary_text = summary_value if isinstance(summary_value, str) else _to_display_text(summary_value)
    hypothesis_text = (
        hypothesis_value if isinstance(hypothesis_value, str) else _to_display_text(hypothesis_value)
    )

    return summary_text, hypothesis_text


def _to_display_text(value: Any) -> str:
    """Convert raw values into compact text for prompts."""

    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "Not provided."
    try:
        return json.dumps(value, ensure_ascii=True, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _build_initial_instruction(
    evaluation_hypothesis: Any,
    performance_summary: Any,
) -> str:
    """Create the opening user instruction for the idea generator."""

    lines = [
        "Use the following evaluation context to seed a fresh program design idea.",
        f"Evaluation hypothesis:\n{_to_display_text(evaluation_hypothesis)}",
        f"Performance summary:\n{_to_display_text(performance_summary)}",
        (
            "Produce the first iteration of the design doc as JSON using the schema "
            "described in the system prompt payload. Include concrete, code-focused "
            "queries you want the RAG system to answer."
        ),
    ]
    return "\n\n".join(lines)


def _extract_queries(parsed_payload: Dict[str, Any]) -> list[str]:
    """Gather query strings from any supported field."""

    seen: set[str] = set()
    collected: list[str] = []
    for key in QUERY_KEYS:
        raw = parsed_payload.get(key)
        if isinstance(raw, str):
            maybe = raw.strip()
            if maybe and maybe not in seen:
                seen.add(maybe)
                collected.append(maybe)
        elif isinstance(raw, Iterable):
            for entry in raw:
                if not isinstance(entry, str):
                    continue
                maybe = entry.strip()
                if maybe and maybe not in seen:
                    seen.add(maybe)
                    collected.append(maybe)
    return collected


def _update_snippet_collection(
    rag_answers: list[Dict[str, Any]],
    snippet_keys: set[tuple[str, str]],
    snippet_list: list[Dict[str, str]],
) -> None:
    """Accumulate unique code snippets from RAG answers."""

    for entry in rag_answers:
        answer_obj = entry.get("answer")
        if not isinstance(answer_obj, dict):
            continue
        raw_snippets = answer_obj.get("code_snippets")
        if not isinstance(raw_snippets, list):
            continue
        for snippet in raw_snippets:
            if not isinstance(snippet, dict):
                continue
            path = snippet.get("path")
            snippet_text = snippet.get("snippet")
            if not isinstance(path, str) or not isinstance(snippet_text, str):
                continue
            key = (path, snippet_text)
            if key in snippet_keys:
                continue
            snippet_keys.add(key)
            snippet_list.append(
                {
                    "path": path,
                    "snippet": snippet_text,
                }
            )


def _format_answer_text(answer: Any) -> str:
    """Turn an arbitrary RAG answer payload into readable text."""

    if isinstance(answer, str):
        return answer
    try:
        return json.dumps(answer, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(answer)


def _extract_idea_summary(parsed_payload: Dict[str, Any]) -> str:
    """Pull the best-effort textual summary of the current idea."""

    summary_keys = (
        "idea",
        "program_idea",
        "design",
        "Summary",
        "summary",
        "proposal",
    )
    for key in summary_keys:
        value = parsed_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(parsed_payload, ensure_ascii=True)[:4000]


def _format_refinement_prompt(
    iteration_index: int,
    parsed_payload: Dict[str, Any],
    rag_answers: list[Dict[str, Any]],
) -> str:
    """Create a follow-up instruction embedding new RAG context."""

    idea_summary = _extract_idea_summary(parsed_payload)
    lines = [
        f"Iteration {iteration_index + 1} idea snapshot:",
        idea_summary,
        "",
        "New application-specific information:",
    ]

    if rag_answers:
        for idx, qa in enumerate(rag_answers, start=1):
            question = qa.get("question") or "<unknown question>"
            answer = qa.get("answer")
            answer_text = _format_answer_text(answer)
            lines.append(f"{idx}. Question: {question}")
            lines.append(f"   Answer: {answer_text}")
    else:
        lines.append("No new answers were retrieved; refine based on prior knowledge.")

    lines.extend(
        [
            "",
            (
                "Refine the program idea/design document with this context. "
                "Return JSON matching the template from the system prompt, "
                "updating any sections that changed and proposing new concrete RAG "
                "queries if further code details are required. Avoid repeating "
                "questions that were already answered unless the new information "
                "invalidates the previous assumption."
            ),
        ]
    )

    return "\n".join(lines)


def run_idea_refinement(
    evaluation_hypothesis: Any,
    performance_summary: Any,
    *,
    iteration_count: int = 2,
    prompt_payload_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = DEFAULT_REASONING,
    temperature: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """Iteratively refine a program idea using LLM + RAG feedback."""

    max_iterations = max(1, iteration_count)
    system_prompt = _load_prompt_payload(prompt_payload_path)

    message_history: list[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_initial_instruction(
                evaluation_hypothesis, performance_summary
            ),
        },
    ]

    iteration_records: list[Dict[str, Any]] = []
    collected_snippets: list[Dict[str, str]] = []
    snippet_keys: set[tuple[str, str]] = set()
    supports_reasoning = (
        isinstance(reasoning_effort, str)
        and reasoning_effort.lower() != "none"
        and _supports_reasoning(model)
    )

    for iteration_index in range(max_iterations):
        print(f"[idea] Iteration {iteration_index + 1}/{max_iterations}")
        payload = {
            "model": model,
            "messages": message_history,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if supports_reasoning:
            payload["reasoning_effort"] = reasoning_effort
        response = make_llm_call(payload)
        response_json = response.model_dump()
        assistant_message = response.choices[0].message
        assistant_text = _message_content_to_text(assistant_message.content)
        parsed_payload = _extract_json_payload(assistant_text)
        if parsed_payload is None or not isinstance(parsed_payload, dict):
            raise ValueError("Assistant response must be valid JSON with object payload.")

        raw_status = parsed_payload.get("status")
        normalized_status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
        status = "finish" if normalized_status == "finish" else "continue"

        queries = _extract_queries(parsed_payload)
        no_query_requested = len(queries) == 0
        should_query = (
            status != "finish"
            and not no_query_requested
            and iteration_index < max_iterations - 1
        )

        if should_query:
            print(f"[idea]   RAG lookups requested: {len(queries)}")
            rag_answers = answer_questions_with_rag(queries)
            if rag_answers:
                print(f"[idea]   RAG answers retrieved: {len(rag_answers)}")
            else:
                print("[idea]   No RAG answers retrieved")
        else:
            if no_query_requested:
                print("[idea]   No queries requested; treating as finish.")
            else:
                print("[idea]   Skipping RAG queries (final iteration or status=finish)")
            rag_answers = []
            if status != "finish" and no_query_requested:
                status = "finish"

        _update_snippet_collection(rag_answers, snippet_keys, collected_snippets)

        if status == "finish":
            parsed_payload["implementation_snippets"] = [
                dict(item) for item in collected_snippets
            ]
        else:
            parsed_payload.pop("implementation_snippets", None)

        iteration_records.append(
            {
                "iteration": iteration_index,
                "idea_payload": parsed_payload,
                "rag_answers": rag_answers,
                "assistant_content_raw": assistant_text,
                "assistant_role": assistant_message.role,
                "full_response": response_json,
            }
        )

        _write_iteration_outputs(iteration_records)

        message_history.append({"role": assistant_message.role, "content": assistant_text})

        if status == "finish":
            print("[idea]   Received finish signal; stopping early.")
            break

        if iteration_index >= max_iterations - 1:
            break

        followup_prompt = _format_refinement_prompt(
            iteration_index,
            parsed_payload,
            rag_answers,
        )
        message_history.append({"role": "user", "content": followup_prompt})
        print("[idea]   Prepared refinement prompt")

    return iteration_records


__all__ = ["run_idea_refinement"]


if __name__ == "__main__":
    summary_text, hypothesis_text = _load_final_evaluation_fields()
    records = run_idea_refinement(
        evaluation_hypothesis=hypothesis_text,
        performance_summary=summary_text,
        prompt_payload_path=DEFAULT_PROMPT_PATH,
        iteration_count=3,
    )
    print(
        json.dumps(
            {
                "iterations": len(records),
                "output_path": str(IDEA_OUTPUT_PATH),
            }
        )
    )
