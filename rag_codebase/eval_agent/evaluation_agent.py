"""Helper utilities for performing OpenAI chat completion calls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Union

from openai import OpenAI
import yaml

API_KEY_ENV = "OPENAI_API_KEY"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_KEYS = ("eval_agent_prompt", "idea_agent_prompt", "prompt", "system_prompt")


def make_llm_call(payload: Dict[str, Any]) -> Any:
    """Call the OpenAI chat completions API with *payload* and print the response."""

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {API_KEY_ENV} is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(**payload)
    #print(json.dumps(response.model_dump(), indent=2))
    return response


def _message_content_to_text(content: Any) -> str:
    """Convert an OpenAI message content payload into a flat string."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content)


def _strip_code_fence(block: str) -> str:
    """Remove wrapping ``` fences, including optional language labels."""

    stripped = block.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        newline_index = stripped.find("\n")
        if newline_index != -1:
            stripped = stripped[newline_index + 1 :]
        else:
            stripped = ""
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _extract_json_payload(content: str) -> Optional[Any]:
    """Return parsed JSON content if *content* contains a JSON blob."""

    candidate = content.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        candidate = _strip_code_fence(candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _resolve_rag_persist_dir() -> Path:
    """Resolve the configured RAG storage directory relative to the project root."""

    configured = os.getenv("RAG_STORAGE_DIR")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    return (PROJECT_ROOT / "storage_code").resolve()


def _format_rag_query(question: str) -> str:
    """Embed formatting instructions so the RAG output is structured JSON."""

    prompt = (
        "You are a retrieval QA assistant for the Pebble server codebase. "
        "Use the retrieved context to answer the question and highlight the most relevant code snippets. "
        "Respond strictly in JSON with two fields: "
        '"answer" (1-3 sentences referencing files and explaining the behavior) and '
        '"code_snippets" (an array of objects with "path" and "snippet" keys that show supporting code).'
    )
    #return f"{prompt}\nQuestion: {question}"
    return f"Question: {question}"


def _format_followup_prompt(iteration_index: int, parsed_response: Dict[str, Any]) -> str:
    """Build the follow-up instruction shared with the LLM."""

    question_lines = ["Questions and Answers:"]
    for idx, entry in enumerate(parsed_response.get("question_answers", []), start=1):
        answer_value = entry.get("answer")
        if isinstance(answer_value, str):
            answer_text = answer_value
        else:
            answer_text = json.dumps(answer_value)
        question_lines.append(f"{idx}. Question: {entry.get('question')}")
        question_lines.append(f"   Answer: {answer_text}")

    instructions = [
        f"Iteration {iteration_index + 1} RAG answers:",
        *question_lines,
        (
            "Given these answers, decide if the hypothesis is fully supported. Spend time to think critically about the responses and include the answer information in your new summary and in the hypothesis if necesarry. "
            'If it is, respond with status "finish" and do not propose new questions (set "questions": []). '
            "Otherwise respond with status \"continue\", come up with a new hypothesis given the previous answers (for example, if one of the answers invalidates the previous hypothesis, you should come up with something different that is not contradicted by the answers), and include new questions for the new hypothesis. These must not be the same as previous questions you already have the answer to."
        ),
        "Any new questions you provide will be sent to the RAG system before the next iteration.",
        (
            "Respond strictly as JSON using this shape:\n"
            '{"status":"finish|continue","Summary":"<concise recap>","hypothesis":{"text":"<refined hypothesis>","score":<0-10>},"questions":["question one","question two"]}'
        ),
        "Do not list more than 5 questions, and make each one concrete and code-specific.",
    ]
    return "\n".join(instructions)


def _format_final_prompt(iteration_records: list[Dict[str, Any]]) -> str:
    """Create the final instruction asking for a concluding hypothesis."""

    evidence_lines: list[str] = []
    for idx, record in enumerate(iteration_records, start=1):
        parsed = record.get("parsed_response", {})
        qa_list = parsed.get("question_answers", []) if isinstance(parsed, dict) else []
        if not qa_list:
            continue
        evidence_lines.append(f"Iteration {idx} answers:")
        for qa_idx, qa in enumerate(qa_list, start=1):
            question_text = qa.get("question")
            answer_value = qa.get("answer")
            if isinstance(answer_value, str):
                answer_text = answer_value
            else:
                answer_text = json.dumps(answer_value)
            evidence_lines.append(f"  {qa_idx}. Question: {question_text}")
            evidence_lines.append(f"     Answer: {answer_text}")

    if not evidence_lines:
        evidence_lines.append("No RAG answers were collected.")

    instructions = [
        "Questioning is complete. Synthesize a final explanation using the evidence below.",
        *evidence_lines,
        (
            "Respond strictly as JSON with this shape:\n"
            '{"status":"finish","Summary":"<final recap>","hypothesis":{"text":"<final hypothesis>","score":<0-10>},"questions":[]}'
        ),
        "Do not request more information or add new questions. Provide one precise, evidence-backed hypothesis.",
    ]
    return "\n".join(instructions)


class LiteralStr(str):
    """YAML literal block string."""


def _repr_literal_str(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralStr, _repr_literal_str)
yaml.add_representer(LiteralStr, _repr_literal_str, Dumper=yaml.SafeDumper)


def _literal(value: Optional[str]) -> Optional[LiteralStr]:
    return LiteralStr(value) if value is not None else None


def _write_outputs(destination: Union[str, Path], iteration_records: list[Dict[str, Any]]) -> None:
    """Persist the current iteration records to disk for inspection (JSON + YAML for readability)."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"iterations": iteration_records}]

    # JSON (compatibility)
    with destination_path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)

    # YAML (readable, with literal blocks for summaries/hypotheses)
    yaml_payload = []
    for entry in payload:
        iters = []
        for record in entry.get("iterations", []):
            rec_copy = dict(record)
            if isinstance(rec_copy.get("Summary"), str):
                rec_copy["Summary"] = _literal(rec_copy["Summary"])
            hypothesis = rec_copy.get("hypothesis")
            if isinstance(hypothesis, dict) and isinstance(hypothesis.get("text"), str):
                rec_copy["hypothesis"] = dict(hypothesis)
                rec_copy["hypothesis"]["text"] = _literal(hypothesis["text"])
            iters.append(rec_copy)
        yaml_payload.append({"iterations": iters})

    yaml_path = destination_path.with_suffix(".yaml")
    yaml_path.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, width=120, default_flow_style=False),
        encoding="utf-8",
    )


def _load_system_prompt(prompt_path: Optional[Path], default_path: Path) -> str:
    """Load system prompt, supporting raw text or JSON/YAML with prompt keys."""

    path = Path(prompt_path or default_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found at {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError:
            parsed = None

    if isinstance(parsed, dict):
        for key in PROMPT_KEYS:
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val

    return raw_text


def run_evaluation_qa_loop(
    *,
    iteration_count: int = 3,
    model: str = "gpt-5.1",
    temperature: Optional[float] = 0.8,
    reasoning_effort: Optional[str] = "high",
    prompt_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    initial_prompt: Optional[str] = None,
    echo_messages: bool = False,
) -> Dict[str, Any]:
    """Run the evaluation Q&A loop and return summaries and iteration records."""

    base_dir = Path(__file__).resolve().parent
    crafted_path = Path(prompt_path or (base_dir / "payloads" / "codex_payload_crafted.json"))
    system_prompt = _load_system_prompt(crafted_path, crafted_path)

    opening_prompt = (
        initial_prompt
        or (
            "Begin the investigation by proposing an initial hypothesis and the verification questions you want "
            "to run through RAG. Respond strictly as JSON with this shape: "
            '{"status":"continue","Summary":"<brief recap of artifacts>","hypothesis":{"text":"<your best hypothesis>","score":<0-10>},"questions":["question one","question two"]}. '
            "Do not list more than 5 questions, keep each one concrete and code-focused, and do not repeat questions you already have the answer to. "
            'If you determine the hypothesis from a previous iteration is fully supported, respond with status "finish" and set "questions": []. '
            "All future turns must use the same JSON format."
        )
    )

    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": opening_prompt},
    ]

    iteration_records: list[Dict[str, Any]] = []
    question_iterations = max(iteration_count - 1, 1)
    destination = Path(output_path or (base_dir / "eval_output.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)

    for iteration_index in range(question_iterations):
        payload = {"model": model, "messages": message_history}
        if reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            payload["temperature"] = temperature
        response = make_llm_call(payload)
        response_json = response.model_dump()
        assistant_message = response.choices[0].message
        if echo_messages:
            print(json.dumps(assistant_message.model_dump(), indent=2))
        assistant_text = _message_content_to_text(assistant_message.content)
        assistant_json = _extract_json_payload(assistant_text)
        if assistant_json is None:
            raise ValueError("Assistant response must be valid JSON.")

        questions = assistant_json.get("questions") or []
        rag_results = answer_questions_with_rag(questions)

        question_answers = []
        for index, question_text in enumerate(questions):
            rag_answer = rag_results[index]["answer"] if index < len(rag_results) else None
            question_answers.append(
                {
                    "question": question_text,
                    "answer": rag_answer,
                }
            )

        summary_text = assistant_json.get("Summary") or assistant_json.get("summary", "")
        hypothesis_obj = assistant_json.get("hypothesis") or {}
        hypothesis_text = hypothesis_obj.get("text", "")
        hypothesis_score = hypothesis_obj.get("score")
        raw_status = assistant_json.get("status", "")
        normalized_status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
        status = "finish" if normalized_status == "finish" else "continue"

        parsed_response = {
            "status": status,
            "Summary": summary_text,
            "hypothesis": {"text": hypothesis_text, "score": hypothesis_score},
            "questions": questions,
            "question_answers": question_answers,
        }

        iteration_records.append(
            {
                "iteration": len(iteration_records),
                "assistant_role": assistant_message.role,
                "assistant_content_format": "json",
                "assistant_content_raw": assistant_text,
                "full_response": response_json,
                "status": status,
                "Summary": summary_text,
                "hypothesis": {"text": hypothesis_text, "score": hypothesis_score},
                "questions": questions,
                "question_answers": question_answers,
                "iteration_type": "question",
                "parsed_response": parsed_response,
            }
        )

        message_history.append({"role": assistant_message.role, "content": assistant_text})

        _write_outputs(destination, iteration_records)

        if status == "finish":
            break

        if iteration_index < question_iterations - 1:
            followup_prompt = _format_followup_prompt(iteration_index, parsed_response)
            message_history.append({"role": "user", "content": followup_prompt})

    final_prompt = _format_final_prompt(iteration_records)
    message_history.append({"role": "user", "content": final_prompt})
    final_payload = {"model": model, "messages": message_history}
    if reasoning_effort and reasoning_effort != "none":
        final_payload["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        final_payload["temperature"] = temperature
    final_response = make_llm_call(final_payload)
    final_response_json = final_response.model_dump()
    final_assistant_message = final_response.choices[0].message
    if echo_messages:
        print(json.dumps(final_assistant_message.model_dump(), indent=2))
    final_text = _message_content_to_text(final_assistant_message.content)
    final_json = _extract_json_payload(final_text)
    if final_json is None:
        raise ValueError("Final assistant response must be valid JSON.")

    final_summary = final_json.get("Summary") or final_json.get("summary", "")
    final_hypothesis = final_json.get("hypothesis") or {}
    final_status_raw = final_json.get("status", "")
    final_status = (
        "finish"
        if isinstance(final_status_raw, str) and final_status_raw.strip().lower() == "finish"
        else "finish"
    )
    final_questions = final_json.get("questions") or []

    final_parsed = {
        "status": final_status,
        "Summary": final_summary,
        "hypothesis": {
            "text": final_hypothesis.get("text", ""),
            "score": final_hypothesis.get("score"),
        },
        "questions": final_questions,
        "question_answers": [],
    }

    iteration_records.append(
        {
            "iteration": len(iteration_records),
            "assistant_role": final_assistant_message.role,
            "assistant_content_format": "json",
            "assistant_content_raw": final_text,
            "full_response": final_response_json,
            "status": final_status,
            "Summary": final_summary,
            "hypothesis": {
                "text": final_hypothesis.get("text", ""),
                "score": final_hypothesis.get("score"),
            },
            "questions": final_questions,
            "question_answers": [],
            "iteration_type": "final",
            "parsed_response": final_parsed,
        }
    )

    _write_outputs(destination, iteration_records)

    return {
        "iteration_records": iteration_records,
        "final_summary": final_summary,
        "final_hypothesis": final_hypothesis.get("text", ""),
        "final_hypothesis_score": final_hypothesis.get("score"),
        "final_status": final_status,
        "output_path": destination,
        "output_yaml_path": destination.with_suffix(".yaml"),
    }


def ask_rag_question(question: str, top_k: int = 3) -> Dict[str, Any]:
    """Query the repo RAG system with *question* and return structured JSON."""

    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {API_KEY_ENV} is not set")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    try:
        import llama_rag  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import llama_rag.py. Ensure dependencies are installed and the project root is on PYTHONPATH."
        ) from exc

    llama_rag.configure_llamaindex(api_key)
    persist_dir = _resolve_rag_persist_dir()
    formatted_question = _format_rag_query(question.strip())
    try:
        response_text = llama_rag.query_index(
            persist_dir=persist_dir,
            question=formatted_question,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"RAG index missing at {persist_dir}. Run the indexing pipeline before querying."
        ) from exc

    parsed_answer = _extract_json_payload(response_text)
    answer_payload: Any
    if parsed_answer is None:
        answer_payload = {"answer": response_text}
    else:
        answer_payload = parsed_answer

    return {"question": question.strip(), "answer": answer_payload}


def answer_questions_with_rag(raw_questions: Any, max_workers: int = 5) -> list[Dict[str, Any]]:
    """Run each question through the RAG system concurrently."""

    if not isinstance(raw_questions, list) or not raw_questions:
        return []

    cleaned_questions: list[str] = []
    for question in raw_questions:
        if not isinstance(question, str):
            continue
        trimmed = question.strip()
        if trimmed:
            cleaned_questions.append(trimmed)

    if not cleaned_questions:
        return []

    worker_count = max(1, min(max_workers, len(cleaned_questions)))

    def _run_single(question: str) -> Dict[str, Any]:
        try:
            return ask_rag_question(question)
        except Exception as exc:  # pragma: no cover
            return {
                "question": question,
                "answer": {
                    "error": f"Failed to query RAG: {exc}",
                },
            }

    results: list[Optional[Dict[str, Any]]] = [None] * len(cleaned_questions)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_run_single, question): idx
            for idx, question in enumerate(cleaned_questions)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # pragma: no cover
                results[idx] = {
                    "question": cleaned_questions[idx],
                    "answer": {"error": f"Unhandled RAG failure: {exc}"},
                }

    return [result for result in results if result is not None]


if __name__ == "__main__":
    result = run_evaluation_qa_loop(echo_messages=True)
    print(
        json.dumps(
            {
                "iterations": len(result["iteration_records"]),
                "output_path": str(result["output_path"]),
            }
        )
    )
