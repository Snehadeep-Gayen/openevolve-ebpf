"""Helper utilities for performing OpenAI chat completion calls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from openai import OpenAI

API_KEY_ENV = "OPENAI_API_KEY"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_llm_call(payload: Dict[str, Any]) -> Any:
    """Call the OpenAI chat completions API with *payload* and print the response."""

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {API_KEY_ENV} is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(**payload)
    print(json.dumps(response.model_dump(), indent=2))
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
    base_dir = os.path.dirname(__file__)
    crafted_path = os.path.join(
        base_dir, "exp-20251113-162706", "evolve_debug", "codex_payload_crafted.json"
    )

    with open(crafted_path, "r", encoding="utf-8") as crafted_file:
        crafted_text = crafted_file.read()

    system_prompt = crafted_text

    message_history = [{"role": "system", "content": system_prompt}]
    model = "gpt-5.1"
    outputs = []

    for i in [0]:
        response = make_llm_call({"model": model, "messages": message_history})
        response_json = response.model_dump()
        assistant_message = response.choices[0].message
        print(json.dumps(assistant_message.model_dump(), indent=2))
        assistant_text = _message_content_to_text(assistant_message.content)
        assistant_json = _extract_json_payload(assistant_text)
        rag_results = (
            answer_questions_with_rag(assistant_json.get("questions"))
            if isinstance(assistant_json, dict)
            else []
        )

        message_history.append({"role": assistant_message.role, "content": assistant_text})
        outputs.append(
            {
                "assistant_role": assistant_message.role,
                "assistant_content": assistant_json if assistant_json is not None else assistant_text,
                "assistant_content_format": "json" if assistant_json is not None else "text",
                "assistant_content_raw": assistant_text,
                "full_response": response_json,
                "rag_results": rag_results,
            }
        )

    test_path = os.path.join(base_dir, "test.json")
    with open(test_path, "w", encoding="utf-8") as outfile:
        json.dump(outputs, outfile, indent=2)
