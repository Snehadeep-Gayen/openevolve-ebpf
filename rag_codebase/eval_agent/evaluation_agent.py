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


def _write_outputs(destination: str, iteration_records: list[Dict[str, Any]]) -> None:
    """Persist the current iteration records to disk for inspection."""

    payload = [{"iterations": iteration_records}]
    with open(destination, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


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
    initial_prompt = (
        "Begin the investigation by proposing an initial hypothesis and the verification questions you want "
        "to run through RAG. Respond strictly as JSON with this shape: "
        '{"status":"continue","Summary":"<brief recap of artifacts>","hypothesis":{"text":"<your best hypothesis>","score":<0-10>},"questions":["question one","question two"]}. '
        "Do not list more than 5 questions, keep each one concrete and code-focused, and do not repeat questions you already have the answer to. "
        'If you determine the hypothesis from a previous iteration is fully supported, respond with status "finish" and set "questions": []. '
        "All future turns must use the same JSON format."
    )
    message_history.append({"role": "user", "content": initial_prompt})
    model = "gpt-5.1"
    temperature = 0.8
    iteration_records: list[Dict[str, Any]] = []
    iteration_count = 3
    question_iterations = max(iteration_count - 1, 1)
    test_path = os.path.join(base_dir, "test.json")

    for iteration_index in range(question_iterations):
        response = make_llm_call(
            {"model": model, "messages": message_history, "reasoning_effort": "high"}
        )
        response_json = response.model_dump()
        assistant_message = response.choices[0].message
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
            }
        )

        message_history.append({"role": assistant_message.role, "content": assistant_text})

        _write_outputs(test_path, iteration_records)

        if status == "finish":
            break

        if iteration_index < question_iterations - 1:
            followup_prompt = _format_followup_prompt(iteration_index, parsed_response)
            message_history.append({"role": "user", "content": followup_prompt})

    final_prompt = _format_final_prompt(iteration_records)
    message_history.append({"role": "user", "content": final_prompt})
    final_response = make_llm_call(
        {"model": model, "messages": message_history, "reasoning_effort": "high"}
    )
    final_response_json = final_response.model_dump()
    final_assistant_message = final_response.choices[0].message
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
        }
    )

    _write_outputs(test_path, iteration_records)
