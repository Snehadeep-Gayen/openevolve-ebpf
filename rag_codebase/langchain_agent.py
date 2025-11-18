"""
LangChain-powered agent that can answer deep technical questions by combining:

- The existing LlamaIndex RAG query capability (see `llama_rag.py`).
- Read-only shell/file inspection tools that operate inside the repo.

Usage:
    python langchain_agent.py \
        --persist-dir ./storage_code \
        --question "Explain how worker assignment happens."
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from llama_rag import (
    REPO_ROOT,
    configure_llamaindex,
    query_index,
)


class CodebaseRAGTool(BaseTool):
    """Thin wrapper that exposes the RAG query pipeline as a LangChain tool."""

    name: str = "codebase_rag_lookup"
    description: str = (
        "Use this to answer detailed questions about the codebase. "
        "Provide the full question or directive; the tool will return synthesized findings "
        "from the indexed sources."
    )

    persist_dir: Path
    top_k: int = 4

    def _run(self, query: str) -> str:
        return query_index(self.persist_dir, question=query, top_k=self.top_k)

    async def _arun(self, query: str) -> str:  # pragma: no cover - async not used
        raise NotImplementedError("CodebaseRAGTool does not support async")


class ShellCommandTool(BaseTool):
    """Execute read-only shell commands from the repo root."""

    name: str = "shell_command"
    description: str = (
        "Run safe, read-only shell commands (e.g., ls, rg, cat) within the repository. "
        "Pass a single command string as you would type it in the shell. "
        "Avoid destructive commands; output is truncated for safety."
    )

    max_output_chars: int = 4000

    def _run(self, command: str) -> str:
        cmd = command.strip()
        if not cmd:
            return "No command provided."

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        output = output.strip() or "(no output)"
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "\n... [truncated]"
        return f"$ {cmd}\n{output}"

    async def _arun(self, command: str) -> str:  # pragma: no cover - async not used
        raise NotImplementedError("ShellCommandTool does not support async")


class FileReadTool(BaseTool):
    """Read specific files (and optional line ranges) relative to the repo root."""

    name: str = "read_repo_file"
    description: str = (
        "Read a file or snippet from the repository. "
        "Input format: 'relative/path/to/file[:start_line[:end_line]]'. "
        "Line numbers are 1-based and inclusive; omit them to read the whole file."
    )

    max_chars: int = 4000

    def _run(self, query: str) -> str:
        path_str, start_line, end_line = self._parse_request(query)
        file_path = (REPO_ROOT / path_str).resolve()
        if not file_path.exists() or not file_path.is_file():
            return f"File not found: {path_str}"
        try:
            content = file_path.read_text()
        except UnicodeDecodeError:
            return f"File is not readable as text: {path_str}"

        lines = content.splitlines()
        start_idx = max(start_line - 1, 0) if start_line else 0
        end_idx = min(end_line, len(lines)) if end_line else len(lines)
        snippet = "\n".join(lines[start_idx:end_idx])
        if len(snippet) > self.max_chars:
            snippet = snippet[: self.max_chars] + "\n... [truncated]"

        header = f"{path_str}:{start_line or 1}-{end_idx}"
        body = snippet or "(file is empty in specified range)"
        return f"[File Snippet: {header}]\n{body}"

    async def _arun(self, query: str) -> str:  # pragma: no cover - async not used
        raise NotImplementedError("FileReadTool does not support async")

    @staticmethod
    def _parse_request(query: str) -> tuple[str, Optional[int], Optional[int]]:
        request = query.strip()
        if not request:
            raise ValueError("FileReadTool requires 'path[:start[:end]]'.")
        parts = request.split(":")
        path = parts[0]
        start = None
        end = None
        if len(parts) >= 2 and parts[1]:
            start_str = parts[1]
            if "-" in start_str and len(parts) == 2:  # handle path:start-end form
                start_part, end_part = start_str.split("-", 1)
                start = int(start_part) if start_part else None
                end = int(end_part) if end_part else None
            else:
                start = int(start_str)
                if len(parts) >= 3 and parts[2]:
                    end = int(parts[2])
        elif len(parts) >= 3 and parts[2]:
            end = int(parts[2])
        return path, start, end


def build_agent(
    question: str,
    persist_dir: Path,
    model: str,
    verbose: bool,
    stream: bool,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    configure_llamaindex(api_key)

    llm = ChatOpenAI(
        model=model,
        #temperature=0.2,
        openai_api_key=api_key,
    )

    tools = [
        CodebaseRAGTool(persist_dir=persist_dir),
        ShellCommandTool(),
        FileReadTool(),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert systems engineer helping investigate the codebase. "
                "Use the available tools (RAG, shell, file reader) to gather evidence before answering. "
                "Focus on deep, technical reasoning and cite relevant files by path. Prefer the RAG tool over the others and try to return quickly, but with a thorough answer.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=30,
        handle_parsing_errors=True,
    )
    callbacks = [StdOutCallbackHandler()] if verbose else None
    try:
        if stream:
            last_event: Optional[dict] = None
            for update in executor.stream(
                {"input": question},
                config={"callbacks": callbacks} if callbacks else None,
            ):
                last_event = update
                print(f"[stream] {update}")
            result = last_event or {}
        else:
            result = executor.invoke(
                {"input": question},
                config={"callbacks": callbacks} if callbacks else None,
            )
    except Exception as exc:
        return f"Agent failed: {exc}"
    return result.get("output", str(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain agent for deep code questions.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("storage_code"),
        help="Path to the persisted LlamaIndex storage directory.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question or task for the agent to solve.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI chat model to drive the agent.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable LangChain agent tracing/logging.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream intermediate agent thoughts/tool calls instead of only the final answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answer = build_agent(
        question=args.question,
        persist_dir=args.persist_dir,
        model=args.model,
        verbose=args.verbose,
        stream=args.stream,
    )
    print(answer)


if __name__ == "__main__":
    main()
