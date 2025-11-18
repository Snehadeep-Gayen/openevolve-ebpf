#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An agent that answers deep technical questions about the codebase by combining:
  • RAG over your repo via llama_rag.py (index + query tools)
  • A safe, read-only repo shell tool (grep/rg, cat, tree, etc.)

LangChain v1 API:
  - Agent construction: langchain.agents.create_agent
  - Tools: @tool decorator or BaseTool instances

Install (Python 3.10+):
    pip install -U "langchain>=1.0" "langchain-openai>=0.2" "langchain-community>=0.2" llama-index

Environment:
    export OPENAI_API_KEY=sk-...
    # optional, defaults shown below:
    export LANGCHAIN_AGENT_MODEL="openai:gpt-4o-mini"     # any supported chat model id
    export RAG_STORAGE_DIR="rag_storage"

CLI:
    python langchain_agent.py ask "How does the eBPF loader attach kprobes?"
    python langchain_agent.py repl
    python langchain_agent.py index     # force a (re)build of the vector index

Security NOTE:
    The `repo_shell` tool is restricted to read-only operations inside REPO_ROOT.
    Destructive and networked commands are blocked. Adjust at your own risk.

"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

# --- Import your RAG module ---------------------------------------------------
# Try normal import first; if running from sibling file, add the current folder.
try:
    import llama_rag
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    import llama_rag  # type: ignore

# --- LangChain (v1) -----------------------------------------------------------
from langchain.agents import create_agent
from langchain.tools import tool

# If you prefer creating a model object explicitly you can do this instead:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model=os.getenv("LANGCHAIN_AGENT_MODEL", "gpt-4o-mini"), temperature=0)

# -----------------------------------------------------------------------------


# ---------- Configuration -----------------------------------------------------
DEFAULT_PERSIST_DIR = Path(os.getenv("RAG_STORAGE_DIR", "rag_storage"))
REPO_ROOT: Path = getattr(llama_rag, "REPO_ROOT", Path(__file__).resolve().parents[1])

# Commands that may mutate the system or exfiltrate data — blocked by repo_shell.
_DANGEROUS = {
    "rm", "mv", "cp -r", "chmod", "chown", "chgrp", "truncate", "dd", "mkfs",
    "sudo", "service", "systemctl", "kill", "pkill", "docker", "podman",
    "curl", "wget", "ssh", "scp", "rsync", "sftp",
    "apt", "yum", "dnf", "pacman", "brew", "pip", "python -m pip",
    "git reset", "git clean", "git checkout", "git push", "git pull",
}


def _require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return key


def _trim(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


# ---------- Tools: wrap llama_rag.py -----------------------------------------
@tool
def rag_index(
    persist_dir: str = str(DEFAULT_PERSIST_DIR),
    chunk_size: int = (
        llama_rag.DEFAULT_CODE_CHUNK_LINES
        if getattr(llama_rag, "USE_CODE_SPLITTER", True)
        else llama_rag.DEFAULT_TOKEN_CHUNK_SIZE
    ),
    chunk_overlap: int = (
        llama_rag.DEFAULT_CODE_CHUNK_OVERLAP_LINES
        if getattr(llama_rag, "USE_CODE_SPLITTER", True)
        else llama_rag.DEFAULT_TOKEN_CHUNK_OVERLAP
    ),
) -> str:
    """Rebuild the repo vector index via llama_rag.py, returning a short status message.

    Args:
        persist_dir: Directory for the vector store output.
        chunk_size: Chunk granularity (lines or tokens depending on splitter).
        chunk_overlap: Overlap between adjacent chunks.
    """
    _require_openai_key()
    llama_rag.configure_llamaindex(os.environ["OPENAI_API_KEY"])
    llama_rag.build_index(
        persist_dir=Path(persist_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    mode = "CodeSplitter" if getattr(llama_rag, "USE_CODE_SPLITTER", True) else "TokenTextSplitter"
    return f"✅ Index built at {persist_dir} (mode={mode}, chunk_size={chunk_size}, overlap={chunk_overlap})."


@tool
def rag_query(
    question: str,
    top_k: int = 3,
    persist_dir: str = str(DEFAULT_PERSIST_DIR),
) -> str:
    """Query the existing RAG index and return a synthesized answer with snippets.

    Args:
        question: Technical question about the repo.
        top_k: Number of chunks to retrieve.
        persist_dir: Directory holding the persisted index.
    """
    _require_openai_key()
    llama_rag.configure_llamaindex(os.environ["OPENAI_API_KEY"])
    try:
        answer = llama_rag.query_index(
            persist_dir=Path(persist_dir),
            question=question,
            top_k=top_k,
        )
        return _trim(str(answer))
    except FileNotFoundError:
        return (
            "⚠️ No RAG index found at "
            f"{persist_dir}. Call the `rag_index` tool first, then retry `rag_query`."
        )


# ---------- Tool: restricted shell over the repo -----------------------------
@tool
def repo_shell(command: str, timeout: int = 30) -> str:
    """Run a read-only shell command rooted at REPO_ROOT and return trimmed output.

    Args:
        command: POSIX command string (single line).
        timeout: Seconds before the command is killed.
    """
    # Basic denylist to reduce risk; adjust as needed.
    lowered = command.strip().lower()
    for bad in _DANGEROUS:
        if re.search(rf"\b{re.escape(bad)}\b", lowered):
            return f"⛔ Blocked potentially dangerous command token: '{bad}'. Use read-only queries."

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "LC_ALL": "C"},
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        preface = f"$ (cwd={REPO_ROOT}) {command}\n"
        body = out if proc.returncode == 0 else (out + ("\n" if out and err else "") + err)
        return _trim(preface + body)
    except subprocess.TimeoutExpired:
        return f"⏱️ Command timed out after {timeout}s: {command}"
    except Exception as e:  # pragma: no cover
        return f"❌ repo_shell error: {type(e).__name__}: {e}"


# ---------- Agent builder -----------------------------------------------------
def build_agent(model_id: Optional[str] = None):
    """
    Build a LangChain agent that can:
      • run RAG over the repo (rag_index, rag_query)
      • inspect the repo via a read-only shell (repo_shell)
    """
    model = model_id or os.getenv("LANGCHAIN_AGENT_MODEL", "openai:gpt-4o-mini")
    system_prompt = dedent(
        f"""
        You are Codebase Investigator, a senior engineer answering *very* technical,
        code-level questions about this repository.

        Capabilities:
          1) `rag_query` answers questions using the vector index (call `rag_index` first if missing).
          2) `repo_shell` lets you grep and open files within the repo root: {REPO_ROOT}.
             Use `rg -n` (or `grep -nR`) to gather evidence with line numbers.

        Style:
          • Always cite files with relative paths and line numbers when possible.
          • Prefer exact quotes in fenced code blocks for critical snippets.
          • If the index looks stale or a file is missing, call `rag_index` and retry.
          • Never use shell for destructive/system/network operations.

        When asked a question:
          • First, try `rag_query`. If lacking context, augment with `repo_shell` searches.
          • Synthesize a precise answer with citations like: path/to/file.go:L123-L140.
        """
    ).strip()

    # `create_agent` accepts a model string like "openai:gpt-4o-mini" or a ChatModel instance.
    # (You must have the provider's integration installed; e.g., langchain-openai for OpenAI.)
    graph = create_agent(
        model=model,
        tools=[rag_index, rag_query, repo_shell],
        system_prompt=system_prompt,
    )
    return graph


# ---------- CLI ---------------------------------------------------------------
def _extract_text_output(agent_result: Dict[str, Any]) -> str:
    """
    create_agent returns a compiled LangGraph whose .invoke() result is a state dict
    containing a 'messages' list. This helper extracts the final assistant text.
    """
    msgs: List[Any] = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    if not msgs:
        return ""
    last = msgs[-1]
    # Message may be a dict or an AIMessage object.
    content = None
    if isinstance(last, dict):
        content = last.get("content", "")
    else:
        content = getattr(last, "content", "")

    # If content is a list of parts, join text parts.
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            else:
                parts.append(str(p))
        content = "\n".join(parts)

    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangChain agent with RAG + safe repo shell."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LANGCHAIN_AGENT_MODEL", "openai:gpt-4o-mini"),
        help="Model identifier (e.g., 'openai:gpt-4o-mini').",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # One-off question
    ask_p = sub.add_parser("ask", help="Ask a one-off deep technical question.")
    ask_p.add_argument("question", type=str)

    # Interactive REPL
    sub.add_parser("repl", help="Interactive chat REPL.")

    # Convenience: force an index build without going through the agent loop
    idx_p = sub.add_parser("index", help="Force a (re)build of the vector index.")
    idx_p.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))

    args = parser.parse_args()
    graph = build_agent(model_id=args.model)

    if args.command == "ask":
        user_msg = {"role": "user", "content": args.question}
        result = graph.invoke({"messages": [user_msg]})
        print(_extract_text_output(result))

    elif args.command == "repl":
        print("Codebase Investigator REPL. Type 'exit' or Ctrl-D to quit.")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            result = graph.invoke({"messages": [{"role": "user", "content": q}]})
            print(_extract_text_output(result), end="\n\n")

    elif args.command == "index":
        # Fast path: call the tool directly
        print(rag_index.invoke({"persist_dir": args.persist_dir}))

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
