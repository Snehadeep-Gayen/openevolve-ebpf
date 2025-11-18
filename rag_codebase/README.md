# RAG Codebase Explorer

This folder contains a tiny, documented scaffold for building a Retrieval-Augmented Generation (RAG) workflow on top of a source-code repository using [LlamaIndex](https://github.com/run-llama/llama_index). It supports:

- Chunking selected source files (Go/C/Python in the default template) with LlamaIndex's syntax-aware `CodeSplitter`.
- Persisting a lightweight local vector index.
- Querying that index with OpenAI's `gpt-5` (or any compatible) model, pulling the API key from the `OPENAI_API_KEY` environment variable.

The entry point is [`rag_codebase/main.py`](./main.py).

## Prerequisites

```bash
pip install -r rag_codebase/requirements.txt
```

If you already manage dependencies elsewhere, make sure you install `llama-index`, `tree_sitter`, and `tree_sitter_language_pack` (plus anything else you rely on such as `python-dotenv`).

## Basic Usage

```bash
export OPENAI_API_KEY=sk-...
cd rag_codebase

# 1. Build the multi-language index (edit LANGUAGE_CONFIGS before this step)
python main.py index \
  --persist-dir ./rag_storage \
  --chunk-size 40 \
  --chunk-overlap 15

# 2. Ask questions that the LLM will answer using retrieved chunks
python main.py query \
  --persist-dir ./rag_storage \
  --top-k 4 \
  "How does the controller schedule islands?"
```

### LangChain agent

```bash
python langchain_agent.py \
  --persist-dir ./rag_storage \
  --question "Trace how worker assignments flow from Go to the eBPF layer."
```

This interactive agent combines:

- The `llama_rag.py` query engine as a tool for semantic lookup.
- Read-only shell command execution rooted at the repo.
- Direct file reading snippets (e.g., `cmd/server/main.go:1:120`).

It is meant for deep debugging or investigative questions that may require iteratively listing files, grepping, and synthesizing retrieved code.

### Flags

- `--persist-dir`: Where the vector index is stored/read.
- `--chunk-size` and `--chunk-overlap`: Track either lines (CodeSplitter mode) or tokens (TokenTextSplitter mode), depending on the `USE_CODE_SPLITTER` flag in `main.py`.
- `--top-k`: How many similar chunks LlamaIndex should retrieve when answering.
- Per-language source selection is driven by the `LANGUAGE_CONFIGS` list in [`main.py`](./main.py).
- Toggling `USE_CODE_SPLITTER` in `main.py` switches between language-aware chunking (default) and a single TokenTextSplitter pass; the CLI arguments automatically map to lines or tokens based on that flag.

## Configure per-language ingestion

The top of `main.py` contains a `LANGUAGE_CONFIGS` list. Each entry describes one ingestion pass:

```python
REPO_ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_CONFIGS = [
    LanguageIngestionConfig(
        language="go",
        directories=[REPO_ROOT / "cmd", REPO_ROOT / "internal" / "server"],
        extensions=(".go",),
    ),
    LanguageIngestionConfig(
        language="c",
        directories=[REPO_ROOT / "ebpf"],
        extensions=(".c", ".h"),
    ),
    LanguageIngestionConfig(
        language="python",
        directories=[],
        extensions=(".py",),
        explicit_files=[REPO_ROOT / "run_exp.py"],
        allowed_file_names={"run_exp.py"},
        excluded_file_names={"eval.py"},
    ),
    LanguageIngestionConfig(
        language="bash",
        directories=[REPO_ROOT / "workload"],
        extensions=(".sh",),
        explicit_files=[REPO_ROOT / "run.sh"],
    ),
]
```

`REPO_ROOT` automatically resolves to the repository root (the parent of `rag_codebase`), so the configured directories/files work no matter where you run the CLI from.

- `directories`: Each directory is crawled recursively; subdirectories are included automatically.
- `extensions`: Only files with these suffixes are considered for that language.
- `explicit_files`: Optional list of paths that should always be included (e.g., the standalone `run_exp.py` outside the Go/C directories).
- `allowed_file_names` / `excluded_file_names`: Optional allow/deny basenames so you can keep `run_exp.py` but drop `eval.py`.
- Setting `directories=[]` and relying on `explicit_files` is a good way to target only a handful of scripts.
- You can add/remove language blocks or tweak chunk sizes globally via the CLI flags.

## Metadata enrichment

Each chunk carries additional metadata to improve retrieval quality and debugging:

- `file_path`, `relative_path`, `file_name`: allow the LLM (and you) to see where a chunk came from, and let future retrievers filter/boost specific files.
- A `[File: ...]` banner is prepended to each chunk’s text so file-based questions (e.g., “show `agent_backup.c`”) hit the right documents during retrieval.
- `source_language`: tracks which language pipeline produced the chunk.
- `symbol_name`: best-effort extraction of the first function/class name detected inside the chunk (works for Go, C, and Python).

These are populated automatically during ingestion, so downstream consumers can log or filter on them without extra work.

## Switching chunking modes

At the top of `main.py` you'll find:

```python
USE_CODE_SPLITTER = True
DEFAULT_CODE_CHUNK_LINES = 40
DEFAULT_CODE_CHUNK_OVERLAP_LINES = 15
DEFAULT_TOKEN_CHUNK_SIZE = 800
DEFAULT_TOKEN_CHUNK_OVERLAP = 80
```

- Set `USE_CODE_SPLITTER = False` to run a single ingestion pass with `TokenTextSplitter`. The same file selection logic runs, but every document is chunked by tokens instead of language syntax.
- The CLI's `--chunk-size/--chunk-overlap` default to the appropriate constants depending on the current mode, so you can flip the flag and immediately compare behaviors without rewriting arguments.

## How the Scaffold Works

1. For each configured language, `SimpleDirectoryReader` loads the explicitly selected files (recursing through the listed directories).
2. `CodeSplitter` walks the syntax tree for each language block and emits overlapping, function-aware chunks.
3. The resulting nodes feed a `VectorStoreIndex`, which is persisted locally so you do not have to rebuild for every question.
4. Queries reload the stored index, retrieve the most relevant chunks, and hand them—plus the user question—to `gpt-5` through the OpenAI endpoint.

Everything is intentionally minimal so you can adapt it to your workflows.

## Ideas for Extensions

- **Index freshness**: Add change detection (git hashes, file mtimes) so incremental runs only re-chunk updated files.
- **Smarter chunking**: Swap in AST- or function-aware splitters, or add metadata such as file paths/functions to influence retrieval scoring.
- **Richer retrieval**: Layer in hybrid search (BM25 + vectors) using LlamaIndex retrievers, or log retrieved contexts for debugging.
- **Custom pipelines**: Replace the default vector store with a hosted database (Qdrant, Pinecone, pgvector) to support multi-user query clients.
- **UX improvements**: Wrap the CLI with a FastAPI/Streamlit UI, add conversation memory, or integrate with developer chat tools.
- **Testing**: Create smoke tests that run against a tiny fixture repo to guard against embedding/splitter regressions.

Feel free to expand `main.py` or introduce new modules as your needs grow—the scaffold is structured so you can slot in additional ingestion steps, retrievers, and query-time helpers without rewriting everything.
