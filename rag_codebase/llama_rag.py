from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter, TokenTextSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


@dataclass
class LanguageIngestionConfig:
    language: str
    directories: list[Path]
    extensions: tuple[str, ...]
    allowed_file_names: Optional[set[str]] = None
    excluded_file_names: Optional[set[str]] = None
    explicit_files: Optional[list[Path]] = None


# REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path("/home/benny/evolve_agent/agentic-ebpf/pebble-server")


# Toggle this flag to switch between language-aware chunking and plain token splitting.
USE_CODE_SPLITTER = True

RAG_SYSTEM_PROMPT = (
    "You are a retrieval QA assistant for the Pebble high-performance server repository."
    "Always respond in JSON with two fields: "
    '"answer" (succinct explanation referencing files and functions) and '
    '"code_snippets" (an array of objects each with "path" and a short \"snippet\" of supporting code). '
    "Select the most relevant snippets from the retrieved context and trim them to only the lines needed to "
    "support the answer."
)

# Default chunk sizes for each mode.
DEFAULT_CODE_CHUNK_LINES = 40
DEFAULT_CODE_CHUNK_OVERLAP_LINES = 15
DEFAULT_TOKEN_CHUNK_SIZE = 800
DEFAULT_TOKEN_CHUNK_OVERLAP = 80


# Update these paths to match the directories/files you want to index.
LANGUAGE_CONFIGS: list[LanguageIngestionConfig] = [
    LanguageIngestionConfig(
        language="go",
        directories=[
            REPO_ROOT / "cmd",
            REPO_ROOT / "internal" / "server",
        ],
        extensions=(".go",),
    ),
    LanguageIngestionConfig(
        language="c",
        directories=[
            REPO_ROOT / "ebpf",
        ],
        extensions=(".c", ".h"),
    ),
    LanguageIngestionConfig(
        language="python",
        directories=[
        ],
        extensions=(".py",),
        explicit_files=[
            REPO_ROOT / "run_exp.py",
        ],
        allowed_file_names={"run_exp.py"},
        excluded_file_names={
            "eval.py",
        },
    ),
    LanguageIngestionConfig(
        language="bash",
        directories=[
            REPO_ROOT / "workload",
        ],
        extensions=(".sh",),
        explicit_files=[
            REPO_ROOT / "run.sh",
        ],
    ),
]


def configure_llamaindex(api_key: str) -> None:
    """
    Configure LlamaIndex defaults for both the LLM and embedding model.

    Args:
        api_key: OpenAI API key used for the chat/model endpoints.
    """
    llm_kwargs = {"model": "gpt-5.1", "api_key": api_key, "temperature": 0.3}
    try:
        Settings.llm = OpenAI(system_prompt=RAG_SYSTEM_PROMPT, **llm_kwargs)
    except TypeError:
        Settings.llm = OpenAI(**llm_kwargs)
        try:
            Settings.llm.system_prompt = RAG_SYSTEM_PROMPT  # type: ignore[attr-defined]
        except Exception:
            pass
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=api_key,
    )


def discover_files(config: LanguageIngestionConfig) -> list[Path]:
    """
    Return the list of files that match a language config.

    Args:
        config: Language-specific directories, extensions, and filters.
    """
    matched_files: list[Path] = []
    seen: set[Path] = set()

    def _should_include(path: Path) -> bool:
        if config.allowed_file_names and path.name not in config.allowed_file_names:
            return False
        if config.excluded_file_names and path.name in config.excluded_file_names:
            return False
        return True

    for directory in config.directories:
        if not directory.exists():
            print(f"[{config.language}] Skipping missing directory: {directory}")
            continue
        for candidate in directory.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix not in config.extensions:
                continue
            if not _should_include(candidate):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            matched_files.append(resolved)
            seen.add(resolved)

    if config.explicit_files:
        for file_path in config.explicit_files:
            resolved = file_path.resolve()
            if not resolved.exists():
                print(f"[{config.language}] Explicit file not found: {file_path}")
                continue
            if resolved.suffix not in config.extensions:
                print(
                    f"[{config.language}] Explicit file {file_path} ignored "
                    f"(extension not in {config.extensions})."
                )
                continue
            if not _should_include(resolved):
                continue
            if resolved in seen:
                continue
            matched_files.append(resolved)
            seen.add(resolved)

    return matched_files


def load_documents_for_config(config: LanguageIngestionConfig) -> list:
    """
    Read matching files into LlamaIndex Document objects with metadata.

    Args:
        config: Language configuration describing source files.
    """
    files = discover_files(config)
    if not files:
        print(f"[{config.language}] No files matched. Nothing to index for this run.")
        return []
    reader = SimpleDirectoryReader(
        input_files=[str(path) for path in files],
        file_metadata=lambda path: build_file_metadata(Path(path), config.language),
    )
    documents = reader.load_data()
    for doc in documents:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata.setdefault("language", config.language)
    return documents


def build_file_metadata(path: Path, language: str) -> dict:
    """
    Attach basic metadata (paths, language) for each file.

    Args:
        path: Absolute path to the file.
        language: Language label associated with the file.
    """
    absolute = path.resolve()
    metadata = {
        "file_path": str(absolute),
        "file_name": absolute.name,
        "language": language,
    }
    relative = safe_relative_path(absolute)
    if relative:
        metadata["relative_path"] = relative
    return metadata


def safe_relative_path(path: Path) -> Optional[str]:
    """
    Return repo-relative path for display, or None if outside REPO_ROOT.

    Args:
        path: Absolute path that may or may not live under REPO_ROOT.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return None


def extract_symbol_name(text: str, language: str) -> Optional[str]:
    """
    Best-effort regex extractor for leading symbol/function names.

    Args:
        text: Chunk text to inspect.
        language: Source language that determines regex patterns.
    """
    patterns = {
        "python": [r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"],
        "go": [r"^\s*func\s+(?:\([\w\*\s,]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)"],
        "c": [
            r"^\s*(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w\s\*]+?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        ],
    }
    for pattern in patterns.get(language, []):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)
    return None


def enrich_node_metadata(node, doc_metadata: dict, language: str) -> None:
    """
    Copy per-file metadata onto nodes and annotate source paths and symbols.

    Args:
        node: LlamaIndex node/chunk object to enrich.
        doc_metadata: Mapping of document IDs to their metadata dicts.
        language: Source language label for the chunk.
    """
    file_path = node.metadata.get("file_path")
    if not file_path and node.ref_doc_id in doc_metadata:
        source_meta = doc_metadata[node.ref_doc_id]
        file_path = source_meta.get("file_path")
        node.metadata.update(source_meta)
    if file_path:
        node.metadata.setdefault("file_path", file_path)
        node.metadata.setdefault("file_name", Path(file_path).name)
        rel = node.metadata.get("relative_path")
        if not rel:
            relative = safe_relative_path(Path(file_path))
            if relative:
                node.metadata["relative_path"] = relative
    display_path = node.metadata.get("relative_path") or node.metadata.get("file_path")
    original_text = node.text
    if display_path and not node.metadata.get("_file_banner_added"):
        node.text = f"[File: {display_path}]\n{node.text}"
        node.metadata["_file_banner_added"] = True
    node.metadata.setdefault("source_language", language)
    symbol = extract_symbol_name(original_text, language)
    if symbol:
        node.metadata.setdefault("symbol_name", symbol)


def build_index(
    persist_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Build and persist a vector index from the configured language sources.

    Args:
        persist_dir: Directory where the index should be stored.
        chunk_size: Chunk size (lines or tokens depending on mode).
        chunk_overlap: Overlap between adjacent chunks.
    """
    documents_by_language: Dict[str, Tuple[list, Dict[str, dict]]] = {}
    for config in LANGUAGE_CONFIGS:
        documents = load_documents_for_config(config)
        if documents:
            doc_meta_map = {doc.doc_id: doc.metadata or {} for doc in documents}
            documents_by_language[config.language] = (documents, doc_meta_map)

    if not documents_by_language:
        raise RuntimeError(
            "No files were ingested. Update LANGUAGE_CONFIGS to point to valid directories/files."
        )

    all_nodes = []
    if USE_CODE_SPLITTER:
        for config in LANGUAGE_CONFIGS:
            language_payload = documents_by_language.get(config.language)
            if not language_payload:
                continue
            documents, doc_meta_map = language_payload
            splitter = CodeSplitter(
                language=config.language,
                chunk_lines=chunk_size,
                chunk_lines_overlap=chunk_overlap,
            )
            pipeline = IngestionPipeline(
                transformations=[splitter],
            )
            nodes = pipeline.run(documents=documents)
            for node in nodes:
                enrich_node_metadata(node, doc_meta_map, config.language)
            print(
                f"[{config.language}] Added {len(nodes)} chunks from {len(documents)} files."
            )
            all_nodes.extend(nodes)
    else:
        all_documents = []
        combined_meta: Dict[str, dict] = {}
        for docs, doc_meta in documents_by_language.values():
            all_documents.extend(docs)
            combined_meta.update(doc_meta)
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        pipeline = IngestionPipeline(transformations=[splitter])
        nodes = pipeline.run(documents=all_documents)
        for node in nodes:
            language = combined_meta.get(node.ref_doc_id, {}).get("language", "unknown")
            enrich_node_metadata(node, combined_meta, language)
        print(f"[token] Added {len(nodes)} chunks from {len(all_documents)} files.")
        all_nodes.extend(nodes)

    if not all_nodes:
        raise RuntimeError(
            "No chunks were generated. Check LANGUAGE_CONFIGS or adjust chunk parameters."
        )

    index = VectorStoreIndex(all_nodes)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))


def query_index(persist_dir: Path, question: str, top_k: int = 3) -> str:
    """
    Load a persisted index and answer a natural-language question.

    Args:
        persist_dir: Location of the saved vector index.
        question: Natural language query to answer.
        top_k: How many similar chunks to retrieve.
    """
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Index directory {persist_dir} not found. Run the 'index' command first."
        )
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    engine = index.as_query_engine(similarity_top_k=top_k)
    response = engine.query(question)

    # Debug logging: show retrieved chunks before returning.
    # for idx, source_node in enumerate(getattr(response, "source_nodes", []), 1):
    #     metadata = source_node.node.metadata or {}
    #     file_path = metadata.get("relative_path") or metadata.get("file_path") or "<unknown>"
    #     score = getattr(source_node, "score", None)
    #     print(f"[RAG chunk {idx}] {file_path} (score={score})")
    #     print(source_node.node.text[:500])
    #     print("-" * 40)

    return str(response)


def parse_args() -> argparse.Namespace:
    """Create the CLI parser for the indexing/query commands."""
    parser = argparse.ArgumentParser(
        description="Mini RAG app that indexes a codebase with LlamaIndex."
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("rag_storage"),
        help="Directory to store the vector index.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CODE_CHUNK_LINES if USE_CODE_SPLITTER else DEFAULT_TOKEN_CHUNK_SIZE,
        help=(
            "Chunk size (lines when using CodeSplitter, tokens when using TokenTextSplitter)."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CODE_CHUNK_OVERLAP_LINES
        if USE_CODE_SPLITTER
        else DEFAULT_TOKEN_CHUNK_OVERLAP,
        help=("Overlap (lines for CodeSplitter, tokens for TokenTextSplitter)."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Index the given code directory.")

    query_parser = subparsers.add_parser("query", help="Query the existing index.")
    query_parser.add_argument("question", type=str, help="Question about the codebase.")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many similar chunks to retrieve when answering.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    configure_llamaindex(api_key)

    if args.command == "index":
        build_index(
            persist_dir=args.persist_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        mode = "CodeSplitter" if USE_CODE_SPLITTER else "TokenTextSplitter"
        unit = "lines" if USE_CODE_SPLITTER else "tokens"
        print(
            f"Index stored at {args.persist_dir} "
            f"({mode}, chunk_size={args.chunk_size} {unit}, overlap={args.chunk_overlap})."
        )
    elif args.command == "query":
        answer = query_index(
            persist_dir=args.persist_dir,
            question=args.question,
            top_k=args.top_k,
        )
        print(answer)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
