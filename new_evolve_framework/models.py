"""
Data structures for the idea/program evolution loop.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_ts() -> float:
    return time.time()


def _gen_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ProgramRecord:
    """Single evaluated program attempt."""

    id: str
    status: str
    code: str
    metrics: Dict[str, Any]
    error_text: Optional[str] = None
    reasoning: Optional[str] = None
    eval_artifacts: Dict[str, Any] = field(default_factory=dict)
    generation_order: int = 0
    parent_program_id: Optional[str] = None
    timestamp: float = field(default_factory=_now_ts)

    @classmethod
    def create(
        cls,
        code: str,
        metrics: Dict[str, Any],
        *,
        status: str = "success",
        error_text: Optional[str] = None,
        reasoning: Optional[str] = None,
        eval_artifacts: Optional[Dict[str, Any]] = None,
        generation_order: int = 0,
        parent_program_id: Optional[str] = None,
    ) -> "ProgramRecord":
        return cls(
            id=_gen_id(),
            status=status,
            code=code,
            metrics=metrics,
            error_text=error_text,
            reasoning=reasoning,
            eval_artifacts=eval_artifacts or {},
            generation_order=generation_order,
            parent_program_id=parent_program_id,
        )


@dataclass
class IdeaNode:
    """Idea/design container with its evaluated programs."""

    id: str
    idea_payload: Optional[Dict[str, Any]]
    idea_summary: str
    programs: List[str] = field(default_factory=list)
    best_program_id: Optional[str] = None
    best_score: Optional[float] = None
    embedding: Optional[List[float]] = None
    artifacts_dir: Optional[str] = None
    idea_iterations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=_now_ts)

    @classmethod
    def create(
        cls,
        *,
        idea_payload: Optional[Dict[str, Any]],
        idea_summary: str,
        artifacts_dir: Optional[Path] = None,
    ) -> "IdeaNode":
        return cls(
            id=_gen_id(),
            idea_payload=idea_payload,
            idea_summary=idea_summary,
            artifacts_dir=str(artifacts_dir) if artifacts_dir else None,
        )

    def update_best(self, program_id: str, score: float) -> None:
        self.best_program_id = program_id
        self.best_score = score
