
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Node:
    node_id: int
    parent_id: Optional[int]
    depth: int
    action_text: str

    steps: List[str] = field(default_factory=list)
    text: str = ""
    prompt_tokens_len: int = 0
    cum_generated_tokens: int = 0

    finish_reason: Optional[str] = None
    terminal_reason: Optional[str] = None
    is_terminal: bool = False
    is_fully_expanded: bool = False

    children: List[int] = field(default_factory=list)
    Q: float = 0.0
    N: int = 0

    prm_step_scores: List[float] = field(default_factory=list)
    prm_last_score: float = 0.0
    prm_min_score: float = 0.0
    prm_mean_score: float = 0.0
    prm_prod_score: float = 0.0
    search_value: float = 0.0

    is_answer_candidate: bool = False
    pred_answer: Optional[str] = None
