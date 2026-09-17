from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RewardName = Literal["f1", "f2", "f4", "f4s"]


@dataclass(frozen=True)
class HarnessConfig:
    data_dir: Path
    search_default_k: int = 10
    search_max_k: int = 25
    result_snippet_chars: int = 220
    read_max_chars: int = 4_000
    max_curated_docs: int = 30
    max_turns: int = 40
    max_tool_calls: int = 128
    max_trajectory_tokens: int = 65_536
    max_generation_tokens: int = 2_048
    # An episode that ends with an empty curated set scores the flat bad-ending
    # reward (-0.2) instead of F-beta 0 minus penalties, so finishing at once
    # with nothing curated (F-beta 0) can never beat an honest timeout (-0.1).
    penalize_empty_finish: bool = True
    # -0.02 per curate call carrying an unseen ID or overflowing the cap. Off
    # by default: it is a format nit, not part of the retrieval objective.
    penalize_invalid_curations: bool = False
    # -0.1 when the episode ends without a finish call (text reply, turn cap,
    # graded overflow). Off by default because Harness-1 scores those endings
    # the same as end_search.
    penalize_unfinished: bool = False
    # Subtracted once per episode when any tool call in it was emitted with an
    # off-form Harmony header (wrong channel, missing <|constrain|>, `code`
    # instead of `json`, ...). 0 = off. The lenient parser executes such calls
    # either way; this term only rewards format discipline.
    format_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.format_penalty < 0:
            raise ValueError("format_penalty must be non-negative")
        if self.search_default_k < 1 or self.search_default_k > self.search_max_k:
            raise ValueError("search_default_k must be between 1 and search_max_k")
        if self.search_max_k > 25:
            raise ValueError("search_max_k cannot exceed the tool schema limit of 25")
        if self.max_curated_docs < 1:
            raise ValueError("max_curated_docs must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "openai/gpt-oss-20b"
    renderer_name: str = "gpt_oss_no_sysprompt"
    learning_rate: float = 1e-4
    lora_rank: int = 32
    group_size: int = 8
    groups_per_batch: int = 64
    max_steps: int = 16
    eval_every: int = 0
    eval_size: int = 32
    eval_group_size: int = 4
    save_every: int = 1
    seed: int = 42
    # Skip this many groups_per_batch-sized batches of the deterministic train
    # order before step 0. tinker-cookbook restarts the data order at batch 0 for
    # any run with a fresh log_path, even when it loads weights from a checkpoint.
    train_start_batch: int = 0
    # Passes over the train queries the dataset exposes; the cookbook trains
    # min(max_steps, batches_per_epoch * train_epochs) batches. Epochs after the
    # first use a seeded re-shuffle of the query order.
    train_epochs: int = 1

    def __post_init__(self) -> None:
        if self.train_start_batch < 0:
            raise ValueError("train_start_batch must be non-negative")
        if self.train_epochs < 1:
            raise ValueError("train_epochs must be positive")
        if self.eval_size < 1:
            raise ValueError("eval_size must be positive")
        if self.eval_group_size < 1:
            raise ValueError("eval_group_size must be positive")


DEFAULT_DATA_DIR = Path("data/sec_subset")
DEFAULT_RAW_DIR = Path("data/raw")
