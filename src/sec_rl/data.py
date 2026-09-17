from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Split = Literal["train", "dev", "test"]
QUERY_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    text: str
    qrels: dict[str, int]
    fact_groups: tuple[FactGroup, ...] = ()


@dataclass(frozen=True)
class FactGroup:
    """One required fact and its interchangeable supporting chunks."""

    fact: str
    chunk_ids: tuple[str, ...]
    is_final_answer: bool = False


def normalize_query_text(text: str) -> str:
    """Normalize query text for exact cross-split leakage checks."""
    return QUERY_WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def load_corpus(data_dir: Path) -> list[Document]:
    path = data_dir / "corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `sec-rl prepare-sec` first.")
    documents: list[Document] = []
    for row in _read_jsonl(path):
        documents.append(
            Document(
                doc_id=str(row["_id"]),
                title=str(row.get("title") or ""),
                text=str(row.get("text") or ""),
            )
        )
    return documents


def load_queries(data_dir: Path) -> dict[str, str]:
    path = data_dir / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `sec-rl prepare-sec` first.")
    return {str(row["_id"]): str(row["text"]) for row in _read_jsonl(path)}


def load_qrels(data_dir: Path, split: Literal["train", "dev", "test"]) -> dict[str, dict[str, int]]:
    path = data_dir / "qrels" / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `sec-rl prepare-sec` first.")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    return dict(qrels)


def load_examples(data_dir: Path, split: Split) -> list[QueryExample]:
    queries = load_queries(data_dir)
    qrels = load_qrels(data_dir, split)
    query_ids = sorted(qrels)
    missing = [query_id for query_id in query_ids if query_id not in queries]
    if missing:
        raise ValueError(f"Queries missing text: {missing[:5]}")
    fact_groups = load_fact_groups(data_dir, split)
    return [
        QueryExample(
            query_id=query_id,
            text=queries[query_id],
            qrels=qrels[query_id],
            fact_groups=fact_groups.get(query_id, ()),
        )
        for query_id in query_ids
    ]


def load_fact_groups(
    data_dir: Path,
    split: Literal["train", "dev", "test"],
) -> dict[str, tuple[FactGroup, ...]]:
    """Load the fact-level labels used by the Harness-1 SEC dataset."""
    path = data_dir / "fact_qrels" / f"{split}.jsonl"
    if not path.exists():
        return {}
    groups: dict[str, list[FactGroup]] = defaultdict(list)
    for row in _read_jsonl(path):
        chunk_ids = tuple(dict.fromkeys(str(value) for value in row["chunk_ids"]))
        if not chunk_ids:
            raise ValueError(f"Fact group for {row['query_id']} has no supporting chunks")
        groups[str(row["query_id"])].append(
            FactGroup(
                fact=str(row["fact"]),
                chunk_ids=chunk_ids,
                is_final_answer=bool(row.get("is_final_answer", False)),
            )
        )
    return {query_id: tuple(values) for query_id, values in groups.items()}
