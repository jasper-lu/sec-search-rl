from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex

from sec_rl.data import Document

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    snippet: str
    score: float


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class CorpusIndex:
    """SQLite FTS5 BM25 index over the prepared SEC chunk corpus."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self._sqlite_path = self.data_dir / "lexical.sqlite3"
        if not self._sqlite_path.exists():
            raise FileNotFoundError(f"Missing {self._sqlite_path}. Run `sec-rl prepare-sec` first.")
        self._connection_local = threading.local()

    def bm25_search(self, query: str, *, k: int, snippet_chars: int) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in dict.fromkeys(query_tokens)
        )
        # Rank on ids only, then fetch the k texts by key: selecting title/text
        # inside the MATCH query makes FTS5 materialise content for every
        # matching row before sorting, which throttles concurrent throughput.
        connection = self._connection()
        ranked = connection.execute(
            """
            SELECT doc_id, bm25(documents_fts, 0.0, 2.0, 1.0) AS rank
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (expression, k),
        ).fetchall()
        texts = self._texts([str(row[0]) for row in ranked])
        return [
            SearchHit(
                doc_id=str(row[0]),
                title=texts[str(row[0])][0],
                snippet=_query_snippet(texts[str(row[0])][1], query, snippet_chars),
                score=-float(row[1]),
            )
            for row in ranked
            if str(row[0]) in texts
        ]

    def _texts(self, doc_ids: list[str]) -> dict[str, tuple[str, str]]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" * len(doc_ids))
        rows = (
            self._connection()
            .execute(
                f"SELECT doc_id, title, text FROM documents WHERE doc_id IN ({placeholders})",
                doc_ids,
            )
            .fetchall()
        )
        return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}

    def grep(
        self,
        pattern: str,
        *,
        k: int,
        snippet_chars: int,
        case_sensitive: bool = False,
    ) -> list[SearchHit]:
        if not pattern.strip():
            return []
        if len(pattern) > 160:
            raise ValueError("grep pattern must be at most 160 characters")
        flags = 0 if case_sensitive else regex.IGNORECASE
        try:
            compiled = regex.compile(pattern, flags)
        except regex.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc

        terms = sorted(set(tokenize(pattern)), key=len, reverse=True)
        if not terms:
            raise ValueError("grep pattern must contain at least one letter or number")
        expression = " OR ".join(f'"{term}"' for term in terms[:8])
        # Same id-first pattern as bm25_search: rank candidates on ids, then
        # pull texts in small batches and stop scanning as soon as k match.
        candidate_ids = [
            str(row[0])
            for row in self._connection()
            .execute(
                """
            SELECT doc_id
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts, 0.0, 2.0, 1.0)
            LIMIT 5000
            """,
                (expression,),
            )
            .fetchall()
        ]
        hits: list[SearchHit] = []
        batch = 200
        for start in range(0, len(candidate_ids), batch):
            chunk = candidate_ids[start : start + batch]
            texts = self._texts(chunk)
            for doc_id in chunk:
                if doc_id not in texts:
                    continue
                document = Document(doc_id=doc_id, title=texts[doc_id][0], text=texts[doc_id][1])
                try:
                    # Wall-clock timeout; generous enough not to fire under GIL
                    # contention from many concurrent episodes.
                    match = compiled.search(document.searchable_text, timeout=0.05)
                except TimeoutError as exc:
                    raise ValueError("regular expression timed out; use a simpler pattern") from exc
                if match is None:
                    continue
                hits.append(
                    SearchHit(
                        doc_id=document.doc_id,
                        title=document.title,
                        snippet=_window(document.searchable_text, match.start(), snippet_chars),
                        score=1.0,
                    )
                )
                if len(hits) >= k:
                    return hits
        return hits

    def read(self, doc_id: str) -> Document | None:
        row = (
            self._connection()
            .execute("SELECT doc_id, title, text FROM documents WHERE doc_id = ?", (doc_id,))
            .fetchone()
        )
        if row is None:
            return None
        return Document(doc_id=str(row[0]), title=str(row[1]), text=str(row[2]))

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._connection_local, "connection", None)
        if connection is None:
            uri = f"file:{self._sqlite_path}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.execute("PRAGMA query_only = ON")
            self._connection_local.connection = connection
        return connection


@lru_cache(maxsize=4)
def get_index(data_dir: str) -> CorpusIndex:
    return CorpusIndex(Path(data_dir))


def _query_snippet(text: str, query: str, max_chars: int) -> str:
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokenize(query)]
    valid_positions = [position for position in positions if position >= 0]
    start = min(valid_positions) if valid_positions else 0
    return _window(text, start, max_chars)


def _window(text: str, match_start: int, max_chars: int) -> str:
    if len(text) <= max_chars:
        return " ".join(text.split())
    margin = max_chars // 3
    start = max(0, match_start - margin)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = " ".join(text[start:end].split())
    return f"{'…' if start else ''}{snippet}{'…' if end < len(text) else ''}"


def format_hits(
    hits: Sequence[SearchHit],
    *,
    query: str,
    encountered_count: int | None = None,
    curated_count: int | None = None,
) -> str:
    payload = {
        "query": query,
        "candidate_count": len(hits),
        "count": len(hits),
        "results": [
            {
                "document_id": hit.doc_id,
                "title": hit.title,
                "snippet": hit.snippet,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ],
    }
    if encountered_count is not None:
        search_state: dict[str, object] = {
            "encountered_candidates": encountered_count,
            "next_step": (
                "Curate promising documents from these results now with "
                "curate(document_ids). Most failed episodes curate too few "
                "documents - add EVERY plausibly relevant document, not just the "
                "single best one. Then either search for one genuinely missing "
                "clue, or call finish if the curated set covers all distinct clues."
            ),
        }
        if curated_count is not None:
            search_state["curated_count"] = curated_count
        payload["search_state"] = search_state
    return json.dumps(payload, ensure_ascii=False)
