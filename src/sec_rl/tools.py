from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool
from tinker_cookbook.tool_use.types import Tool

from sec_rl.config import HarnessConfig
from sec_rl.retrieval import CorpusIndex, format_hits


@dataclass
class SearchSession:
    encountered_ids: set[str] = field(default_factory=set)
    curated_ids: list[str] = field(default_factory=list)
    finished: bool = False
    bm25_calls: int = 0
    grep_calls: int = 0
    read_calls: int = 0
    curate_calls: int = 0
    drop_calls: int = 0
    invalid_curations: int = 0
    # Tool calls whose Harmony header was not the canonical
    # `<|channel|>commentary to=functions.X <|constrain|>json<|message|>` form
    # (counted from the sampled tokens by the environment; the lenient parser
    # still executes them).
    off_format_calls: int = 0


class SearchTools:
    """Per-episode tools sharing an encountered-document pool and a curated set.

    The model incrementally builds its output with curate/drop_curated and ends
    the episode with finish. The curated set is what gets scored, whether or
    not finish is ever called.
    """

    def __init__(self, index: CorpusIndex, config: HarnessConfig, session: SearchSession):
        self.index = index
        self.config = config
        self.session = session

    def specs(self) -> list[dict]:
        return [tool_impl.to_spec() for tool_impl in self.implementations()]

    def implementations(self) -> list[Tool]:
        return [
            self.bm25_search,
            self.grep_corpus,
            self.read_document,
            self.curate,
            self.drop_curated,
            self.finish,
        ]

    @tool
    async def bm25_search(
        self,
        query: Annotated[str, "A lexical search query; concise keywords usually work best."],
        k: Annotated[
            int | None,
            "Number of compact results to return (1-25). Omit to use the harness default.",
        ] = None,
    ) -> ToolResult:
        """Search the full corpus with BM25 lexical ranking."""
        k = self._resolve_k(k)
        hits = await asyncio.to_thread(
            self.index.bm25_search,
            query,
            k=k,
            snippet_chars=self.config.result_snippet_chars,
        )
        self.session.bm25_calls += 1
        self._remember(hit.doc_id for hit in hits)
        return simple_tool_result(
            format_hits(
                hits,
                query=query,
                encountered_count=len(self.session.encountered_ids),
                curated_count=len(self.session.curated_ids),
            ),
            metrics={"bm25_results": len(hits)},
        )

    @tool
    async def grep_corpus(
        self,
        pattern: Annotated[
            str,
            Field(
                description=(
                    "A Python-style regular expression to find exact terms, phrases, "
                    "or variants. Keep it short and targeted."
                ),
                max_length=160,
            ),
        ],
        case_sensitive: Annotated[bool, "Whether letter case must match exactly."] = False,
        k: Annotated[
            int | None,
            "Maximum compact matches to return (1-25). Omit to use the harness default.",
        ] = None,
    ) -> ToolResult:
        """Scan the full corpus with a bounded regular expression."""
        k = self._resolve_k(k)
        hits = await asyncio.to_thread(
            self.index.grep,
            pattern,
            k=k,
            snippet_chars=self.config.result_snippet_chars,
            case_sensitive=case_sensitive,
        )
        self.session.grep_calls += 1
        self._remember(hit.doc_id for hit in hits)
        return simple_tool_result(
            format_hits(
                hits,
                query=pattern,
                encountered_count=len(self.session.encountered_ids),
                curated_count=len(self.session.curated_ids),
            ),
            metrics={"grep_results": len(hits)},
        )

    @tool
    async def read_document(
        self,
        document_id: Annotated[
            str,
            "A document ID previously returned by one of the search tools.",
        ],
    ) -> ToolResult:
        """Read a candidate document's title and text."""
        self.session.read_calls += 1
        if document_id not in self.session.encountered_ids:
            return simple_tool_result(
                json.dumps(
                    {
                        "error": "document_not_encountered",
                        "message": "Search for this document before reading it.",
                    }
                )
            )
        document = self.index.read(document_id)
        if document is None:
            return simple_tool_result(json.dumps({"error": "unknown_document"}))
        content = document.text[: self.config.read_max_chars]
        return simple_tool_result(
            json.dumps(
                {
                    "document_id": document.doc_id,
                    "title": document.title,
                    "text": content,
                    "truncated": len(content) < len(document.text),
                },
                ensure_ascii=False,
            )
        )

    @tool
    async def curate(
        self,
        document_ids: Annotated[
            list[str],
            "Document IDs to add to the curated evidence set. Must have been returned "
            "by a search tool earlier in this episode.",
        ],
    ) -> ToolResult:
        """Add relevant documents to the curated set that is returned as your output."""
        self.session.curate_calls += 1
        requested = list(dict.fromkeys(document_ids))
        invalid = [doc_id for doc_id in requested if doc_id not in self.session.encountered_ids]
        valid = [doc_id for doc_id in requested if doc_id in self.session.encountered_ids]
        already = [doc_id for doc_id in valid if doc_id in self.session.curated_ids]
        new_ids = [doc_id for doc_id in valid if doc_id not in self.session.curated_ids]
        capacity = self.config.max_curated_docs - len(self.session.curated_ids)
        added, over_capacity = new_ids[: max(capacity, 0)], new_ids[max(capacity, 0) :]
        self.session.curated_ids.extend(added)
        if invalid or over_capacity:
            self.session.invalid_curations += 1
        payload: dict[str, object] = {
            "added": added,
            "curated_count": len(self.session.curated_ids),
            "curated_ids": list(self.session.curated_ids),
        }
        if already:
            payload["already_curated"] = already
        if invalid:
            payload["invalid_document_ids"] = invalid[:20]
            payload["message"] = "Only curate IDs returned by a search or grep tool."
        if over_capacity:
            payload["dropped_over_capacity"] = over_capacity[:20]
            payload["maximum"] = self.config.max_curated_docs
        return simple_tool_result(
            json.dumps(payload, ensure_ascii=False),
            metrics={"curated": len(added)},
        )

    @tool
    async def drop_curated(
        self,
        document_ids: Annotated[
            list[str],
            "Document IDs to remove from the curated evidence set.",
        ],
    ) -> ToolResult:
        """Remove documents from the curated set (e.g. redundant or off-topic ones)."""
        self.session.drop_calls += 1
        requested = list(dict.fromkeys(document_ids))
        removed = [doc_id for doc_id in requested if doc_id in self.session.curated_ids]
        not_curated = [doc_id for doc_id in requested if doc_id not in self.session.curated_ids]
        self.session.curated_ids = [
            doc_id for doc_id in self.session.curated_ids if doc_id not in set(removed)
        ]
        payload: dict[str, object] = {
            "removed": removed,
            "curated_count": len(self.session.curated_ids),
            "curated_ids": list(self.session.curated_ids),
        }
        if not_curated:
            payload["not_curated"] = not_curated[:20]
        return simple_tool_result(json.dumps(payload, ensure_ascii=False))

    @tool
    async def finish(self) -> ToolResult:
        """End the search. The curated set is returned as the final evidence."""
        self.session.finished = True
        return simple_tool_result(
            json.dumps(
                {
                    "accepted": True,
                    "curated_count": len(self.session.curated_ids),
                    "curated_ids": list(self.session.curated_ids),
                },
                ensure_ascii=False,
            ),
            should_stop=True,
            metrics={"curated_final": len(self.session.curated_ids)},
        )

    def _remember(self, ids: Iterable[str]) -> None:
        self.session.encountered_ids.update(ids)

    def _resolve_k(self, value: int | None) -> int:
        if value is not None and value < 1:
            raise ValueError("k must be positive")
        return min(value or self.config.search_default_k, self.config.search_max_k)
