from __future__ import annotations

import bisect
import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from sec_rl.data import normalize_query_text

HF_ROOT = "https://huggingface.co/datasets/pat-jj/harness-1-train-data/resolve/main"
QUERY_FILE = "data/train-00000-of-00001.parquet"
CORPUS_FILES = tuple(f"corpora/sec/train/train-{index:05d}.parquet" for index in range(43))
SEC_CORPUS_SIZE = 2_115_106
PREPARED_FILES = (
    "corpus.jsonl",
    "queries.jsonl",
    "query_metadata.jsonl",
    "qrels/train.tsv",
    "qrels/dev.tsv",
    "fact_qrels/train.jsonl",
    "fact_qrels/dev.jsonl",
    "lexical.sqlite3",
    "dataset.json",
)


def download_sec(raw_dir: Path, *, force: bool = False) -> None:
    """Download the public Harness-1 query table and SEC corpus shards into ``raw_dir``."""
    destinations = [(QUERY_FILE, raw_dir / "queries.parquet")]
    destinations.extend(
        (relative, raw_dir / "corpus" / Path(relative).name) for relative in CORPUS_FILES
    )
    for relative, destination in destinations:
        if destination.exists() and not force:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with httpx.stream(
            "GET", f"{HF_ROOT}/{relative}", follow_redirects=True, timeout=300.0
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        temporary.replace(destination)


def prepare_sec(
    data_dir: Path,
    *,
    raw_dir: Path,
    train_size: int = 256,
    dev_size: int = 64,
    random_distractors: int = 60_000,
    neighbor_radius: int = 4,
    seed: int = 42,
    force: bool = False,
) -> dict[str, object]:
    """Build a fact-level SEC subset with family-safe query splits.

    Exactly one 3-, 5-, or 7-fact variant is selected per semantic query family.
    Connected components induced by shared gold chunks are assigned wholly away
    from the opposite split. The corpus includes every gold alternative, nearby
    chunks from the same filings, and a stable random background sample.
    """
    if train_size < 1 or dev_size < 1:
        raise ValueError("train_size and dev_size must be positive")
    if random_distractors < 0 or neighbor_radius < 0:
        raise ValueError("distractor settings cannot be negative")
    data_dir.mkdir(parents=True, exist_ok=True)
    if not force and all((data_dir / relative).exists() for relative in PREPARED_FILES):
        return json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))

    query_path = raw_dir / "queries.parquet"
    corpus_paths = [raw_dir / "corpus" / Path(value).name for value in CORPUS_FILES]
    missing = [str(path) for path in [query_path, *corpus_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Harness-1 SEC source files: {missing[:5]}")

    rows = _load_query_rows(query_path)
    # A few Harness-1 queries cite gold chunks that are missing from the
    # published corpus shards; they are skipped at selection time so the
    # corpus build below never has to fail on them.
    available_chunks = _corpus_chunk_ids(corpus_paths)
    selected = _select_queries(
        rows,
        train_size=train_size,
        dev_size=dev_size,
        seed=seed,
        available_chunks=available_chunks,
    )
    _write_query_outputs(data_dir, selected, seed=seed)
    corpus_summary = _build_corpus(
        data_dir,
        corpus_paths,
        selected,
        random_distractors=random_distractors,
        neighbor_radius=neighbor_radius,
        seed=seed,
    )

    split_counts = Counter(str(row["split"]) for row in selected)
    fact_counts = {
        split: dict(
            sorted(
                Counter(
                    len(row["fact_groups"]) for row in selected if row["split"] == split
                ).items()
            )
        )
        for split in ("train", "dev")
    }
    train_gold = _gold_ids(row for row in selected if row["split"] == "train")
    dev_gold = _gold_ids(row for row in selected if row["split"] == "dev")
    train_families = {str(row["family_id"]) for row in selected if row["split"] == "train"}
    dev_families = {str(row["family_id"]) for row in selected if row["split"] == "dev"}
    summary: dict[str, object] = {
        "dataset": "Harness-1 public SEC RL data (fact-level subset)",
        "source": "pat-jj/harness-1-train-data",
        "queries": dict(split_counts),
        "fact_groups_per_query": fact_counts,
        "documents": corpus_summary["documents"],
        "gold_chunks": corpus_summary["gold_chunks"],
        "same_filing_neighbor_distractors": corpus_summary["same_filing_neighbors"],
        "random_distractors": corpus_summary["random_distractors"],
        "requested_random_distractors": random_distractors,
        "neighbor_radius": neighbor_radius,
        "seed": seed,
        "embedding_index": False,
        "lexical_index": "SQLite FTS5 BM25",
        "split_guards": {
            "query_family_overlap": len(train_families & dev_families),
            "gold_chunk_overlap": len(train_gold & dev_gold),
            "normalized_query_text_overlap": len(
                {
                    normalize_query_text(str(row["query"]))
                    for row in selected
                    if row["split"] == "train"
                }
                & {
                    normalize_query_text(str(row["query"]))
                    for row in selected
                    if row["split"] == "dev"
                }
            ),
        },
        "scoring": {
            "precision": "submitted chunks in any gold fact / submitted chunks",
            "recall": "fact groups covered by any alternative chunk / fact groups",
        },
    }
    (data_dir / "dataset.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _load_query_rows(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(
        path,
        columns=["stage", "dataset_name", "query_id", "query", "answer", "document_ids_json"],
        filters=[("stage", "=", "rl"), ("dataset_name", "=", "sec")],
    )
    rows: list[dict[str, object]] = []
    for source in table.to_pylist():
        query_id = str(source["query_id"])
        suffix = query_id.rsplit("_", 1)
        family_id = suffix[0] if len(suffix) == 2 and suffix[1].isdigit() else query_id
        facts = json.loads(str(source["document_ids_json"]))
        rows.append(
            {
                "query_id": query_id,
                "family_id": family_id,
                "query": str(source["query"]),
                "answer": str(source.get("answer") or ""),
                "fact_groups": facts,
            }
        )
    return rows


class _UnionFind:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _corpus_chunk_ids(corpus_paths: list[Path]) -> set[str]:
    chunk_ids: set[str] = set()
    for path in corpus_paths:
        column = pq.read_table(path, columns=["chunk_id"]).column("chunk_id")
        chunk_ids.update(str(value) for value in column.to_pylist())
    return chunk_ids


def _select_queries(
    rows: list[dict[str, object]],
    *,
    train_size: int,
    dev_size: int,
    seed: int,
    available_chunks: set[str] | None = None,
) -> list[dict[str, object]]:
    def usable(row: dict[str, object]) -> bool:
        return available_chunks is None or _gold_ids([row]) <= available_chunks

    by_family: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_family[str(row["family_id"])][len(row["fact_groups"])] = row  # type: ignore[arg-type]

    union_find = _UnionFind(set(by_family))
    owner: dict[str, str] = {}
    for family_id, variants in by_family.items():
        all_chunks = {
            str(chunk_id)
            for row in variants.values()
            for fact in row["fact_groups"]  # type: ignore[union-attr]
            for chunk_id in fact["chunk_ids"]
        }
        for chunk_id in all_chunks:
            previous = owner.setdefault(chunk_id, family_id)
            union_find.union(family_id, previous)

    component = {family_id: union_find.find(family_id) for family_id in by_family}
    train_quotas = _fact_quotas(train_size)
    dev_quotas = _fact_quotas(dev_size)
    rng = random.Random(seed)

    selected: list[dict[str, object]] = []
    used_components: set[str] = set()
    used_texts: set[str] = set()

    def take(split: str, quotas: dict[int, int]) -> None:
        # Shuffled candidate lists are kept so a shortfall in one fact count
        # can be backfilled from the next easier one without re-drawing: the
        # RNG is consumed identically whether or not a quota is met, so any
        # smaller dataset built with the same seed is a prefix of a larger one.
        shuffled: dict[int, list[tuple[str, dict[str, object]]]] = {}
        shortfall = 0
        for fact_count in (7, 5, 3):
            candidates = [
                (family_id, variants[fact_count])
                for family_id, variants in by_family.items()
                if fact_count in variants
            ]
            rng.shuffle(candidates)
            shuffled[fact_count] = candidates
            taken = 0
            for family_id, row in candidates:
                text = normalize_query_text(str(row["query"]))
                root = component[family_id]
                if root in used_components or text in used_texts or not usable(row):
                    continue
                selected.append({**row, "split": split, "component_id": root})
                used_components.add(root)
                used_texts.add(text)
                taken += 1
                if taken == quotas[fact_count]:
                    break
            shortfall += quotas[fact_count] - taken
        if shortfall and split == "dev":
            raise RuntimeError(f"Could not fill the {split} fact-count quotas (short {shortfall})")
        for fact_count in (5, 3):
            if shortfall <= 0:
                break
            for family_id, row in shuffled[fact_count]:
                text = normalize_query_text(str(row["query"]))
                root = component[family_id]
                if root in used_components or text in used_texts or not usable(row):
                    continue
                selected.append({**row, "split": split, "component_id": root})
                used_components.add(root)
                used_texts.add(text)
                shortfall -= 1
                if shortfall == 0:
                    break
        if shortfall:
            raise RuntimeError(
                f"Could select only {sum(quotas.values()) - shortfall}/{sum(quotas.values())} "
                f"{split} queries: the source has too few independent query families"
            )

    # Reserve held-out components first; training can use any of the many remaining components.
    take("dev", dev_quotas)
    take("train", train_quotas)
    selected.sort(key=lambda row: (str(row["split"]), str(row["query_id"])))
    return selected


def _fact_quotas(total: int) -> dict[int, int]:
    weights = {3: 0.50, 5: 0.3125, 7: 0.1875}
    quotas = {key: int(total * value) for key, value in weights.items()}
    remaining = total - sum(quotas.values())
    for key in (3, 5, 7):
        if remaining <= 0:
            break
        quotas[key] += 1
        remaining -= 1
    return quotas


def _gold_ids(rows: Iterator[dict[str, object]] | list[dict[str, object]]) -> set[str]:
    return {
        str(chunk_id)
        for row in rows
        for fact in row["fact_groups"]  # type: ignore[union-attr]
        for chunk_id in fact["chunk_ids"]
    }


def _write_query_outputs(data_dir: Path, rows: list[dict[str, object]], *, seed: int) -> None:
    queries_path = data_dir / "queries.jsonl"
    metadata_path = data_dir / "query_metadata.jsonl"
    with (
        queries_path.open("w", encoding="utf-8") as queries,
        metadata_path.open("w", encoding="utf-8") as metadata,
    ):
        for row in sorted(rows, key=lambda value: str(value["query_id"])):
            queries.write(
                json.dumps({"_id": row["query_id"], "text": row["query"]}, ensure_ascii=False)
                + "\n"
            )
            metadata.write(json.dumps(row, ensure_ascii=False) + "\n")

    qrels_dir = data_dir / "qrels"
    facts_dir = data_dir / "fact_qrels"
    qrels_dir.mkdir(exist_ok=True)
    facts_dir.mkdir(exist_ok=True)
    for split in ("train", "dev", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        with (qrels_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("query-id", "corpus-id", "score"))
            for row in split_rows:
                for chunk_id in sorted(_gold_ids([row])):
                    writer.writerow((row["query_id"], chunk_id, 1))
        with (facts_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                for fact_index, fact in enumerate(row["fact_groups"]):  # type: ignore[union-attr]
                    handle.write(
                        json.dumps(
                            {
                                "query_id": row["query_id"],
                                "fact_id": fact_index,
                                **fact,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    schema = pa.schema(
        [
            ("query_id", pa.string()),
            ("family_id", pa.string()),
            ("component_id", pa.string()),
            ("split", pa.string()),
            ("query", pa.string()),
            ("answer", pa.string()),
            ("document_ids_json", pa.string()),
        ]
    )
    selected_table = pa.Table.from_pylist(
        [
            {
                "query_id": row["query_id"],
                "family_id": row["family_id"],
                "component_id": row["component_id"],
                "split": row["split"],
                "query": row["query"],
                "answer": row["answer"],
                "document_ids_json": json.dumps(row["fact_groups"], ensure_ascii=False),
            }
            for row in rows
        ],
        schema=schema,
    )
    pq.write_table(selected_table, data_dir / "selected_queries.parquet")


def _build_corpus(
    data_dir: Path,
    corpus_paths: list[Path],
    query_rows: list[dict[str, object]],
    *,
    random_distractors: int,
    neighbor_radius: int,
    seed: int,
) -> dict[str, int]:
    gold = _gold_ids(query_rows)
    gold_positions: dict[str, list[int]] = defaultdict(list)
    for chunk_id in gold:
        filing, chunk_index = _split_chunk_id(chunk_id)
        gold_positions[filing].append(chunk_index)
    for positions in gold_positions.values():
        positions.sort()

    database_tmp = data_dir / "lexical.sqlite3.partial"
    if database_tmp.exists():
        database_tmp.unlink()
    connection = sqlite3.connect(database_tmp)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )

    random_probability = min(1.0, random_distractors / SEC_CORPUS_SIZE)
    counts = Counter()
    corpus_path = data_dir / "corpus.jsonl"
    found_gold: set[str] = set()
    batch: list[tuple[str, str, str]] = []
    with corpus_path.open("w", encoding="utf-8") as corpus:
        for path in corpus_paths:
            for source_rows in _iter_rows(path):
                for row in source_rows:
                    chunk_id = str(row["chunk_id"])
                    filing, chunk_index = _split_chunk_id(chunk_id)
                    is_gold = chunk_id in gold
                    is_neighbor = _near_gold(
                        gold_positions.get(filing, []), chunk_index, neighbor_radius
                    )
                    is_random = _stable_random(seed, chunk_id) < random_probability
                    if not (is_gold or is_neighbor or is_random):
                        continue
                    text = str(row["document_text"])
                    metadata = json.loads(str(row.get("metadata_json") or "{}"))
                    title = _document_title(text, metadata)
                    batch.append((chunk_id, title, text))
                    corpus.write(
                        json.dumps(
                            {"_id": chunk_id, "title": title, "text": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if is_gold:
                        found_gold.add(chunk_id)
                        counts["gold_chunks"] += 1
                    elif is_neighbor:
                        counts["same_filing_neighbors"] += 1
                    else:
                        counts["random_distractors"] += 1
                    if len(batch) >= 2_000:
                        connection.executemany(
                            "INSERT OR IGNORE INTO documents(doc_id, title, text) VALUES (?, ?, ?)",
                            batch,
                        )
                        batch.clear()
            connection.commit()
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO documents(doc_id, title, text) VALUES (?, ?, ?)", batch
            )
        connection.execute(
            """
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                doc_id UNINDEXED,
                title,
                text,
                tokenize='porter unicode61 remove_diacritics 2'
            )
            """
        )
        connection.execute(
            "INSERT INTO documents_fts(doc_id, title, text) SELECT doc_id, title, text FROM documents"
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
        connection.commit()
        counts["documents"] = int(
            connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        )
    connection.close()

    missing = sorted(gold - found_gold)
    if missing:
        database_tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing)} selected gold chunks were absent: {missing[:10]}")
    database_tmp.replace(data_dir / "lexical.sqlite3")
    return dict(counts)


def _iter_rows(path: Path, batch_size: int = 4096) -> Iterator[list[dict[str, object]]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["chunk_id", "document_text", "metadata_json"],
    ):
        yield batch.to_pylist()


def _split_chunk_id(chunk_id: str) -> tuple[str, int]:
    filing, separator, suffix = chunk_id.rpartition("_")
    if not separator or not suffix.isdigit():
        return chunk_id, -1
    return filing, int(suffix)


def _near_gold(positions: list[int], value: int, radius: int) -> bool:
    if not positions or value < 0:
        return False
    index = bisect.bisect_left(positions, value)
    return any(
        abs(positions[position] - value) <= radius
        for position in (index - 1, index)
        if 0 <= position < len(positions)
    )


def _stable_random(seed: int, chunk_id: str) -> float:
    digest = hashlib.blake2b(f"{seed}\0{chunk_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _document_title(text: str, metadata: dict[str, object]) -> str:
    company = next(
        (
            line.removeprefix("Company:").strip()
            for line in text.splitlines()[:4]
            if line.startswith("Company:")
        ),
        str(metadata.get("ticker") or "SEC filing"),
    )
    form_type = str(metadata.get("form_type") or "filing")
    filing_date = str(metadata.get("filing_date") or "")
    return " · ".join(value for value in (company, form_type, filing_date) if value)
