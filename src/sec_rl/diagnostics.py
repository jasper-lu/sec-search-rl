from __future__ import annotations

from pathlib import Path

from sec_rl.data import Split, load_examples
from sec_rl.retrieval import CorpusIndex
from sec_rl.rewards import candidate_recall


def retrieval_diagnostics(
    *, data_dir: Path, split: Split, ks: list[int], limit: int | None = None
) -> dict[str, object]:
    examples = load_examples(data_dir, split)
    if limit is not None:
        examples = examples[:limit]
    checked_ks = sorted(set(ks))
    if not checked_ks or checked_ks[0] < 1 or checked_ks[-1] > 100:
        raise ValueError("diagnostic k values must be between 1 and 100")
    index = CorpusIndex(data_dir)
    totals = {k: {"recall": 0.0, "all_gold_found": 0, "any_gold_found": 0} for k in checked_ks}
    missing_gold_articles: set[str] = set()
    for example in examples:
        gold = {doc_id for doc_id, score in example.qrels.items() if score > 0}
        for doc_id in gold:
            if index.read(doc_id) is None:
                missing_gold_articles.add(doc_id)
        hits = index.bm25_search(example.text, k=checked_ks[-1], snippet_chars=160)
        ranked = [hit.doc_id for hit in hits]
        for k in checked_ks:
            found = gold.intersection(ranked[:k])
            recall = candidate_recall(ranked[:k], example.qrels, example.fact_groups)
            totals[k]["recall"] += recall
            totals[k]["all_gold_found"] += int(recall == 1.0)
            totals[k]["any_gold_found"] += int(bool(found))

    count = len(examples)
    return {
        "split": split,
        "queries": count,
        "mean_gold_documents": (
            sum(
                len(example.fact_groups)
                if example.fact_groups
                else sum(score > 0 for score in example.qrels.values())
                for example in examples
            )
            / count
            if count
            else 0.0
        ),
        "missing_gold_articles": sorted(missing_gold_articles),
        "question_bm25": {
            str(k): {
                "mean_recall": values["recall"] / count if count else 0.0,
                "all_gold_rate": values["all_gold_found"] / count if count else 0.0,
                "any_gold_rate": values["any_gold_found"] / count if count else 0.0,
            }
            for k, values in totals.items()
        },
    }
