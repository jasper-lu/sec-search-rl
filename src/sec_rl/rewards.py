from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from sec_rl.config import RewardName

# Fixed reward shaping. This repo favors a few named reward functions over
# tunable knobs; edit these constants (or add a RewardName) to experiment.
# "f4s" is F4 plus a discovery bonus: episodes earn credit for fact groups
# they merely encountered, so search depth is paid for directly instead of
# only through the curated set. The small per-doc cost guards the
# recall-heavy objective against the curate-everything exploit.
REWARD_BETAS: dict[RewardName, float] = {"f1": 1.0, "f2": 2.0, "f4": 4.0, "f4s": 4.0}
DISCOVERY_BONUS: dict[RewardName, float] = {"f4s": 0.2}
CURATED_DOC_COST: dict[RewardName, float] = {"f4s": 0.02}
UNFINISHED_PENALTY = 0.1
INVALID_CURATION_PENALTY = 0.02


class FactGroupLike(Protocol):
    chunk_ids: Sequence[str]
    is_final_answer: bool


@dataclass(frozen=True)
class RetrievalScore:
    precision: float
    recall: float
    f_beta: float
    true_positives: int
    submitted: int
    relevant: int
    final_answer_recall: float = 0.0


def deduplicate(ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def score_submission(
    submitted_ids: Sequence[str],
    qrels: Mapping[str, int],
    *,
    beta: float,
    fact_groups: Sequence[FactGroupLike] = (),
) -> RetrievalScore:
    """Score output using document qrels or Harness-1's fact-level protocol.

    For fact-level labels, precision is the fraction of submitted chunks that
    support any fact. Recall is the fraction of fact groups covered by at least
    one interchangeable supporting chunk, matching Harness-1's evaluator.
    """
    if beta <= 0:
        raise ValueError("beta must be greater than zero")

    submitted = deduplicate(submitted_ids)
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    true_positives = len(set(submitted) & relevant)
    precision = true_positives / len(submitted) if submitted else 0.0
    if fact_groups:
        submitted_set = set(submitted)
        covered = sum(bool(submitted_set.intersection(group.chunk_ids)) for group in fact_groups)
        recall = covered / len(fact_groups)
        final_groups = [group for group in fact_groups if group.is_final_answer]
        final_answer_recall = (
            sum(bool(submitted_set.intersection(group.chunk_ids)) for group in final_groups)
            / len(final_groups)
            if final_groups
            else 0.0
        )
        relevant_count = len(fact_groups)
    else:
        recall = true_positives / len(relevant) if relevant else 0.0
        final_answer_recall = recall
        relevant_count = len(relevant)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    f_beta = (1.0 + beta_squared) * precision * recall / denominator if denominator > 0 else 0.0
    return RetrievalScore(
        precision=precision,
        recall=recall,
        f_beta=f_beta,
        true_positives=true_positives,
        submitted=len(submitted),
        relevant=relevant_count,
        final_answer_recall=final_answer_recall,
    )


def compute_reward(
    score: RetrievalScore,
    *,
    finished: bool,
    invalid_curations: int,
    discovery_bonus: float = 0.0,
) -> float:
    """Terminal reward: F-beta on the curated set, penalized for sloppy episodes.

    The curated set is scored whether or not the episode finished cleanly;
    timing out costs a flat penalty instead of zeroing the reward, so partial
    credit still produces gradient signal. ``discovery_bonus`` adds shaped
    credit (bonus weight x candidate recall) for gold merely encountered.
    """
    value = score.f_beta + discovery_bonus
    if not finished:
        value -= UNFINISHED_PENALTY
    value -= INVALID_CURATION_PENALTY * invalid_curations
    return max(-1.0, value)


def candidate_recall(
    encountered_ids: Sequence[str],
    qrels: Mapping[str, int],
    fact_groups: Sequence[FactGroupLike] = (),
) -> float:
    """Fraction of relevant documents encountered anywhere in the trajectory."""
    if fact_groups:
        encountered = set(encountered_ids)
        return sum(bool(encountered.intersection(group.chunk_ids)) for group in fact_groups) / len(
            fact_groups
        )
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(set(encountered_ids) & relevant) / len(relevant)
