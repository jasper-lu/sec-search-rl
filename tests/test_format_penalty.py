import asyncio

from sec_rl.environment import RetrievalReward, count_off_format_calls
from sec_rl.tools import SearchSession

CANON = '<|channel|>commentary to=functions.bm25_search <|constrain|>json<|message|>{"query":"x"}<|call|>'
ANALYSIS_CODE = '<|channel|>analysis to=functions.bm25_search code<|message|>{"query":"x"}<|call|>'
NO_CONSTRAIN = (
    '<|channel|>commentary to=functions.curate json<|message|>{"document_ids":["a"]}<|call|>'
)


def test_count_off_format_calls() -> None:
    assert count_off_format_calls(CANON) == 0
    assert count_off_format_calls(ANALYSIS_CODE) == 1
    assert count_off_format_calls(NO_CONSTRAIN) == 1
    assert count_off_format_calls("<|channel|>analysis<|message|>thinking...<|end|>" + CANON) == 0
    assert count_off_format_calls(CANON + ANALYSIS_CODE) == 1
    assert count_off_format_calls("plain text reply") == 0


def _reward(session: SearchSession, penalty: float) -> float:
    fn = RetrievalReward(
        qrels={"a": 1, "b": 1},
        fact_groups=(),
        reward="f4",
        session=session,
        format_penalty=penalty,
    )
    value, metrics = asyncio.run(fn([]))
    assert metrics["off_format_calls"] == float(session.off_format_calls)
    assert metrics["off_format_episode"] == float(session.off_format_calls > 0)
    return value


def test_format_penalty_applies_once_per_episode() -> None:
    clean = SearchSession(curated_ids=["a"], finished=True)
    one = SearchSession(curated_ids=["a"], finished=True, off_format_calls=1)
    many = SearchSession(curated_ids=["a"], finished=True, off_format_calls=7)
    base = _reward(clean, 0.1)
    assert abs(_reward(one, 0.1) - (base - 0.1)) < 1e-9
    assert abs(_reward(many, 0.1) - (base - 0.1)) < 1e-9
    assert _reward(one, 0.0) == base


def test_format_penalty_stacks_on_empty_floor() -> None:
    empty = SearchSession(curated_ids=[], finished=True, off_format_calls=2)
    assert abs(_reward(empty, 0.1) - (-0.2 - 0.1)) < 1e-9
