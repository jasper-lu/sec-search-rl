import asyncio
from pathlib import Path

from tinker_cookbook.tool_use import ToolResult
from tinker_cookbook.tool_use.types import ToolInput

from sec_rl.config import HarnessConfig
from sec_rl.data import FactGroup
from sec_rl.environment import RetrievalReward
from sec_rl.rewards import (
    INVALID_CURATION_PENALTY,
    UNFINISHED_PENALTY,
    candidate_recall,
    compute_reward,
    score_submission,
)
from sec_rl.tools import SearchSession, SearchTools

GROUPS = (
    FactGroup("identity", ("a1", "a2")),
    FactGroup("date", ("b1",), is_final_answer=True),
    FactGroup("amount", ("c1", "shared")),
)
QRELS = {chunk_id: 1 for group in GROUPS for chunk_id in group.chunk_ids}


def test_fact_precision_counts_chunks_and_recall_counts_groups() -> None:
    score = score_submission(["a2", "b1", "noise"], QRELS, beta=1.0, fact_groups=GROUPS)
    assert score.precision == 2 / 3
    assert score.recall == 2 / 3
    assert score.f_beta == 2 / 3
    assert score.final_answer_recall == 1.0
    assert score.relevant == 3


def test_alternative_chunks_cover_one_fact() -> None:
    score = score_submission(["a1", "a2"], QRELS, beta=1.0, fact_groups=GROUPS)
    assert score.precision == 1.0
    assert score.recall == 1 / 3


def test_candidate_recall_counts_encountered_fact_groups() -> None:
    assert candidate_recall(["a1", "b1", "noise"], QRELS, GROUPS) == 2 / 3


def test_compute_reward_penalizes_timeouts_but_keeps_partial_credit() -> None:
    score = score_submission(["a1", "b1"], QRELS, beta=1.0, fact_groups=GROUPS)
    finished = compute_reward(score, finished=True, invalid_curations=0)
    timed_out = compute_reward(score, finished=False, invalid_curations=0)
    sloppy = compute_reward(score, finished=True, invalid_curations=2)
    assert finished == score.f_beta
    assert timed_out == score.f_beta - UNFINISHED_PENALTY
    assert sloppy == score.f_beta - 2 * INVALID_CURATION_PENALTY
    assert timed_out > 0  # a timeout with useful curation still earns credit


def _reward_metrics(
    session: SearchSession,
    reward_name: str = "f1",
    *,
    penalize_empty_finish: bool = True,
    penalize_invalid_curations: bool = False,
    penalize_unfinished: bool = False,
) -> tuple[float, dict[str, float]]:
    reward = RetrievalReward(
        qrels=QRELS,
        fact_groups=GROUPS,
        reward=reward_name,
        session=session,
        penalize_empty_finish=penalize_empty_finish,
        penalize_invalid_curations=penalize_invalid_curations,
        penalize_unfinished=penalize_unfinished,
    )
    return asyncio.run(reward([]))


def test_unfinished_penalty_is_opt_in() -> None:
    """By default an unfinished episode scores its curated set like a finished
    one (Harness-1 semantics); the penalty applies only when enabled."""
    unfinished = SearchSession(curated_ids=["a1", "b1"], finished=False)
    finished = SearchSession(curated_ids=["a1", "b1"], finished=True)
    assert _reward_metrics(unfinished)[0] == _reward_metrics(finished)[0]
    assert _reward_metrics(unfinished)[1]["finished"] == 0.0  # still logged
    assert (
        _reward_metrics(unfinished, penalize_unfinished=True)[0]
        == _reward_metrics(finished)[0] - UNFINISHED_PENALTY
    )


def test_invalid_curation_penalty_is_opt_in() -> None:
    """Invalid curate calls cost nothing by default and a fixed amount each when
    the penalty is enabled; the count is logged either way."""
    session = SearchSession(curated_ids=["a1", "b1"], finished=True, invalid_curations=2)
    clean = score_submission(["a1", "b1"], QRELS, beta=1.0, fact_groups=GROUPS).f_beta
    assert _reward_metrics(session)[0] == clean
    assert (
        _reward_metrics(session, penalize_invalid_curations=True)[0]
        == clean - 2 * INVALID_CURATION_PENALTY
    )
    assert _reward_metrics(session)[1]["invalid_curations"] == 2.0  # still logged


def test_empty_curated_set_scores_bad_ending_by_default() -> None:
    """Nothing curated earns the flat floor however the episode ended, so an
    immediate empty finish cannot beat a timeout."""
    from sec_rl.environment import BAD_ENDING_REWARD

    empty_finish = SearchSession(finished=True)
    empty_timeout = SearchSession(encountered_ids={"a1"}, finished=False)
    assert _reward_metrics(empty_finish, "f4")[0] == BAD_ENDING_REWARD
    assert _reward_metrics(empty_timeout, "f4")[0] == BAD_ENDING_REWARD
    assert _reward_metrics(empty_timeout, "f4s")[0] == BAD_ENDING_REWARD
    # With the floor off: empty finish scores 0.0, empty timeout the unfinished penalty.
    assert _reward_metrics(empty_finish, "f4", penalize_empty_finish=False)[0] == 0.0
    assert (
        _reward_metrics(empty_timeout, "f4", penalize_empty_finish=False, penalize_unfinished=True)[
            0
        ]
        == -UNFINISHED_PENALTY
    )
    # Curated episodes are untouched by the flag.
    curated = SearchSession(curated_ids=["a1"], finished=True)
    assert _reward_metrics(curated)[0] == _reward_metrics(curated, penalize_empty_finish=False)[0]


def test_f4s_adds_discovery_bonus_for_encountered_gold() -> None:
    """A rollout that found gold but curated nothing earns shaped credit under
    f4s, so it separates from a group-mate that found nothing. Tested with the
    empty-set floor off, since by default the floor overrides the bonus."""
    no_floor = {"penalize_empty_finish": False, "penalize_unfinished": True}
    session = SearchSession(encountered_ids={"a1", "b1", "noise"}, finished=False)
    plain_value, _ = _reward_metrics(session, "f4", **no_floor)
    shaped_value, metrics = _reward_metrics(session, "f4s", **no_floor)
    assert plain_value == -UNFINISHED_PENALTY
    assert abs(shaped_value - (0.2 * (2 / 3) - UNFINISHED_PENALTY)) < 1e-9
    assert metrics["candidate_recall"] == 2 / 3

    empty = SearchSession()
    assert _reward_metrics(empty, "f4s", **no_floor)[0] == -UNFINISHED_PENALTY

    # The per-doc cost bites: same curated set, same score, minus 0.02/doc.
    curated = SearchSession(encountered_ids={"a1", "b1"}, curated_ids=["a1", "b1"], finished=True)
    plain_curated, _ = _reward_metrics(curated, "f4")
    shaped_curated, _ = _reward_metrics(curated, "f4s")
    assert abs(shaped_curated - (plain_curated + 0.2 * (2 / 3) - 0.02 * 2)) < 1e-9


def test_reward_scores_curated_set_without_finish() -> None:
    session = SearchSession(curated_ids=["a1", "b1"], finished=False)
    value, metrics = _reward_metrics(session, penalize_unfinished=True)
    assert metrics["finished"] == 0.0
    assert metrics["f_beta"] == 0.8
    assert value == 0.8 - UNFINISHED_PENALTY
    assert "f_beta_given_finish" not in metrics


def test_reward_metric_keys_consistent_across_finish_states() -> None:
    """The trainer averages each key over only the trajectories reporting it,
    so keys must match between finished and unfinished episodes (except the
    deliberately finish-conditioned ones)."""
    _, finished = _reward_metrics(SearchSession(curated_ids=["a1"], finished=True))
    _, unfinished = _reward_metrics(SearchSession())
    conditioned_only = {"precision_given_finish", "recall_given_finish", "f_beta_given_finish"}
    assert set(finished) - set(unfinished) == conditioned_only
    assert unfinished["f_beta"] == 0.0
    assert unfinished["used_curate"] == 0.0


def _make_tools(**config_overrides) -> tuple[SearchTools, SearchSession]:
    session = SearchSession()
    session.encountered_ids.update({"a1", "a2", "b1", "c1"})
    config = HarnessConfig(data_dir=Path("."), **config_overrides)
    tools = SearchTools(index=object(), config=config, session=session)  # type: ignore[arg-type]
    return tools, session


def _run_tool(tool_impl, arguments: dict) -> ToolResult:
    return asyncio.run(tool_impl.run(ToolInput(arguments=arguments, call_id="test")))


def test_curate_adds_deduplicates_and_rejects_unencountered() -> None:
    tools, session = _make_tools()
    _run_tool(tools.curate, {"document_ids": ["a1", "a1", "b1"]})
    assert session.curated_ids == ["a1", "b1"]
    assert session.invalid_curations == 0

    _run_tool(tools.curate, {"document_ids": ["a1", "unseen", "c1"]})
    assert session.curated_ids == ["a1", "b1", "c1"]  # valid ids still land
    assert session.invalid_curations == 1


def test_curate_respects_capacity() -> None:
    tools, session = _make_tools(max_curated_docs=2)
    _run_tool(tools.curate, {"document_ids": ["a1", "b1", "c1"]})
    assert session.curated_ids == ["a1", "b1"]
    assert session.invalid_curations == 1


def test_drop_curated_and_finish() -> None:
    tools, session = _make_tools()
    _run_tool(tools.curate, {"document_ids": ["a1", "b1"]})
    _run_tool(tools.drop_curated, {"document_ids": ["a1", "never-curated"]})
    assert session.curated_ids == ["b1"]

    result = _run_tool(tools.finish, {})
    assert result.should_stop
    assert session.finished


def test_overflow_endings_are_graded() -> None:
    from types import SimpleNamespace

    from sec_rl.environment import BAD_ENDING_REWARD, _grade_overflow_endings

    class FakeMessageEnv:
        def __init__(self, curated: float):
            self.curated = curated

        async def _grade(self):
            return 0.42, {"f_beta": 0.52, "finished": 0.0, "curated_documents": self.curated}

        async def step(self, message):
            return SimpleNamespace(reward=0.0, metrics={}, logs={})

    def make_env(curated: float) -> SimpleNamespace:
        async def step(action, *, extra=None):
            overflowed = extra is not None and extra.get("overflow")
            metrics = {"context_overflow": 1.0} if overflowed else {"tool_stopped": 1.0}
            return SimpleNamespace(reward=-0.2 if overflowed else 0.9, metrics=metrics)

        env = SimpleNamespace(step=step, message_env=FakeMessageEnv(curated))
        return _grade_overflow_endings(env)

    env = make_env(curated=2.0)
    overflow = asyncio.run(env.step(None, extra={"overflow": True}))
    assert overflow.reward == 0.42  # graded, not the flat overflow reward
    assert overflow.metrics["f_beta"] == 0.52
    assert overflow.metrics["context_overflow"] == 1.0

    normal = asyncio.run(env.step(None, extra=None))
    assert normal.reward == 0.9  # non-overflow endings untouched

    # The message-level hook records the analysis channel for the rollout logs.
    thinking_msg = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "search first"}],
        "tool_calls": [],
    }
    logged = asyncio.run(env.message_env.step(thinking_msg))
    assert logged.logs["assistant_thinking"] == "search first"

    empty = asyncio.run(make_env(curated=0.0).step(None, extra={"overflow": True}))
    assert empty.reward == BAD_ENDING_REWARD  # empty curated set earns the flat floor


def test_train_raw_instrumentation_records_prefilter_stats() -> None:
    from types import SimpleNamespace

    from sec_rl import train as sec_train

    def group(rewards, curate_flags):
        trajectories = [
            SimpleNamespace(
                transitions=[
                    SimpleNamespace(
                        metrics={
                            "curate_calls": 1.0 if used else 0.0,
                            "finished": 1.0 if used else 0.0,
                            "curated_documents": 1.0 if used else 0.0,
                        },
                        ob=SimpleNamespace(length=100),
                        ac=SimpleNamespace(tokens=[1, 2], maybe_logprobs=[-0.5, -1.0]),
                    )
                ]
            )
            for used in curate_flags
        ]
        return SimpleNamespace(
            trajectories_G=trajectories,
            get_total_rewards=lambda rewards=rewards: rewards,
        )

    groups = [
        group([0.5, 0.0], [True, False]),  # mixed - kept
        group([-0.1, -0.1], [False, False]),  # constant - dropped
    ]
    kept = sec_train._filter_and_record(groups)
    metrics = dict(sec_train._pending_raw_metrics)
    sec_train._pending_raw_metrics.clear()
    assert len(kept) == 1
    assert metrics["train_raw/groups_total"] == 2.0
    assert metrics["train_raw/groups_dropped_constant"] == 1.0
    assert abs(metrics["train_raw/reward"] - 0.075) < 1e-9
    assert metrics["train_raw/frac_used_curate"] == 0.25
    assert abs(metrics["train_raw/entropy"] - 0.75) < 1e-9  # mean -logprob over 4 x 2 tokens
    assert metrics["train_raw/prefill_tokens"] == 400.0
    assert metrics["train_raw/sampled_tokens"] == 8.0
    assert metrics["train_raw/turns_per_episode"] == 1.0
