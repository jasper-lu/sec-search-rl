from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import gpt_oss as gpt_oss_renderer
from tinker_cookbook.rl import data_processing
from tinker_cookbook.rl import train as rl_train

from sec_rl.config import HarnessConfig, RewardName, TrainingConfig
from sec_rl.environment import SecDatasetBuilder

# ---------------------------------------------------------------------------
# Two tinker-cookbook monkeypatches.
#
# 1. Tool-call rendering must match what gpt-oss samples. The cookbook packs a
#    multi-turn episode into ONE training sequence only if each turn's prompt
#    is a token-prefix extension of the previous prompt + sampled action
#    (rl/data_processing.trajectory_to_data). gpt-oss samples a tool call as
#    `<|channel|>commentary to=functions.X <|constrain|>json<|message|>` (the
#    Harmony order), while the cookbook renderer writes history as
#    `to=functions.X<|channel|>commentary`. With the stock renderer the prefix
#    check fails at the first tool call, every turn becomes its own datum
#    carrying the full context, and training tokens grow from the final
#    context to the sum of all per-turn contexts (roughly the number of turns
#    times more). Gradients are unaffected; only cost.
#
# 2. Pre-filter metrics. The cookbook computes every env/all/* train metric
#    AFTER dropping constant-reward groups, which averages away the failures.
#    The pre-filter groups exist in exactly one place, the argument to
#    remove_constant_reward_groups, so we wrap it there and attach the results
#    to the step's metrics under train_raw/*. The KL hook sees the assembled
#    datums, so datum packing and training-token counts (the check for patch 1)
#    are logged from there.
# ---------------------------------------------------------------------------


def _render_tool_calls_as_sampled(self, tool_calls) -> str:
    parts = []
    for i, tc in enumerate(tool_calls):
        parts.append(
            f"<|channel|>commentary to=functions.{tc.function.name} <|constrain|>json<|message|>"
            f"{tc.function.arguments}<|call|>"
        )
        if i < len(tool_calls) - 1:
            parts.append("<|start|>assistant")
    return "".join(parts)


gpt_oss_renderer.GptOssRenderer._render_tool_calls = _render_tool_calls_as_sampled


def _parse_response_without_raw_echo(self, response):
    # When a tool call comes with no analysis/text part, the cookbook parser
    # sets content to the raw response string; re-rendering then emits that
    # string as a commentary preamble ahead of the tool call, so the history
    # shows the call twice and the prefix check fails. Nothing is lost by
    # dropping it: the tool call itself is rendered from tool_calls.
    message, termination = _original_parse_response(self, response)
    if message.get("tool_calls") and isinstance(message.get("content"), str):
        message["content"] = []
    return message, termination


_original_parse_response = gpt_oss_renderer.GptOssRenderer.parse_response
gpt_oss_renderer.GptOssRenderer.parse_response = _parse_response_without_raw_echo


# Packing diagnostic: log where a turn's prompt stops being a prefix of the
# previous prompt + sampled action, with decoded context, for the first few
# failures. Read with: grep PREFIX_BREAK <log_path>/logs.log
_prefix_break_logs = 0
_PREFIX_BREAK_LOG_LIMIT = 60
_diag_tokenizer = None


def _is_prefix_logged(seq1, seq2) -> bool:
    global _prefix_break_logs, _diag_tokenizer
    ok = _original_is_prefix(seq1, seq2)
    if ok or _prefix_break_logs >= _PREFIX_BREAK_LOG_LIMIT:
        return ok
    _prefix_break_logs += 1
    try:
        if _diag_tokenizer is None:
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            _diag_tokenizer = get_tokenizer("openai/gpt-oss-20b")
        i = next(
            (k for k, (a, b) in enumerate(zip(seq1, seq2)) if a != b), min(len(seq1), len(seq2))
        )
        ints = lambda seq: [t for t in seq if isinstance(t, int)]
        before = _diag_tokenizer.decode(ints(seq1[max(0, i - 24) : i]))
        old = _diag_tokenizer.decode(ints(seq1[i : i + 16]))
        new = _diag_tokenizer.decode(ints(seq2[i : i + 16]))
        logging.getLogger(__name__).info(
            "PREFIX_BREAK at token %d/%d (new prompt %d): ...%r | sampled: %r | re-rendered: %r",
            i,
            len(seq1),
            len(seq2),
            before,
            old,
            new,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break training
        logging.getLogger(__name__).info("PREFIX_BREAK (undecodable): %s", exc)
    return ok


_original_is_prefix = data_processing._is_prefix
data_processing._is_prefix = _is_prefix_logged


_pending_raw_metrics: dict[str, float] = {}
_kept_trajectories = 0


def _final_metrics(trajectory) -> dict:
    return trajectory.transitions[-1].metrics if trajectory.transitions else {}


def _filter_and_record(trajectory_groups):
    global _kept_trajectories
    kept = _original_filter(trajectory_groups)
    _kept_trajectories = sum(len(group.trajectories_G) for group in kept)
    rewards = [reward for group in trajectory_groups for reward in group.get_total_rewards()]
    trajectories = [
        trajectory for group in trajectory_groups for trajectory in group.trajectories_G
    ]
    finals = [_final_metrics(trajectory) for trajectory in trajectories]
    transitions = [
        transition for trajectory in trajectories for transition in trajectory.transitions
    ]
    # Mean negative log-prob of the sampled action tokens is a Monte-Carlo
    # estimate of the policy's per-token entropy (the cookbook's optim/entropy
    # is the same quantity after constant-group filtering).
    sampled_logprobs = [
        logprob for transition in transitions for logprob in (transition.ac.maybe_logprobs or [])
    ]
    prefill_tokens = sum(transition.ob.length for transition in transitions)
    sampled_tokens = sum(len(transition.ac.tokens) for transition in transitions)

    def frac(predicate) -> float:
        return sum(1.0 for m in finals if predicate(m)) / len(finals) if finals else 0.0

    def mean_where_present(name: str) -> float:
        values = [m[name] for m in finals if name in m]
        return sum(values) / len(values) if values else 0.0

    _pending_raw_metrics.update(
        {
            "train_raw/reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "train_raw/precision_given_finish": mean_where_present("precision_given_finish"),
            "train_raw/recall_given_finish": mean_where_present("recall_given_finish"),
            "train_raw/groups_total": float(len(trajectory_groups)),
            "train_raw/groups_dropped_constant": float(len(trajectory_groups) - len(kept)),
            "train_raw/frac_used_curate": frac(lambda m: m.get("curate_calls", 0) > 0),
            "train_raw/frac_finished": frac(lambda m: m.get("finished", 0) > 0),
            "train_raw/frac_produced_output": frac(
                lambda m: m.get("finished", 0) > 0 or m.get("curated_documents", 0) > 0
            ),
            "train_raw/entropy": (
                -sum(sampled_logprobs) / len(sampled_logprobs) if sampled_logprobs else 0.0
            ),
            "train_raw/turns_per_episode": len(transitions) / len(trajectories)
            if trajectories
            else 0.0,
            "train_raw/prefill_tokens": float(prefill_tokens),
            "train_raw/sampled_tokens": float(sampled_tokens),
            "train_raw/prefill_tokens_per_episode": (
                prefill_tokens / len(trajectories) if trajectories else 0.0
            ),
        }
    )
    return kept


def _metrics_with_raw(trajectory_groups, taglist):
    out = _original_compute_metrics(trajectory_groups, taglist)
    if _pending_raw_metrics:
        out.update(_pending_raw_metrics)
        _pending_raw_metrics.clear()
    return out


def _kl_with_datum_stats(data_D, training_logprobs_D):
    out = _original_kl(data_D, training_logprobs_D)
    train_tokens = sum(datum.model_input.length for datum in data_D)
    out["train_raw/datums"] = float(len(data_D))
    out["train_raw/train_tokens"] = float(train_tokens)
    if _kept_trajectories:
        # 1.0 means every episode packed into one training sequence.
        out["train_raw/datums_per_trajectory"] = len(data_D) / _kept_trajectories
        out["train_raw/train_tokens_per_trajectory"] = train_tokens / _kept_trajectories
    return out


_original_filter = rl_train.remove_constant_reward_groups
_original_compute_metrics = rl_train.compute_trajectory_metrics
_original_kl = rl_train.compute_kl_sample_train
rl_train.remove_constant_reward_groups = _filter_and_record
rl_train.compute_trajectory_metrics = _metrics_with_raw
rl_train.compute_kl_sample_train = _kl_with_datum_stats


@dataclass(frozen=True)
class TrainingRun:
    harness: HarnessConfig
    training: TrainingConfig
    reward: RewardName
    log_path: Path
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    # In-place resume: tinker-cookbook reloads weights + optimizer from the last
    # entry of <log_path>/checkpoints.jsonl and continues at that batch index,
    # so the data order and Adam state are both continuous. The default "ask"
    # prompt raises EOFError under nohup, so this must be decided up front.
    resume: bool = False


async def run_training(run: TrainingRun) -> None:
    """Run synchronous fresh-rollout Dr. GRPO-style training."""
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    cli_utils.check_log_dir(str(run.log_path), behavior_if_exists="resume" if run.resume else "ask")

    dataset_builder = SecDatasetBuilder(
        data_dir=str(run.harness.data_dir.resolve()),
        model_name_for_tokenizer=run.training.model_name,
        renderer_name=run.training.renderer_name,
        reward=run.reward,
        groups_per_batch=run.training.groups_per_batch,
        group_size=run.training.group_size,
        eval_size=run.training.eval_size,
        eval_group_size=run.training.eval_group_size,
        max_turns=run.harness.max_turns,
        max_tool_calls=run.harness.max_tool_calls,
        max_trajectory_tokens=run.harness.max_trajectory_tokens,
        max_generation_tokens=run.harness.max_generation_tokens,
        search_default_k=run.harness.search_default_k,
        search_max_k=run.harness.search_max_k,
        result_snippet_chars=run.harness.result_snippet_chars,
        read_max_chars=run.harness.read_max_chars,
        max_curated_docs=run.harness.max_curated_docs,
        seed=run.training.seed,
        start_batch=run.training.train_start_batch,
        epochs=run.training.train_epochs,
        penalize_empty_finish=run.harness.penalize_empty_finish,
        penalize_invalid_curations=run.harness.penalize_invalid_curations,
        penalize_unfinished=run.harness.penalize_unfinished,
        format_penalty=run.harness.format_penalty,
    )
    config = rl_train.Config(
        model_name=run.training.model_name,
        recipe_name="sec_retrieval_dr_grpo",
        renderer_name=run.training.renderer_name,
        log_path=str(run.log_path),
        dataset_builder=dataset_builder,
        learning_rate=run.training.learning_rate,
        max_tokens=run.harness.max_generation_tokens,
        lora_rank=run.training.lora_rank,
        loss_fn="importance_sampling",
        num_substeps=1,
        temperature=1.0,
        remove_constant_reward_groups=True,
        kl_penalty_coef=0.0,
        eval_every=run.training.eval_every,
        save_every=run.training.save_every,
        max_steps=run.training.max_steps,
        wandb_project=run.wandb_project,
        wandb_name=run.wandb_name,
        load_checkpoint_path=run.load_checkpoint_path,
    )
    await rl_train.main(config)


def default_log_path(reward: str) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d-%H%M%S")
    return Path("runs") / f"sec-{reward}-{timestamp}"
