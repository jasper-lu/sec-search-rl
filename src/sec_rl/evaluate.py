"""Checkpoint and hosted-model evaluation over the SEC curation harness.

Two entry points share the task, reward, and summary format:

- ``run_evaluation``: rolls out a Tinker base model or checkpoint through the
  exact training environment (``make_environment``).
- ``run_api_evaluation``: drives any chat-completions API (OpenRouter or
  Mixedbread) through a minimal loop that mirrors the training harness —
  same tools, system prompt, turn budget, and turn-countdown warning.

Both support ``tries`` > 1: every query is rolled out N times, summaries
report the pooled single-try means plus best-of-N (per-query max) numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import tinker
from tinker_cookbook import checkpoint_utils, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tool_use.types import ToolInput
from tinker_cookbook.utils.git_rev import recipe_user_metadata

from sec_rl.config import HarnessConfig, RewardName
from sec_rl.data import QueryExample, Split, load_examples
from sec_rl.environment import RetrievalReward, make_environment, system_prompt
from sec_rl.retrieval import get_index
from sec_rl.tools import SearchSession, SearchTools

SUMMARY_METRIC_NAMES = (
    "precision",
    "recall",
    "f_beta",
    "final_answer_recall",
    "candidate_recall",
    "finished",
    "used_curate",
    "produced_output",
    "curated_documents",
    "bm25_calls",
    "grep_calls",
    "read_calls",
    "curate_calls",
    "drop_calls",
    "invalid_curations",
)

BEST_OF_N_METRIC_NAMES = ("f_beta", "recall", "precision")

# Reported only by episodes that finished; averaged over those episodes alone.
CONDITIONED_METRIC_NAMES = (
    "precision_given_finish",
    "recall_given_finish",
    "f_beta_given_finish",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_tries(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Pooled single-try means plus per-query best-of-N means.

    ``rows`` are per-query rows, each with a ``tries`` list of successful
    per-rollout dicts carrying ``metrics``.
    """
    flat = [try_row for row in rows for try_row in row["tries"]]
    summary: dict[str, float] = {}
    for name in SUMMARY_METRIC_NAMES:
        summary[f"mean_{name}"] = _mean(
            [float(try_row["metrics"].get(name, 0.0)) for try_row in flat]
        )
    for name in BEST_OF_N_METRIC_NAMES:
        summary[f"best_of_n_{name}"] = _mean(
            [
                max(float(try_row["metrics"].get(name, 0.0)) for try_row in row["tries"])
                for row in rows
                if row["tries"]
            ]
        )
    for name in CONDITIONED_METRIC_NAMES:
        values = [float(try_row["metrics"][name]) for try_row in flat if name in try_row["metrics"]]
        summary[f"mean_{name}"] = _mean(values)
    return summary


def _best_try_index(tries: list[dict[str, Any]]) -> int | None:
    scored = [
        (float(try_row["metrics"]["f_beta"]), index)
        for index, try_row in enumerate(tries)
        # Parse-error and context-overflow endings skip the reward fn, so
        # their metrics dict exists but has no f_beta.
        if "f_beta" in try_row.get("metrics", {})
    ]
    return max(scored)[1] if scored else None


# --------------------------------------------------------------------------
# Tinker checkpoint evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRun:
    harness: HarnessConfig
    model_name: str
    renderer_name: str
    reward: RewardName = "f1"
    split: Split = "dev"
    checkpoint: str | None = None
    concurrency: int = 16
    tries: int = 1
    limit: int | None = None
    output_path: Path | None = None


async def run_evaluation(run: EvaluationRun) -> dict[str, float]:
    if run.tries < 1:
        raise ValueError("tries must be positive")
    examples = load_examples(run.harness.data_dir, run.split)
    if run.limit is not None:
        if run.limit < 1:
            raise ValueError("evaluation limit must be positive")
        examples = examples[: run.limit]
    service_client = tinker.ServiceClient(user_metadata=recipe_user_metadata("sec_retrieval_eval"))
    if run.checkpoint:
        sampling_client = service_client.create_sampling_client(model_path=run.checkpoint)
        stored_renderer = await checkpoint_utils.get_renderer_name_from_checkpoint_async(
            service_client, run.checkpoint
        )
        renderer_name = stored_renderer or run.renderer_name
    else:
        sampling_client = service_client.create_sampling_client(base_model=run.model_name)
        renderer_name = run.renderer_name

    tokenizer = tokenizer_utils.get_tokenizer(run.model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    policy = TinkerTokenCompleter(sampling_client, max_tokens=run.harness.max_generation_tokens)
    semaphore = asyncio.Semaphore(run.concurrency)

    async def rollout_once(example: QueryExample) -> dict[str, Any]:
        env, session = make_environment(
            example=example,
            renderer=renderer,
            harness_config=run.harness,
            reward=run.reward,
        )
        async with semaphore:
            trajectory = await do_single_rollout(policy, env)
        final_metrics = trajectory.transitions[-1].metrics if trajectory.transitions else {}
        return {
            "curated_ids": list(session.curated_ids),
            "finished": session.finished,
            "metrics": final_metrics,
            "stop_reason": trajectory.stop_reason,
            "turns": len(trajectory.transitions),
            "sampled_tokens": sum(len(step.ac.tokens) for step in trajectory.transitions),
            "aggregate_prompt_tokens": sum(step.ob.length for step in trajectory.transitions),
            "messages": _messages_from_env(env),
        }

    async def evaluate_one(example: QueryExample) -> dict[str, Any]:
        tries = await asyncio.gather(*(rollout_once(example) for _ in range(run.tries)))
        tries = list(tries)
        best_index = _best_try_index(tries)
        # Full transcripts are large; keep only the best try's messages.
        for index, try_row in enumerate(tries):
            if index != best_index:
                try_row.pop("messages", None)
        return {
            "query_id": example.query_id,
            "query": example.text,
            "reward_name": run.reward,
            "best_try_index": best_index,
            "tries": tries,
        }

    rows = await asyncio.gather(*(evaluate_one(example) for example in examples))
    output_path = run.output_path or _default_eval_path(run.split, run.reward)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = _summarize_tries(rows)
    flat = [try_row for row in rows for try_row in row["tries"]]
    for name in ("turns", "sampled_tokens", "aggregate_prompt_tokens"):
        summary[f"mean_{name}"] = _mean([float(try_row[name]) for try_row in flat])
    summary["queries"] = float(len(rows))
    summary["tries_per_query"] = float(run.tries)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                **summary,
                "reward": run.reward,
                "split": run.split,
                "model_name": run.model_name,
                "checkpoint": run.checkpoint,
                "renderer_name": renderer_name,
                "output": str(output_path),
                "harness": asdict(run.harness),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _messages_from_env(env: Any) -> list[dict[str, Any]]:
    if isinstance(env, EnvFromMessageEnv):
        return list(getattr(env.message_env, "history", []))
    return []


def _default_eval_path(split: str, reward: str) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d-%H%M%S")
    return Path("evals") / f"{split}-{reward}-{timestamp}.jsonl"


# --------------------------------------------------------------------------
# Hosted-model (chat-completions API) evaluation
# --------------------------------------------------------------------------

Provider = Literal["openrouter", "mixedbread"]


@dataclass(frozen=True)
class ProviderConfig:
    name: Provider
    endpoint: str
    api_key_envs: tuple[str, ...]
    input_cost_per_million: float
    cached_input_cost_per_million: float
    output_cost_per_million: float


PROVIDERS: dict[Provider, ProviderConfig] = {
    "openrouter": ProviderConfig(
        name="openrouter",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_envs=("OPENROUTER_API_KEY",),
        input_cost_per_million=0.14,
        cached_input_cost_per_million=0.14,
        output_cost_per_million=1.0,
    ),
    "mixedbread": ProviderConfig(
        name="mixedbread",
        endpoint="https://api.mixedbread.com/v1/chat/completions",
        api_key_envs=("MXBAI_API_KEY", "MIXEDBREAD_API_KEY"),
        input_cost_per_million=0.30,
        cached_input_cost_per_million=0.036,
        output_cost_per_million=0.72,
    ),
}


@dataclass(frozen=True)
class ApiEvaluationRun:
    harness: HarnessConfig
    provider: Provider
    model: str
    reward: RewardName = "f1"
    split: Split = "dev"
    concurrency: int = 8
    tries: int = 1
    temperature: float = 1.0
    openrouter_provider: str | None = None
    query_index: int | None = None
    limit: int | None = None
    output_dir: Path = Path("evals/api")


async def run_api_evaluation(run: ApiEvaluationRun) -> dict[str, Any]:
    if run.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if run.tries < 1:
        raise ValueError("tries must be positive")
    provider = PROVIDERS[run.provider]
    api_key = next(
        (os.environ[name] for name in provider.api_key_envs if os.environ.get(name)), None
    )
    if not api_key:
        raise RuntimeError(f"none of {provider.api_key_envs} are set")

    examples = list(enumerate(load_examples(run.harness.data_dir, run.split)))
    if run.query_index is not None:
        if run.query_index < 0 or run.query_index >= len(examples):
            raise ValueError("query_index is outside the selected split")
        examples = [examples[run.query_index]]
    elif run.limit is not None:
        if run.limit < 1:
            raise ValueError("evaluation limit must be positive")
        examples = examples[: run.limit]

    index = get_index(str(run.harness.data_dir.resolve()))
    run.output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = run.output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(run.concurrency)
    started = time.perf_counter()

    limits = httpx.Limits(
        max_connections=max(10, run.concurrency * 2),
        max_keepalive_connections=max(10, run.concurrency),
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=limits) as client:

        async def rollout_once(
            index_number: int, attempt: int, example: QueryExample
        ) -> dict[str, Any]:
            async with semaphore:
                try:
                    result = await _run_api_trajectory(
                        run=run,
                        provider=provider,
                        api_key=api_key,
                        example=example,
                        index=index,
                        client=client,
                    )
                except Exception as exc:  # noqa: BLE001 - record per-try API failures and keep evaluating
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            trace_path = traces_dir / (
                f"{index_number:03d}-{attempt:02d}-{_safe_filename(example.query_id)}.json"
            )
            result["trace_path"] = str(trace_path)
            trace_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            result.pop("messages", None)
            result.pop("calls", None)
            return result

        async def evaluate_one(index_number: int, example: QueryExample) -> dict[str, Any]:
            attempts = await asyncio.gather(
                *(rollout_once(index_number, attempt, example) for attempt in range(run.tries))
            )
            attempts = list(attempts)
            successful = [row for row in attempts if "error" not in row]
            return {
                "query_id": example.query_id,
                "query": example.text,
                "best_try_index": _best_try_index(attempts),
                "tries": successful,
                "failed_tries": [row for row in attempts if "error" in row],
            }

        rows = await asyncio.gather(
            *(evaluate_one(index_number, example) for index_number, example in examples)
        )

    wall_time = time.perf_counter() - started
    results_path = run.output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = _summarize_api(rows, run, wall_time)
    summary["results_path"] = str(results_path)
    summary["traces_dir"] = str(traces_dir)
    summary_path = run.output_dir / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


async def _run_api_trajectory(
    *,
    run: ApiEvaluationRun,
    provider: ProviderConfig,
    api_key: str,
    example: QueryExample,
    index: Any,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    session = SearchSession()
    search_tools = SearchTools(index=index, config=run.harness, session=session)
    tools_by_name = {tool.name: tool for tool in search_tools.implementations()}
    tool_schemas = [{"type": "function", "function": spec} for spec in search_tools.specs()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": example.text},
    ]
    calls: list[dict[str, Any]] = []
    tool_call_count = 0
    started = time.perf_counter()
    stop = False

    for turn in range(1, run.harness.max_turns + 1):
        # Reasoning fields stay in the local transcript but are stripped from
        # requests: echoing them back breaks some harmony serving templates.
        outgoing = [
            {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_details")}
            for m in messages
        ]
        request = {
            "model": run.model,
            "messages": outgoing,
            "tools": tool_schemas,
            "tool_choice": "auto",
            "temperature": run.temperature,
        }
        if run.provider == "openrouter":
            request["max_tokens"] = run.harness.max_generation_tokens
            request["usage"] = {"include": True}
            if run.openrouter_provider:
                request["provider"] = {
                    "only": [run.openrouter_provider],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
        else:
            # Evaluation transcripts are local; do not persist conversations remotely.
            request["max_completion_tokens"] = run.harness.max_generation_tokens
            request["parallel_tool_calls"] = True
            request["store"] = False

        request_started = time.perf_counter()
        response, retry_count = await _post_with_retries(
            client=client,
            endpoint=provider.endpoint,
            api_key=api_key,
            payload=request,
        )
        latency = time.perf_counter() - request_started
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"API response had no choices: {str(payload)[:1000]}")

        choice = choices[0]
        model_message = choice.get("message") or {}
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": model_message.get("content"),
        }
        # Reasoning models (e.g. gpt-oss harmony) return chain-of-thought in
        # separate fields; pass them back so tool-calling turns keep their
        # reasoning context, per the harmony convention.
        for key in ("reasoning", "reasoning_details"):
            if model_message.get(key):
                assistant_message[key] = model_message[key]
        tool_calls = model_message.get("tool_calls") or []
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        calls.append(
            {
                "turn": turn,
                "response_id": payload.get("id"),
                "provider": payload.get("provider") or provider.name,
                "finish_reason": choice.get("finish_reason"),
                "latency_seconds": latency,
                "retry_count": retry_count,
                "usage": payload.get("usage") or {},
            }
        )

        if not tool_calls:
            break

        for tool_call in tool_calls:
            call_id = str(tool_call.get("id") or f"call_{turn}_{tool_call_count}")
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "")
            raw_arguments = function.get("arguments") or "{}"

            if tool_call_count >= run.harness.max_tool_calls:
                messages.append(_tool_error(call_id, tool_name, "maximum_tool_calls_reached"))
                stop = True
                continue

            tool_call_count += 1
            try:
                arguments = (
                    json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                )
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must decode to a JSON object")  # noqa: TRY004
            except (json.JSONDecodeError, ValueError) as exc:
                messages.append(_tool_error(call_id, tool_name, "invalid_arguments", str(exc)))
                continue

            tool = tools_by_name.get(tool_name)
            if tool is None:
                messages.append(_tool_error(call_id, tool_name, "unknown_tool"))
                continue

            result = await tool.run(ToolInput(arguments=arguments, call_id=call_id))
            messages.extend(dict(result_message) for result_message in result.messages)
            if result.should_stop:
                stop = True

        if stop:
            break

    reward_fn = RetrievalReward(
        qrels=example.qrels,
        fact_groups=example.fact_groups,
        reward=run.reward,
        session=session,
    )
    reward_value, metrics = await reward_fn([])
    usage = _aggregate_usage(calls)
    uncached_prompt_tokens = usage["prompt_tokens"] - usage["cached_prompt_tokens"]
    estimated_cost = (
        uncached_prompt_tokens * provider.input_cost_per_million
        + usage["cached_prompt_tokens"] * provider.cached_input_cost_per_million
        + usage["completion_tokens"] * provider.output_cost_per_million
    ) / 1_000_000
    # OpenRouter reports the routed endpoint's actual charge. Prefer it over the
    # fallback estimate, whose rates cannot be correct for every model/provider pair.
    if run.provider == "openrouter" and usage["reported_cost_usd"]:
        estimated_cost = float(usage["reported_cost_usd"])
    return {
        "provider": run.provider,
        "model": run.model,
        "reward_name": run.reward,
        "openrouter_provider": (run.openrouter_provider if run.provider == "openrouter" else None),
        "reward": reward_value,
        "metrics": metrics,
        "curated_ids": list(session.curated_ids),
        "finished": session.finished,
        "encountered_ids": sorted(session.encountered_ids),
        "turns": len(calls),
        "tool_calls": tool_call_count,
        "wall_time_seconds": time.perf_counter() - started,
        "usage": usage,
        "estimated_cost_usd": estimated_cost,
        "calls": calls,
        "messages": messages,
    }


async def _post_with_retries(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    max_attempts: int = 5,
) -> tuple[httpx.Response, int]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(max_attempts):
        response = await client.post(endpoint, headers=headers, json=payload)
        if response.status_code not in {429, 500, 502, 503, 504}:
            if response.is_error:
                raise RuntimeError(
                    f"API returned HTTP {response.status_code}: {response.text[:1000]}"
                )
            return response, attempt
        if attempt == max_attempts - 1:
            raise RuntimeError(
                f"API returned HTTP {response.status_code} after {max_attempts} attempts: "
                f"{response.text[:1000]}"
            )
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else min(2**attempt, 20)
        except ValueError:
            delay = min(2**attempt, 20)
        await asyncio.sleep(max(0.25, min(delay, 60.0)))
    raise AssertionError("unreachable")


def _tool_error(
    call_id: str,
    tool_name: str,
    error: str,
    message: str | None = None,
) -> dict[str, Any]:
    content: dict[str, str] = {"error": error}
    if message:
        content["message"] = message
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": json.dumps(content),
    }


def _aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, float | int]:
    aggregate: dict[str, float | int] = {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reported_cost_usd": 0.0,
    }
    for call in calls:
        usage = call["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                aggregate[key] += int(value)
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            aggregate["reported_cost_usd"] += float(cost)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens")
        if isinstance(cached_tokens, (int, float)):
            aggregate["cached_prompt_tokens"] += int(cached_tokens)
    return aggregate


def _summarize_api(
    rows: list[dict[str, Any]],
    run: ApiEvaluationRun,
    wall_time: float,
) -> dict[str, Any]:
    flat = [try_row for row in rows for try_row in row["tries"]]
    failed = [failure for row in rows for failure in row["failed_tries"]]
    prompt_tokens = sum(int(try_row["usage"]["prompt_tokens"]) for try_row in flat)
    cached_prompt_tokens = sum(
        int(try_row["usage"].get("cached_prompt_tokens", 0)) for try_row in flat
    )
    completion_tokens = sum(int(try_row["usage"]["completion_tokens"]) for try_row in flat)
    reported_cost = sum(float(try_row["usage"]["reported_cost_usd"]) for try_row in flat)
    estimated_cost = sum(float(try_row["estimated_cost_usd"]) for try_row in flat)
    return {
        "provider": run.provider,
        "model": run.model,
        "split": run.split,
        "reward": run.reward,
        "openrouter_provider": (run.openrouter_provider if run.provider == "openrouter" else None),
        "queries": len(rows),
        "tries_per_query": run.tries,
        "completed_trajectories": len(flat),
        "failed_trajectories": len(failed),
        "mean_reward": _mean([float(try_row["reward"]) for try_row in flat]),
        **_summarize_tries(rows),
        "mean_turns": _mean([float(try_row["turns"]) for try_row in flat]),
        "mean_tool_calls": _mean([float(try_row["tool_calls"]) for try_row in flat]),
        "total_prompt_tokens": prompt_tokens,
        "total_cached_prompt_tokens": cached_prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "reported_cost_usd": reported_cost,
        "estimated_cost_usd": estimated_cost,
        "wall_time_seconds": wall_time,
        "concurrency": run.concurrency,
        "harness": asdict(run.harness),
    }


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
