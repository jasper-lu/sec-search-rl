from __future__ import annotations

import argparse
import asyncio
import json
import tomllib
from pathlib import Path
from typing import Any

from sec_rl.config import DEFAULT_DATA_DIR, DEFAULT_RAW_DIR, HarnessConfig, TrainingConfig

DEFAULT_MODEL = "openai/gpt-oss-20b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sec-rl",
        description="RL a search agent on the Harness-1 SEC filing retrieval task.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_sec = subparsers.add_parser(
        "prepare-sec",
        help="Build a leakage-safe Harness-1 SEC subset with hard and random distractors.",
    )
    prepare_sec.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prepare_sec.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Where the Hugging Face shards are downloaded; shared by every prepared subset.",
    )
    prepare_sec.add_argument("--train-size", type=int, default=256)
    prepare_sec.add_argument("--dev-size", type=int, default=64)
    prepare_sec.add_argument("--random-distractors", type=int, default=60_000)
    prepare_sec.add_argument("--neighbor-radius", type=int, default=4)
    prepare_sec.add_argument("--seed", type=int, default=42)
    prepare_sec.add_argument("--force-download", action="store_true")
    prepare_sec.add_argument("--force-prepare", action="store_true")
    prepare_sec.set_defaults(handler=_prepare_sec)

    diagnostics = subparsers.add_parser(
        "retrieval-diagnostics",
        help="Measure question-only BM25 coverage and verify gold chunks are searchable.",
    )
    diagnostics.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    diagnostics.add_argument("--split", choices=("dev", "train"), default="dev")
    diagnostics.add_argument("--limit", type=int)
    diagnostics.add_argument("--k", type=int, nargs="+", default=[5, 10, 25])
    diagnostics.set_defaults(handler=_retrieval_diagnostics)

    train = subparsers.add_parser("train", help="Train with Dr. GRPO-style grouped advantages.")
    _add_config_argument(train)
    _add_harness_arguments(train)
    _add_model_arguments(train)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--lora-rank", type=int, default=32)
    train.add_argument("--group-size", type=int, default=8, help="Rollouts per query.")
    train.add_argument("--groups-per-batch", type=int, default=64, help="Queries per update.")
    train.add_argument("--max-steps", type=int, default=16, help="Number of policy updates.")
    train.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run the trainer's built-in held-out eval every N updates; 0 disables it.",
    )
    train.add_argument(
        "--train-eval-size",
        type=int,
        default=32,
        help="Number of held-out queries used by the trainer's periodic eval.",
    )
    train.add_argument(
        "--eval-group-size",
        type=int,
        default=4,
        help="Rollouts per held-out query in the periodic eval; >1 reduces eval variance.",
    )
    train.add_argument("--save-every", type=int, default=1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-path", type=Path)
    train.add_argument("--wandb-project")
    train.add_argument("--wandb-name")
    train.add_argument("--load-checkpoint")
    train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a run in place: reload weights and optimizer from the last "
            "checkpoint in --log-path and pick up at the next batch. Pass the "
            "same training flags as the original launch."
        ),
    )
    train.add_argument(
        "--train-start-batch",
        type=int,
        default=0,
        help=(
            "Skip this many groups-per-batch-sized batches of the train order before "
            "step 0, so a warm start from a checkpoint continues on unseen queries."
        ),
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=1,
        help=(
            "Passes over the train queries (default 1). Epochs after the first use a "
            "seeded re-shuffle. Use with --max-steps beyond one epoch, e.g. "
            "--resume --epochs 2 --max-steps 24 continues a finished 16-step epoch."
        ),
    )
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser(
        "eval", help="Evaluate a base model or Tinker checkpoint with one rollout per query."
    )
    _add_config_argument(evaluate)
    _add_harness_arguments(evaluate)
    _add_model_arguments(evaluate)
    evaluate.add_argument(
        "--checkpoint", help="tinker://.../sampler_weights/N; omit for the base model."
    )
    evaluate.add_argument(
        "--tries",
        type=int,
        default=1,
        help="Rollouts per query; summaries report pooled means and best-of-N.",
    )
    evaluate.add_argument("--split", choices=("dev", "train"), default="dev")
    evaluate.add_argument("--concurrency", type=int, default=16)
    evaluate.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N queries; useful for a one-query preview.",
    )
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_eval)

    api_eval = subparsers.add_parser(
        "api-eval",
        help="Evaluate a chat-completions model and save browseable JSON traces.",
    )
    _add_config_argument(api_eval)
    _add_harness_arguments(api_eval)
    api_eval.add_argument("--provider", choices=("openrouter", "mixedbread"), required=True)
    api_eval.add_argument("--model", required=True)
    api_eval.add_argument("--split", choices=("dev", "train"), default="dev")
    api_eval.add_argument("--concurrency", type=int, default=8)
    api_eval.add_argument(
        "--tries",
        type=int,
        default=1,
        help="Rollouts per query; summaries report pooled means and best-of-N.",
    )
    api_eval.add_argument("--temperature", type=float, default=1.0)
    api_eval.add_argument(
        "--openrouter-provider",
        default="",
        help="Exact OpenRouter endpoint tag; pass an empty string to use automatic routing.",
    )
    api_eval.add_argument("--limit", type=int)
    api_eval.add_argument(
        "--query-index",
        type=int,
        help="Evaluate one zero-based query index from the selected split; useful for retries.",
    )
    api_eval.add_argument("--output-dir", type=Path, required=True)
    api_eval.set_defaults(handler=_api_eval)
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "TOML file of defaults for this command, keyed by flag name (see configs/). "
            "Flags given on the command line override it."
        ),
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--renderer",
        help="Renderer override. By default, prefer the model's non-thinking renderer.",
    )


def _add_harness_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reward",
        choices=("f1", "f2", "f4", "f4s"),
        default="f4",
        help="Named reward function over the curated set: balanced F1, recall-weighted "
        "F2/F4, or f4s (F4 plus a candidate-recall discovery bonus minus a per-document cost).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--search-max-k", type=int, default=25)
    parser.add_argument("--snippet-chars", type=int, default=220)
    parser.add_argument("--read-max-chars", type=int, default=4_000)
    parser.add_argument("--max-curated-docs", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=128)
    parser.add_argument("--max-trajectory-tokens", type=int, default=30_720)
    parser.add_argument("--max-generation-tokens", type=int, default=2_048)
    parser.add_argument(
        "--penalize-empty-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Score any episode that ends with an empty curated set at the flat bad-ending "
            "reward (-0.2) instead of F-beta 0 minus penalties."
        ),
    )
    parser.add_argument(
        "--penalize-invalid-curations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Charge -0.02 per curate call that names an unseen document or exceeds the cap.",
    )
    parser.add_argument(
        "--penalize-unfinished",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Charge -0.1 when the episode ends without a finish call. Off by default: "
            "finish, a plain-text reply and the turn cap all score the curated set the "
            "same way, as in Harness-1."
        ),
    )
    parser.add_argument(
        "--format-penalty",
        type=float,
        default=0.0,
        help=(
            "Subtract this once per episode when any tool call used an off-form Harmony "
            "header (wrong channel, missing <|constrain|>, `code` instead of `json`). "
            "0 = off."
        ),
    )


def load_config_file(path: Path) -> dict[str, Any]:
    """Read a flat TOML file of flag values; keys may use hyphens or underscores."""
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    return {key.replace("-", "_"): value for key, value in values.items()}


def apply_config_file(subparser: argparse.ArgumentParser, path: Path) -> None:
    """Install a config file's values as the subparser's defaults."""
    actions = {
        action.dest: action
        for action in subparser._actions
        if action.dest not in ("help", "config", "handler")
    }
    values = load_config_file(path)
    unknown = sorted(set(values) - set(actions))
    if unknown:
        raise SystemExit(
            f"{path}: unknown keys {unknown}; valid keys are "
            + ", ".join(sorted(key.replace("_", "-") for key in actions))
        )
    converted = {}
    for key, value in values.items():
        action = actions[key]
        if action.type is not None and isinstance(value, str):
            value = action.type(value)
        if action.choices is not None and value not in action.choices:
            raise SystemExit(f"{path}: {key} must be one of {sorted(action.choices)}")
        converted[key] = value
    subparser.set_defaults(**converted)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    config_path = getattr(preliminary, "config", None)
    if config_path is not None:
        subparser = _subparser(parser, preliminary.command)
        apply_config_file(subparser, config_path)
    return parser.parse_args(argv)


def _subparser(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[command]
    raise KeyError(command)


def _harness(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        data_dir=args.data_dir,
        search_default_k=args.search_k,
        search_max_k=args.search_max_k,
        result_snippet_chars=args.snippet_chars,
        read_max_chars=args.read_max_chars,
        max_curated_docs=args.max_curated_docs,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_generation_tokens=args.max_generation_tokens,
        penalize_empty_finish=args.penalize_empty_finish,
        penalize_invalid_curations=args.penalize_invalid_curations,
        penalize_unfinished=args.penalize_unfinished,
        format_penalty=args.format_penalty,
    )


def _renderer(model_name: str, override: str | None) -> str:
    if override is not None:
        return override
    from tinker_cookbook import model_info

    renderer_names = model_info.get_recommended_renderer_names(model_name)
    return next(
        (name for name in renderer_names if name.endswith("_disable_thinking")),
        renderer_names[0],
    )


def _prepare_sec(args: argparse.Namespace) -> None:
    from sec_rl.sec import download_sec, prepare_sec

    download_sec(args.raw_dir, force=args.force_download)
    summary = prepare_sec(
        args.data_dir,
        raw_dir=args.raw_dir,
        train_size=args.train_size,
        dev_size=args.dev_size,
        random_distractors=args.random_distractors,
        neighbor_radius=args.neighbor_radius,
        seed=args.seed,
        force=args.force_prepare or args.force_download,
    )
    print(json.dumps(summary, indent=2))


def _retrieval_diagnostics(args: argparse.Namespace) -> None:
    from sec_rl.diagnostics import retrieval_diagnostics

    summary = retrieval_diagnostics(
        data_dir=args.data_dir,
        split=args.split,
        ks=args.k,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


def _train(args: argparse.Namespace) -> None:
    from sec_rl.train import TrainingRun, default_log_path, run_training

    training = TrainingConfig(
        model_name=args.model,
        renderer_name=_renderer(args.model, args.renderer),
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_size=args.train_eval_size,
        eval_group_size=args.eval_group_size,
        save_every=args.save_every,
        seed=args.seed,
        train_start_batch=args.train_start_batch,
        train_epochs=args.epochs,
    )
    run = TrainingRun(
        harness=_harness(args),
        training=training,
        reward=args.reward,
        log_path=args.log_path or default_log_path(args.reward),
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        load_checkpoint_path=args.load_checkpoint,
        resume=args.resume,
    )
    asyncio.run(run_training(run))


def _eval(args: argparse.Namespace) -> None:
    from sec_rl.evaluate import EvaluationRun, run_evaluation

    run = EvaluationRun(
        harness=_harness(args),
        model_name=args.model,
        renderer_name=_renderer(args.model, args.renderer),
        reward=args.reward,
        split=args.split,
        checkpoint=args.checkpoint,
        concurrency=args.concurrency,
        tries=args.tries,
        limit=args.limit,
        output_path=args.output,
    )
    summary = asyncio.run(run_evaluation(run))
    print(json.dumps(summary, indent=2))


def _api_eval(args: argparse.Namespace) -> None:
    from sec_rl.evaluate import ApiEvaluationRun, run_api_evaluation

    run = ApiEvaluationRun(
        harness=_harness(args),
        provider=args.provider,
        model=args.model,
        reward=args.reward,
        split=args.split,
        concurrency=args.concurrency,
        tries=args.tries,
        temperature=args.temperature,
        openrouter_provider=args.openrouter_provider or None,
        query_index=args.query_index,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    summary = asyncio.run(run_api_evaluation(run))
    print(json.dumps(summary, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    handler: Any = args.handler
    handler(args)


if __name__ == "__main__":
    main()
