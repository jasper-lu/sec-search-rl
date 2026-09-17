from pathlib import Path

import pytest

from sec_rl.cli import parse_args

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TRAIN_CONFIGS = ("f4.toml", "f4_w_traj_recall.toml", "f4_w_format_penalty.toml", "smoke.toml")


@pytest.mark.parametrize("name", TRAIN_CONFIGS)
def test_train_configs_parse(name: str) -> None:
    args = parse_args(["train", "--config", str(CONFIGS / name)])
    assert args.model == "openai/gpt-oss-20b"
    assert isinstance(args.data_dir, Path)
    assert args.log_path == Path("runs") / name.removesuffix(".toml")


def test_eval_config_parses() -> None:
    args = parse_args(["eval", "--config", str(CONFIGS / "eval.toml")])
    assert args.data_dir == Path("data/sec_subset")
    assert (args.limit, args.tries, args.reward) == (32, 1, "f4")


def _train_settings(name: str) -> dict[str, object]:
    args = vars(parse_args(["train", "--config", str(CONFIGS / name)]))
    return {k: v for k, v in args.items() if k not in ("config", "log_path", "handler")}


def test_train_recipes_differ_only_in_reward_terms() -> None:
    plain, shaped, fmt = (_train_settings(n) for n in TRAIN_CONFIGS[:3])
    assert {k for k in plain if plain[k] != shaped[k]} == {"reward"}
    assert {k for k in plain if plain[k] != fmt[k]} == {"format_penalty"}
    assert (shaped["reward"], fmt["format_penalty"]) == ("f4s", 0.1)


def test_command_line_overrides_config() -> None:
    args = parse_args(["train", "--config", str(CONFIGS / "f4.toml"), "--max-steps", "3"])
    assert args.max_steps == 3
    assert args.groups_per_batch == 64


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("learning_rate = 1e-4\nbatch_size = 8\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="batch_size"):
        parse_args(["train", "--config", str(bad)])
