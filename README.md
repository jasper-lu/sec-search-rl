# sec-search-rl

Reinforcement learning for a small search agent on SEC filings, using
[Tinker](https://thinkingmachines.ai/tinker/) and
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).
This is the code behind the write-up *How easy is it to RL a search model?*

> **Disclaimer:** This repo is heavily AI-coded, human directed.

The task comes from the [Harness-1 paper](https://arxiv.org/abs/2606.02373)
(`pat-jj/harness-1-train-data`): multi-clue questions over SEC filing chunks, each
annotated with 3, 5 or 7 evidence facts. The agent (gpt-oss-20b with a LoRA) searches a
124k–238k chunk corpus with BM25 and grep, reads documents, and builds a curated set of
evidence chunks. Training is synchronous Dr. GRPO via tinker-cookbook. Reward is
recall-weighted F4 of the curated set against the gold facts.

## Results

Held-out F4 on 32 dev queries, mean of two independent eval draws per checkpoint
(single-draw standard error is about 0.03). Each run is 64 queries × 8 rollouts per
update, 24 updates, 12,288 rollouts.

| Recipe | Config | Update 16 | Update 20 | Update 24 | Cost at list prices |
|---|---|---|---|---|---|
| Base model | – | 0.166 | | | |
| F4 | `configs/f4.toml` | 0.202 | 0.284 | 0.316 | $134–384 |
| F4 + discovery bonus − per-doc cost | `configs/f4_w_traj_recall.toml` | 0.246 | 0.295 | 0.305 | $149–412 |
| F4 + format penalty 0.1 | `configs/f4_w_format_penalty.toml` | 0.353 | **0.424** | 0.418 | $174–533 |

The cost range spans none to all of the prompt prefill being served from cache. Each run
took about 8–12 hours of wall-clock time, dominated by sampling.

The format penalty subtracts 0.1 once per episode whenever any tool call was emitted
with a non-canonical Harmony header (for example `analysis` channel or `code` instead of
`json`). The lenient parser executes those calls either way. The base model emits
off-form headers in about a quarter of episodes and plain F4 training locks that in;
the penalty drives it to near zero and the eval score rises with it.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Every dependency is pinned in
`pyproject.toml` and `uv.lock`.

```bash
uv sync
./scripts/setup_env.sh   # prompts for TINKER_API_KEY and the optional keys, writes .env
```

Or copy `.env.example` to `.env` and fill it in by hand.

Load the keys into your shell before running anything that talks to Tinker:

```bash
set -a; . ./.env; set +a
```

## Data

The dataset is a filtered subset of the public Harness-1 SEC data. `prepare-sec` downloads
the query table and corpus shards from Hugging Face once (about 2 GB into `data/raw`),
then builds a leakage-checked query split plus a corpus of every gold chunk, its
same-filing neighbours, and a random background sample, indexed with SQLite FTS5.

Two subsets are used. The small one is the held-out eval set for every number above; the
large one is the training set.

```bash
uv run sec-rl prepare-sec --data-dir data/sec_subset --train-size 256 --dev-size 64
```

```bash
uv run sec-rl prepare-sec --data-dir data/sec_1024 --train-size 1024 --dev-size 64
```

Both use seed 42 and select the same 64 dev queries. The 1,024-query set has a
237,533-chunk corpus; the 256-query set has 124,395. Each build takes a few minutes and
a few GB of disk. Check that gold chunks are reachable by BM25:

```bash
uv run sec-rl retrieval-diagnostics --data-dir data/sec_subset --limit 16
```

## Train

Every recipe is a TOML file in `configs/` whose keys are the `sec-rl train` flag names.
Flags on the command line override the file. Start with the cheap smoke test:

```bash
uv run sec-rl train --config configs/smoke.toml
```

The best recipe, one epoch (16 updates):

```bash
uv run sec-rl train --config configs/f4_w_format_penalty.toml
```

Then continue the same run for half an epoch more on re-shuffled queries, keeping the
optimizer state (this is how all three results above reached update 24):

```bash
uv run sec-rl train --config configs/f4_w_format_penalty.toml --resume --epochs 2 --max-steps 24
```

Add `--wandb-project <name>` to log to Weights & Biases (needs `WANDB_API_KEY`). The run
directory holds `config.json`, `metrics.jsonl`, `checkpoints.jsonl` and per-update
`train_rollout_summaries.jsonl` with every episode's tool calls and sampled text.

Two things the trainer logs that tinker-cookbook does not:

- `train_raw/*` metrics are computed before constant-reward groups are dropped, so the
  logged train reward is not inflated by the filter.
- `train_raw/datums_per_trajectory` should be 1.0. Each multi-turn episode trains as a
  single packed sequence; `src/sec_rl/train.py` patches the gpt-oss renderer so the
  cookbook's prefix check passes on tool calls, which cuts training tokens by about 9×.

## Evaluate

Evaluate a checkpoint on the standard held-out set (paths come from
`runs/<name>/checkpoints.jsonl`):

```bash
uv run sec-rl eval --config configs/eval.toml --checkpoint "tinker://.../sampler_weights/000020" --output evals/fmt_20.jsonl
```

Omit `--checkpoint` for the base model. A single 32-query draw is noisy; run two and
average them before comparing checkpoints. Hosted models can run the same harness through
a chat-completions API:

```bash
uv run sec-rl api-eval --config configs/eval.toml --provider openrouter --model qwen/qwen3.6-35b-a3b --output-dir evals/api/qwen
```

## The harness

Tools: `bm25_search`, `grep_corpus`, `read_document`, `curate` (add chunks, capped at
30), `drop_curated`, `finish`. The system prompt is adapted from Harness-1's retrieval
subagent. The curated set is scored however the episode ends, so a turn-cap timeout still
gets partial credit.

Rewards are named functions in `src/sec_rl/rewards.py`:

- `f4` (and `f1`, `f2`): F-beta over the curated set. Precision counts curated chunks that
  support any fact; recall counts fact groups covered by at least one of their
  interchangeable chunks, matching Harness-1's evaluator.
- `f4s`: F4 plus 0.2 × candidate recall (fact groups the agent encountered in any tool
  result) minus 0.02 per curated document.

Flat terms, all on `HarnessConfig`: −0.2 for an episode that ends with nothing curated
(on by default), `--format-penalty` (off by default), and two off-by-default penalties
kept for ablations, −0.1 for not calling `finish` and −0.02 per invalid `curate` call.
One caveat: the −0.2 empty-set floor is applied last and overrides the format penalty,
so an empty, off-form episode scores −0.2, not −0.3.

## Layout

```
configs/         training and eval recipes (TOML, keys = CLI flags)
src/sec_rl/
  cli.py         sec-rl entry point: prepare-sec, train, eval, api-eval
  sec.py         Harness-1 download and subset builder
  data.py        corpus, query and fact-group loaders
  retrieval.py   SQLite FTS5 BM25 and regex grep over the corpus
  tools.py       the agent's tools and per-episode session state
  rewards.py     named reward functions
  environment.py tinker-cookbook environment, dataset and system prompt
  train.py       Dr. GRPO training run and cookbook patches
  evaluate.py    checkpoint and hosted-model evaluation
tests/           offline unit tests (no API key needed)
```

## Acknowledgements

Data and task from [Harness-1](https://github.com/pat-jj/harness-1). Training on
[Tinker](https://thinkingmachines.ai/tinker/) with tinker-cookbook. MIT licensed; see
`CITATION.cff` to cite.
