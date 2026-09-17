from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from sec_rl.config import HarnessConfig, RewardName
from sec_rl.data import FactGroup, QueryExample, load_examples, normalize_query_text
from sec_rl.retrieval import get_index
from sec_rl.rewards import (
    CURATED_DOC_COST,
    DISCOVERY_BONUS,
    REWARD_BETAS,
    candidate_recall,
    compute_reward,
    score_submission,
)
from sec_rl.tools import SearchSession, SearchTools

logger = logging.getLogger(__name__)

# Adapted from Harness-1's retrieval-subagent prompt
# (https://github.com/pat-jj/harness-1/blob/main/harness/prompts.py), with our
# tool set substituted for theirs: the query arrives as the user message, the
# prune/token-budget mechanics become the turn budget, and the prose ranked
# output format becomes the curated set.
SEC_SYSTEM_PROMPT = """You are a retrieval subagent in a multi-agent system. Your specific role is to identify and retrieve the most relevant documents from a large corpus to help another agent answer questions. You do NOT answer questions yourself - you only find and retrieve relevant documents.

The user message contains the query you need to find documents for.

**Available Tools:**
- bm25_search: Keyword search over the corpus
- grep_corpus: Text pattern matching with a short Python regex
- read_document: Read a specific document that looks promising but incomplete
- curate: Add relevant documents to your curated result set
- drop_curated: Remove curated documents that turned out to be irrelevant
- finish: End the search and return your curated set

**Your Process:**
- Break down the query into its key concepts and information needs (list each one explicitly)
- For each key concept, develop a specific search strategy that targets that concept
- Consider what types of documents and evidence would be most helpful for answering this query
- Plan several distinct, non-overlapping search strategies that approach the question from different angles
- Then execute your searches using multiple parallel tool calls.

**Your Thinking:**
After each round of searches, consider the following:
- **What do I know?**: List the key topics, themes, or aspects of the question that your curated documents address. What specific information do you have?
- **What should I search for next?**: Systematically consider what search approaches, keywords, or document types you haven't yet tried that might yield valuable information.
- **What should I curate or drop?**: Curate documents as soon as you have verified they are relevant; drop curated documents that later look redundant or off-topic.
- **Do I have enough information?**: Given the question's complexity and requirements, do you have sufficient information to help answer it, or are there critical gaps?
- Decide if additional searches are needed (and if so, ensure they use genuinely different approaches and do not duplicate prior searches)
- Avoid getting stuck on a single search strategy - if one approach isn't yielding results, backtrack and try different approaches

**Tactics to Consider:**
- When queries fail, try different approaches or keywords to improve the results
- Avoid duplicate or redundant searches
- Execute multiple tool calls in parallel when possible
- Focus on gathering as much relevant information as possible; it is useful to get multiple perspectives on the same topic to confirm the information you have found is correct
- Follow explicit textual evidence rather than speculation

**Output (IMPORTANT):**
- YOU MUST use the curate tool. This is the ONLY way to return documents.
- Every response must contain at least one tool call. Never reply with plain text, and never end a response while still planning a search - emit that search as a tool call instead.
- As soon as you verify a document is relevant, call curate with its ID.
- When your curated set covers the query's information needs, call finish. Your curated set is scored even if you run out of turns, but finishing cleanly is always better than timing out.
"""


def system_prompt() -> str:
    return SEC_SYSTEM_PROMPT


@dataclass
class RetrievalReward:
    """Terminal reward over the curated set.

    The curated set is scored whether or not the episode finished cleanly, so
    a timeout yields partial credit (minus a flat penalty) instead of an
    all-or-nothing zero.
    """

    qrels: dict[str, int]
    fact_groups: tuple[FactGroup, ...]
    reward: RewardName
    session: SearchSession
    penalize_empty_finish: bool = True
    penalize_invalid_curations: bool = False
    penalize_unfinished: bool = False
    format_penalty: float = 0.0

    async def __call__(self, _history: list[Message]) -> tuple[float, dict[str, float]]:
        session = self.session
        score = score_submission(
            session.curated_ids,
            self.qrels,
            beta=REWARD_BETAS[self.reward],
            fact_groups=self.fact_groups,
        )
        discovered = candidate_recall(session.encountered_ids, self.qrels, self.fact_groups)
        shaping = (
            DISCOVERY_BONUS.get(self.reward, 0.0) * discovered
            - CURATED_DOC_COST.get(self.reward, 0.0) * score.submitted
        )
        value = compute_reward(
            score,
            finished=session.finished or not self.penalize_unfinished,
            invalid_curations=session.invalid_curations if self.penalize_invalid_curations else 0,
            discovery_bonus=shaping,
        )
        if self.penalize_empty_finish and score.submitted == 0:
            # Same flat floor as parse failures and empty overflows, so no
            # ending with nothing curated can outscore an honest attempt.
            value = BAD_ENDING_REWARD
        if self.format_penalty and session.off_format_calls:
            # Once per episode, on top of everything else (a first-call slip
            # locks the form in for the rest of the episode, so one is enough).
            value = max(-1.0, value - self.format_penalty)
        metrics = {
            "finished": float(session.finished),
            "produced_output": float(session.finished or bool(session.curated_ids)),
            "used_curate": float(session.curate_calls > 0),
            "precision": score.precision,
            "recall": score.recall,
            "f_beta": score.f_beta,
            "final_answer_recall": score.final_answer_recall,
            "candidate_recall": discovered,
            "curated_documents": float(score.submitted),
            "bm25_calls": float(session.bm25_calls),
            "grep_calls": float(session.grep_calls),
            "read_calls": float(session.read_calls),
            "curate_calls": float(session.curate_calls),
            "drop_calls": float(session.drop_calls),
            "invalid_curations": float(session.invalid_curations),
            "off_format_calls": float(session.off_format_calls),
            "off_format_episode": float(session.off_format_calls > 0),
        }
        if session.finished:
            # Emitted only on finished episodes: the trainer averages each key
            # over the trajectories that report it, so these become
            # finished-conditioned means in the logged metrics.
            metrics["precision_given_finish"] = score.precision
            metrics["recall_given_finish"] = score.recall
            metrics["f_beta_given_finish"] = score.f_beta
        return value, metrics


# Flat reward for endings that produced nothing gradable: parse failures,
# overflow before the first sample, and overflow with an empty curated set.
BAD_ENDING_REWARD = -0.2
# Per-turn cap on logged thinking text, so rollout summaries stay readable.
THINKING_LOG_CHARS = 800
RAW_LOG_CHARS = 2500
# The one header form the gpt-oss renderer itself produces for a tool call.
CANONICAL_TOOL_HEADER = re.compile(
    r"<\|channel\|>commentary to=functions\.[a-z0-9_]+ <\|constrain\|>json<\|message\|>"
)


def count_off_format_calls(raw: str) -> int:
    """Tool calls in one sampled turn whose header is not the canonical form."""
    total = raw.count("to=functions.")
    return max(0, total - len(CANONICAL_TOOL_HEADER.findall(raw)))


def _model_input_tokens(model_input) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.chunks:
        tokens.extend(chunk.tokens)
    return tokens


def _token_in_token_out(env: Env, renderer: Renderer) -> Env:
    """Build each next prompt from tokens, not from re-rendered messages.

    The cookbook's EnvFromMessageEnv re-renders the whole conversation from
    parsed messages every turn, so the assistant turn in the next prompt is the
    renderer's canonical version of what the model said, not what it actually
    sampled. Anything the parser tolerates but the renderer does not reproduce
    (drifted tool-call headers, extra spaces, non-canonical channels) then
    breaks the prefix check in trajectory_to_data and the episode trains as
    one sequence per turn. Here the next prompt is literally the previous
    prompt + the sampled action tokens + the rendered tool-result messages +
    the generation suffix, so the prefix property holds by construction and
    the policy trains on exactly the context it sampled from. Paths whose new
    messages are not all tool results (parse-error retries, truncation
    handling) fall back to the full re-render.
    """
    from tinker_cookbook.renderers.base import RenderContext

    state: dict[str, list[int] | None] = {"tokens": None, "action": None}
    original_initial = env.initial_observation
    original_step = env.step
    original_render = env._render_in_thread

    async def initial_observation():
        result = await original_initial()
        if isinstance(result, tuple):
            state["tokens"] = _model_input_tokens(result[0])
        return result

    async def step(action, *, extra=None):
        state["action"] = list(action)
        return await original_step(action, extra=extra)

    async def render(messages, **kwargs):
        previous, action = state["tokens"], state["action"]
        last_assistant = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
        )
        tail = messages[last_assistant + 1 :]
        incremental = (
            previous is not None
            and action is not None
            and last_assistant >= 0
            and tail
            and all(m["role"] == "tool" for m in tail)
            and not kwargs
        )
        if not incremental:
            model_input = await original_render(messages, **kwargs)
            state["tokens"] = _model_input_tokens(model_input)
            return model_input
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        tokens = list(previous) + list(action)
        for offset, message in enumerate(tail):
            idx = last_assistant + 1 + offset
            rendered = renderer.render_message(
                message,
                RenderContext(
                    idx=idx,
                    is_last=idx == len(messages) - 1,
                    prev_message=messages[idx - 1],
                    last_user_index=last_user,
                    in_last_assistant_turn=True,
                ),
            )
            if rendered.header:
                tokens.extend(rendered.header.tokens)
            for chunk in rendered.output:
                if getattr(chunk, "tokens", None):
                    tokens.extend(chunk.tokens)
        tokens.extend(
            renderer._get_generation_suffix(
                "assistant",
                RenderContext(
                    idx=len(messages),
                    is_last=True,
                    prev_message=messages[-1],
                    last_user_index=last_user,
                    in_last_assistant_turn=True,
                ),
            )
        )
        state["tokens"] = tokens
        return tinker.ModelInput.from_ints(tokens)

    env.initial_observation = initial_observation
    env.step = step
    env._render_in_thread = render
    return env


def _grade_overflow_endings(env: Env, tokenizer=None, session: SearchSession | None = None) -> Env:
    """Score episodes that end via context overflow.

    The cookbook's token adapter terminates an overflowing episode with the
    flat context_overflow_reward and never calls reward_fn, so a long episode
    with a good curated set would train on a constant and log no retrieval
    metrics. Our reward reads the session, not the messages, so grade anyway:
    overflow then behaves like max_turns (partial credit, unfinished penalty),
    except that an empty curated set earns the flat BAD_ENDING_REWARD.
    """
    original_step = env.step
    original_message_step = env.message_env.step

    async def logged_message_step(message):
        # The cookbook logs only the assistant's text parts; the analysis
        # (thinking) channel would otherwise never reach the rollout logs.
        # This hook sees the parsed Message (the token-level step above it
        # only sees token ids).
        result = await original_message_step(message)
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            thinking = "".join(
                part.get("thinking", "") for part in content if part.get("type") == "thinking"
            )
            if thinking:
                result.logs["assistant_thinking"] = thinking[:THINKING_LOG_CHARS]
        return result

    env.message_env.step = logged_message_step

    async def graded_step(action, *, extra=None):
        result = await original_step(action, extra=extra)
        # Raw sampled text with special tokens visible, so parse errors and
        # header drift can be read directly from the rollout logs.
        if tokenizer is not None:
            try:
                raw = tokenizer.decode(list(action))
                result.logs["assistant_raw"] = raw[:RAW_LOG_CHARS]
                if session is not None:
                    session.off_format_calls += count_off_format_calls(raw)
            except Exception as exc:  # noqa: BLE001 - logging must never break a step
                logger.debug("could not decode sampled tokens: %s", exc)
        if result.metrics.get("context_overflow"):
            reward_value, reward_metrics = await env.message_env._grade()
            if reward_metrics.get("curated_documents", 0) == 0:
                reward_value = BAD_ENDING_REWARD
            result.reward = reward_value
            result.metrics.update(reward_metrics)
        return result

    env.step = graded_step
    return env


def make_environment(
    *,
    example: QueryExample,
    renderer: Renderer,
    harness_config: HarnessConfig,
    reward: RewardName,
) -> tuple[Env, SearchSession]:
    index = get_index(str(harness_config.data_dir.resolve()))
    session = SearchSession()
    tools = SearchTools(index=index, config=harness_config, session=session)
    reward_fn = RetrievalReward(
        qrels=example.qrels,
        fact_groups=example.fact_groups,
        reward=reward,
        session=session,
        penalize_empty_finish=harness_config.penalize_empty_finish,
        penalize_invalid_curations=harness_config.penalize_invalid_curations,
        penalize_unfinished=harness_config.penalize_unfinished,
        format_penalty=harness_config.format_penalty,
    )
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tools.specs(),
        system_prompt=system_prompt(),
    )
    env = build_agent_tool_env(
        renderer=renderer,
        tools=tools.implementations(),
        initial_messages=prefix + [{"role": "user", "content": example.text}],
        reward_fn=reward_fn,
        rollout_config=RolloutConfig(tool_execution="parallel"),
        max_turns=harness_config.max_turns,
        max_tool_calls=harness_config.max_tool_calls,
        max_trajectory_tokens=harness_config.max_trajectory_tokens,
        max_generation_tokens=harness_config.max_generation_tokens,
        failed_parse_reward=BAD_ENDING_REWARD,
        context_overflow_reward=BAD_ENDING_REWARD,
    )
    return _grade_overflow_endings(
        _token_in_token_out(env, renderer), renderer.tokenizer, session
    ), session


class RetrievalEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        *,
        example: QueryExample,
        model_name: str,
        renderer_name: str | None,
        reward: RewardName,
        group_size: int,
        harness_config: HarnessConfig,
    ):
        self.example = example
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.reward = reward
        self.group_size = group_size
        self.harness_config = harness_config

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)
        return [
            make_environment(
                example=self.example,
                renderer=renderer,
                harness_config=self.harness_config,
                reward=self.reward,
            )[0]
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["sec", self.reward]


class RetrievalDataset(RLDataset):
    """Batches of query builders; past one epoch the order is re-shuffled per epoch.

    The cookbook trains ``min(max_steps, len(dataset))`` batches, so ``epochs``
    is what lets a run (or an in-place ``--resume``) continue beyond the first
    pass over the queries. Epoch 0 keeps the curated interleave from
    ``_sec_train_order``; later epochs use a seeded permutation of it.
    """

    def __init__(
        self,
        builders: list[RetrievalEnvGroupBuilder],
        groups_per_batch: int,
        epochs: int = 1,
        seed: int = 0,
    ):
        self.builders = builders
        self.groups_per_batch = groups_per_batch
        self.epochs = max(1, epochs)
        self.seed = seed

    @property
    def batches_per_epoch(self) -> int:
        return (len(self.builders) + self.groups_per_batch - 1) // self.groups_per_batch

    def _epoch_order(self, epoch: int) -> list[RetrievalEnvGroupBuilder]:
        if epoch == 0:
            return self.builders
        order = list(self.builders)
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        epoch, within = divmod(index, self.batches_per_epoch)
        start = within * self.groups_per_batch
        return self._epoch_order(epoch)[start : start + self.groups_per_batch]

    def __len__(self) -> int:
        return self.batches_per_epoch * self.epochs


@chz.chz
class SecDatasetBuilder(RLDatasetBuilder):
    data_dir: str
    model_name_for_tokenizer: str
    reward: RewardName = "f1"
    renderer_name: str | None = "tml_v0"
    groups_per_batch: int = 16
    group_size: int = 8
    eval_size: int = 32
    eval_group_size: int = 4
    max_turns: int = 40
    max_tool_calls: int = 128
    max_trajectory_tokens: int = 65_536
    max_generation_tokens: int = 2_048
    search_default_k: int = 10
    search_max_k: int = 25
    result_snippet_chars: int = 220
    read_max_chars: int = 4_000
    max_curated_docs: int = 30
    seed: int = 42
    start_batch: int = 0
    epochs: int = 1
    penalize_empty_finish: bool = True
    penalize_invalid_curations: bool = False
    penalize_unfinished: bool = False
    format_penalty: float = 0.0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        harness = self._harness_config()
        train_examples = load_examples(Path(self.data_dir), "train")
        heldout_texts = {
            normalize_query_text(example.text)
            for example in load_examples(Path(self.data_dir), "dev")
        }
        original_count = len(train_examples)
        train_examples = [
            example
            for example in train_examples
            if normalize_query_text(example.text) not in heldout_texts
        ]
        logger.info(
            "Filtered %d train queries with held-out normalized-text duplicates",
            original_count - len(train_examples),
        )
        train_examples = _sec_train_order(train_examples, self.seed)
        if self.start_batch:
            skipped = self.start_batch * self.groups_per_batch
            if skipped >= len(train_examples):
                raise ValueError(
                    f"start_batch={self.start_batch} skips all {len(train_examples)} train queries"
                )
            train_examples = train_examples[skipped:]
            logger.info(
                "Skipping the first %d train queries (start_batch=%d x %d groups); %d remain",
                skipped,
                self.start_batch,
                self.groups_per_batch,
                len(train_examples),
            )
        dev_examples = load_examples(Path(self.data_dir), "dev")[: self.eval_size]
        train_builders = [
            self._make_builder(example, harness, group_size=self.group_size)
            for example in train_examples
        ]
        eval_builders = [
            self._make_builder(example, harness, group_size=self.eval_group_size)
            for example in dev_examples
        ]
        return (
            RetrievalDataset(
                train_builders, self.groups_per_batch, epochs=self.epochs, seed=self.seed
            ),
            RetrievalDataset(eval_builders, max(1, min(self.eval_size, len(eval_builders)))),
        )

    def _harness_config(self) -> HarnessConfig:
        return HarnessConfig(
            data_dir=Path(self.data_dir),
            search_default_k=self.search_default_k,
            search_max_k=self.search_max_k,
            result_snippet_chars=self.result_snippet_chars,
            read_max_chars=self.read_max_chars,
            max_curated_docs=self.max_curated_docs,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            penalize_empty_finish=self.penalize_empty_finish,
            penalize_invalid_curations=self.penalize_invalid_curations,
            penalize_unfinished=self.penalize_unfinished,
            format_penalty=self.format_penalty,
        )

    def _make_builder(
        self,
        example: QueryExample,
        harness: HarnessConfig,
        *,
        group_size: int,
    ) -> RetrievalEnvGroupBuilder:
        return RetrievalEnvGroupBuilder(
            example=example,
            model_name=self.model_name_for_tokenizer,
            renderer_name=self.renderer_name,
            reward=self.reward,
            group_size=group_size,
            harness_config=harness,
        )


def _sec_train_order(examples: list[QueryExample], seed: int) -> list[QueryExample]:
    """Spread the 3/5/7-fact tasks evenly across the epoch.

    A pool in exactly the 8:5:3 mix of the 256-query set uses a fixed 16-query
    interleave. Any other mix (the 1,024-query set is 638/317/69) is
    interleaved in proportion to the pool, so the hardest queries are not
    exhausted early and the last batches are not 3-fact only.
    """
    buckets: dict[int, list[QueryExample]] = {3: [], 5: [], 7: []}
    for example in examples:
        buckets.setdefault(len(example.fact_groups), []).append(example)
    for index, bucket in enumerate(buckets.values()):
        random.Random(seed + index).shuffle(bucket)

    ordered: list[QueryExample] = []
    sizes = {fact_count: len(bucket) for fact_count, bucket in buckets.items()}
    legacy_mix = sizes[3] * 5 == sizes[5] * 8 and sizes[3] * 3 == sizes[7] * 8
    if legacy_mix:
        # Interleaved so even 4- or 8-group smoke batches span all difficulties.
        pattern = (3, 5, 3, 7, 3, 5, 3, 7, 3, 5, 3, 7, 3, 5, 3, 5)
        while any(buckets.values()):
            for fact_count in pattern:
                if buckets.get(fact_count):
                    ordered.append(buckets[fact_count].pop())
            for fact_count in sorted(buckets):
                if fact_count not in {3, 5, 7} and buckets[fact_count]:
                    ordered.append(buckets[fact_count].pop())
        return ordered

    # Smooth weighted round-robin: always take from the bucket that is
    # furthest behind its share of the epoch, so every prefix has roughly the
    # pool's own mix. Ties go to the easier task.
    remaining = {fact_count: len(bucket) for fact_count, bucket in buckets.items() if bucket}
    while remaining:
        fact_count = max(
            sorted(remaining),
            key=lambda count: remaining[count] / sizes[count],
        )
        ordered.append(buckets[fact_count].pop())
        remaining[fact_count] -= 1
        if remaining[fact_count] == 0:
            del remaining[fact_count]
    return ordered
