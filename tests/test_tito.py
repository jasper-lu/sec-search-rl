"""Token-in/token-out prompt building must match the renderer's full re-render
for canonical output, and must preserve the sampled tokens verbatim for
non-canonical output (which the full re-render would rewrite)."""

from __future__ import annotations

import asyncio

import pytest
from tinker_cookbook import renderers
from tinker_cookbook.rl.data_processing import _is_prefix
from tinker_cookbook.tokenizer_utils import get_tokenizer

from sec_rl.environment import _model_input_tokens, _token_in_token_out


@pytest.fixture(scope="module")
def renderer():
    try:
        tokenizer = get_tokenizer("openai/gpt-oss-20b")
    except Exception as exc:  # noqa: BLE001 - tokenizer must be cached locally
        pytest.skip(f"gpt-oss tokenizer unavailable: {exc}")
    return renderers.get_renderer("gpt_oss_no_sysprompt", tokenizer)


class FakeEnv:
    """Just enough of EnvFromMessageEnv for the wrapper: it re-renders the
    whole conversation each turn."""

    def __init__(self, renderer, messages):
        self.renderer = renderer
        self.history = list(messages)

    async def initial_observation(self):
        return self.renderer.build_generation_prompt(self.history), None

    async def step(self, action, *, extra=None):
        message, _ = self.renderer.parse_response(action)
        self.history.append(message)
        for call in message.get("tool_calls") or []:
            self.history.append(
                {
                    "role": "tool",
                    "name": call.function.name,
                    "content": '{"results": []}',
                    "tool_call_id": "x",
                }
            )
        return await self._render_in_thread(self.history)

    async def _render_in_thread(self, messages, **kwargs):
        return self.renderer.build_generation_prompt(messages, **kwargs)


def _run(renderer, response_text: str):
    tokenizer = renderer.tokenizer
    messages = [
        {"role": "system", "content": "You search."},
        {"role": "user", "content": "Find it."},
    ]
    tito = _token_in_token_out(FakeEnv(renderer, messages), renderer)
    plain = FakeEnv(renderer, messages)
    prompt, _ = asyncio.run(tito.initial_observation())
    action = tokenizer.encode(response_text, add_special_tokens=False)
    next_tito = asyncio.run(tito.step(action))
    asyncio.run(plain.initial_observation())
    next_plain = asyncio.run(plain.step(action))
    return (
        _model_input_tokens(prompt),
        action,
        _model_input_tokens(next_tito),
        _model_input_tokens(next_plain),
    )


def test_tito_matches_full_render_for_canonical_output(renderer) -> None:
    text = (
        "<|channel|>analysis<|message|>Search first.<|end|><|start|>assistant"
        '<|channel|>commentary to=functions.bm25_search <|constrain|>json<|message|>{"query": "acme", "k": 5}<|call|>'
    )
    prompt, action, tito_next, plain_next = _run(renderer, text)
    assert tito_next == plain_next
    assert _is_prefix(prompt + action, tito_next)


def test_tito_keeps_sampled_tokens_for_drifted_header(renderer) -> None:
    text = (
        "<|channel|>analysis<|message|>Search first.<|end|><|start|>assistant"
        '<|channel|>commentary to=functions.bm25_search <|constrain|>rejson<|message|>{"query": "acme", "k": 5}<|call|>'
    )
    prompt, action, tito_next, plain_next = _run(renderer, text)
    # Full re-render canonicalises the header and breaks the prefix; TITO does not.
    assert not _is_prefix(prompt + action, plain_next)
    assert _is_prefix(prompt + action, tito_next)
    # ...and the tool result + generation suffix after the action are identical.
    assert (
        tito_next[len(prompt) + len(action) :]
        == plain_next[len(plain_next) - (len(tito_next) - len(prompt) - len(action)) :]
    )
