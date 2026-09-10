"""Session history matcher for agents that cannot echo ``reasoning_content`` back.

Miles' default ``strict`` matcher compares the four template-relevant message keys: ``role``,
``content``, ``reasoning_content`` and ``tool_calls``. A thinking model (Qwen3.6 and friends,
whose TITO template sets ``preserve_thinking=True``) returns its chain of thought in a separate
``reasoning_content`` field, so an agent that keeps only ``content`` + ``tool_calls`` in its
history replays a message that never matches what the session stored. The session then rolls
back to the empty checkpoint on *every* turn: history grows unbounded, prompts reach tens of
thousands of tokens, trials go from seconds to minutes, and each turn's recorded tokens are
discarded. Measured 2026-09-10 on Qwen3.6-27B: 5312 rollbacks, all to ``checkpoint -1``,
sessions at 1309 messages, trials 5s -> 311s, and the run stalled.

Our own agent is fixed to echo the field (see ``agent/agent.py``). A third-party agent cannot
be: Strands' ``OpenAIModel`` drops it too, and the point of ``rft_front_door.py`` is to run such
an agent unmodified. For that case select this matcher::

    --session-message-matcher session_message_matcher.matches

It is ``loose_tool_call`` minus ``reasoning_content``: role, content and the whole tool-call
structure (id, type, name, JSON-normalised arguments, order, count) are still compared, so the
only thing it forgives is a chain of thought the agent never saw. That is far narrower than the
built-in ``role_content_only``, which ignores ``tool_calls`` entirely.

The trade-off is real but bounded: the stored prefix wins, so the tokens trained on are the ones
the engine actually generated -- including the thinking the agent dropped. A turn whose *only*
difference is reasoning content is treated as a replay of the stored turn, which is exactly the
intent; a turn that differs in content or tool calls still mismatches and still rolls back.
"""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub import loose_tool_call_message_matches

_IGNORED_KEYS = ("reasoning_content",)


def matches(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """``loose_tool_call`` with ``reasoning_content`` projected out of both sides."""
    # Copy rather than mutate: these dicts are the live session history and the request body.
    stored_projected = {k: v for k, v in stored.items() if k not in _IGNORED_KEYS}
    replayed_projected = {k: v for k, v in replayed.items() if k not in _IGNORED_KEYS}
    return loose_tool_call_message_matches(stored_projected, replayed_projected)
