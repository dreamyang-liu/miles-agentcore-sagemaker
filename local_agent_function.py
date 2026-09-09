"""In-cluster agent function that runs the AgentCore episode loop without AgentCore.

Same tools, same prompt, same loop -- ``agent.run_episode`` is imported verbatim from the
container image's source. The only difference is the network path: this talks straight to the
session server instead of going out through the proxy and back.

That makes it the right way to bring the recipe up, because it separates two independent
risks. Here you are testing Miles: FSDP, TITO token accounting, the reward function, and
whether the policy can learn to call the tools at all. Switching
``--custom-agent-function-path`` to ``agentcore_agent_function.run`` then tests exactly one
new thing -- the network path -- with everything else already known to work.

It also proves the agent code is portable: nothing in ``agent.py`` knows or cares whether it
is running in a microVM or next to the trainer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# agent.py lives in ./agent/ and is not a package; add it to the path so the episode loop can
# be shared with the containerised copy rather than duplicated here.
_AGENT_DIR = str(Path(__file__).resolve().parent / "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from agent import run_episode  # noqa: E402  (path must be set first)

logger = logging.getLogger(__name__)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one rollout in-process against the session server."""
    metadata = metadata or {}
    try:
        result = await run_episode(
            {
                # base_url already carries /sessions/<sid>; the session server serves the
                # OpenAI surface under /v1 beneath it.
                "base_url": f"{base_url}/v1",
                # The session server has no auth; the OpenAI SDK still requires a non-empty key.
                "token": "unused-in-cluster",
                "prompt": prompt,
                "sampling_params": request_kwargs or {},
                "instance_id": metadata.get("instance_id"),
                "max_seq_len": metadata.get("max_seq_len"),
            }
        )
    except Exception as exc:
        logger.warning("local episode failed: %s", exc, exc_info=True)
        return None

    return {
        "submitted_answer": result.get("submitted_answer"),
        "exit_status": result.get("exit_status", ""),
        "agent_metrics": result.get("agent_metrics", {}),
    }
