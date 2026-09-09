"""Miles agent function for agents built on the SageMaker RFT contract (``sagemaker.train.rft``).

Same job as ``agentcore_agent_function.run`` -- one Miles trajectory = one AgentCore
invocation -- but the wire format is the RFT one, so an unmodified RFT agent (Strands +
``sagemaker_rft_handler``) can drive a Miles policy:

* payload ``{"prompt": <JSON string of the data record>, "metadata": {job_arn, trajectory_id,
  endpoint, region}, "inferenceParams": {temperature, max_tokens, top_p}}``;
* the agent calls ``$RFT_RUNTIME_ENDPOINT/v1/chat/completions`` with the trajectory id in a
  header, so before invoking we register ``trajectory_id -> session URL`` with the in-VPC
  front door (``rft_front_door.py``) that the runtime's ``RFT_RUNTIME_ENDPOINT`` points at;
* the reply is ``{status, agent_answer, reward, trajectory_reward, ...}``. We map
  ``agent_answer`` to ``submitted_answer`` so ``math_reward.py`` grades exactly as for the
  native agent; the agent's own reward is kept in ``agent_metrics`` for comparison.

The RFT contract requires ``reward_spec.ground_truth`` inside the prompt record (the agent
refuses otherwise and computes its own reward from it), so unlike the native recipe the ground
truth does cross the wire here -- inside the VPC.

Environment:
    AGENTCORE_RUNTIME_ARN          required
    MILES_RFT_FRONT_DOOR_URL       required -- what the agent resolves, e.g. http://miles-head.miles.internal:30100
    MILES_RFT_FRONT_DOOR_LOCAL     how this process reaches the front door (default http://127.0.0.1:30100)
    TRAINING_JOB_ARN               used as metadata.job_arn when present (SageMaker sets it)
    AWS_REGION, AGENTCORE_MAX_CONCURRENT, AGENT_TRIAL_TIMEOUT   as for agentcore_agent_function
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

from agentcore_agent_function import (
    _active_sessions,
    _get_semaphore,
    _invoke_with_backoff,
    _require,
    _session_id_of,
    _trial_timeout_s,
    abort,  # noqa: F401  -- re-exported: Miles looks up `abort` next to `run`
)

logger = logging.getLogger(__name__)

_SESSION_PREFIX = "miles-rft-"  # + 32 hex chars = 42 >= AgentCore's 33-char minimum
_INFERENCE_PARAM_KEYS = ("temperature", "max_tokens", "top_p")


def _front_door_local() -> str:
    return os.environ.get("MILES_RFT_FRONT_DOOR_LOCAL", "http://127.0.0.1:30100").rstrip("/")


def _rft_record(prompt: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """The record the agent parses out of ``payload["prompt"]``.

    ``convert_rft_parquet.py`` keeps the original under ``metadata.rft_record``; otherwise one
    is built from the messages and the ground truth the reward function already carries.
    """
    if record := metadata.get("rft_record"):
        return record
    instance = ""
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                instance = str(message.get("content", ""))
                break
    return {
        "instance_id": metadata.get("instance_id"),
        "data_source": "miles",
        "instance": instance,
        "prompt": prompt,
        "reward_spec": {"ground_truth": str(metadata.get("answer", ""))},
        "extra_info": {},
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one rollout as one RFT-contract AgentCore invocation."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}
    arn = _require("AGENTCORE_RUNTIME_ARN")
    public_front_door = _require("MILES_RFT_FRONT_DOOR_URL").rstrip("/")
    timeout = _trial_timeout_s()
    sid = _session_id_of(base_url)

    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(f"{_front_door_local()}/miles/register", json={"trajectory_id": sid, "session_url": base_url})

    payload = {
        "prompt": json.dumps(_rft_record(prompt, metadata), ensure_ascii=False),
        "metadata": {
            "job_arn": os.environ.get("TRAINING_JOB_ARN", "arn:aws:sagemaker:::training-job/miles"),
            "trajectory_id": sid,
            "endpoint": public_front_door,
            "region": os.environ.get("AWS_REGION", "us-west-2"),
        },
        "inferenceParams": {k: request_kwargs[k] for k in _INFERENCE_PARAM_KEYS if request_kwargs.get(k) is not None},
    }
    runtime_session_id = _SESSION_PREFIX + sid

    started = time.monotonic()
    _active_sessions.add(runtime_session_id)
    try:
        async with _get_semaphore():
            result = await _invoke_with_backoff(arn, runtime_session_id, json.dumps(payload).encode(), timeout)
    except asyncio.CancelledError:
        logger.info("AgentCore invocation cancelled (session=%s)", runtime_session_id)
        return None
    finally:
        _active_sessions.discard(runtime_session_id)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.delete(f"{_front_door_local()}/miles/register/{sid}")
        except httpx.HTTPError:
            pass
    if result is None:
        return None

    status = result.get("status", "")
    answer = result.get("agent_answer") or None
    if status != "success":
        logger.warning("RFT agent returned status=%s error=%s (session=%s)", status, result.get("error"), sid)
    exit_status = "submitted" if answer else ("error" if status == "error" else "stopped_without_submitting")
    per_turn = result.get("per_turn_rewards") or []
    logger.info(
        "trial done in %.1fs: status=%s answer=%r agent_reward=%s",
        time.monotonic() - started, exit_status, answer, result.get("trajectory_reward"),
    )
    return {
        "submitted_answer": answer,
        "exit_status": exit_status,
        "agent_metrics": {
            "turns": len(per_turn),
            "rft_reward": float(result.get("trajectory_reward") or 0.0),
            "total_time": time.monotonic() - started,
        },
    }
