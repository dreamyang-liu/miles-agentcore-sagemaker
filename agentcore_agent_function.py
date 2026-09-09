"""Miles agent function that dispatches each rollout to a Bedrock AgentCore runtime.

The AgentCore analogue of ``swe_agent_function.run``: instead of ``POST /run`` on a Harbor
server, it calls ``InvokeAgentRuntime``. Everything else in the rollout stack is untouched --
``agentic_tool_call.generate`` still opens the TITO session and collects the samples, and
``--custom-rm-path`` still owns the reward.

The one piece of real work here is closing the network loop, and there are two ways:

* **Proxy** (``MILES_PROXY_BASE`` set): Miles' session URL is cluster-private, so we mint a
  short-lived HMAC token carrying the upstream ``ip``/``port`` and hand the agent a proxy URL.
  The proxy verifies the token and reads the target out of it. See ``proxy.py``.
* **Direct** (``MILES_PROXY_BASE`` unset): the runtime is attached to the same VPC as the
  training hosts (AgentCore ``networkMode: VPC``, e.g. a SageMaker training job), so the
  agent gets the session URL as-is. The security group is the whole boundary here -- the
  session server has no authentication of its own. See ``sagemaker/``.

Environment:
    AGENTCORE_RUNTIME_ARN   required -- the runtime to invoke
    MILES_PROXY_BASE        optional -- e.g. https://proxy.example.com (no trailing /); unset = direct
    MILES_PROXY_SECRET      required with MILES_PROXY_BASE -- same secret the proxy verifies with
    AWS_REGION              defaults to us-west-2
    AGENT_TRIAL_TIMEOUT     per-trial ceiling in seconds (default 900)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any
from urllib.parse import urlsplit

import boto3
from botocore.config import Config

# Resolved via PYTHONPATH, which the launcher points at this directory (same convention as
# the Harbor recipe's `swe_agent_function`).
from proxy import sign_token

logger = logging.getLogger(__name__)

# Synchronous InvokeAgentRuntime is capped at 15 minutes by AgentCore. GSM8K episodes are a
# handful of short turns, so the synchronous path is right here; a long agentic trial
# (SWE-bench-scale) has to move to the async job path, which allows 8 hours.
_DEFAULT_TRIAL_TIMEOUT_S = 900

# AgentCore requires runtimeSessionId >= 33 chars; uuid4().hex is 32, hence the prefix.
_SESSION_PREFIX = "miles-"

# Cap on invocations in flight from this process, which decouples the GRPO batch shape from
# AgentCore's capacity. Two account limits bite here, and neither scales with your batch:
#   * maxVms -- concurrent microVMs. NOT exposed in Service Quotas, so it cannot be queried;
#     128 concurrent trials exceeded it, 32 did not.
#   * "Rate of new Runtime session creation" -- 25/s (adjustable).
# Without this cap a large rollout_batch_size turns into a quota storm: every trial fails,
# check_no_aborted drops the sample, Miles resamples, and the run live-locks.
_MAX_CONCURRENT = int(os.environ.get("AGENTCORE_MAX_CONCURRENT", "16"))

# Retryable throttles get an exponential backoff instead of costing the sample. A trial that
# gives up here takes its whole GRPO group down with it, so it is worth waiting.
_RETRYABLE = ("ThrottlingException", "ServiceQuotaExceededException", "RetryableConflictException")
_MAX_ATTEMPTS = int(os.environ.get("AGENTCORE_MAX_ATTEMPTS", "6"))
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 45.0

# Sessions in flight from this process, so `abort` knows what to stop.
_active_sessions: set[str] = set()

_client = None
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Created lazily so it binds to the running rollout loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


def _trial_timeout_s() -> int:
    return int(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_TRIAL_TIMEOUT_S))


def _runtime_client():
    """One lazily-built boto3 client per worker process; boto3 clients are thread-safe."""
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-agentcore",
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
            config=Config(
                read_timeout=_trial_timeout_s(),
                connect_timeout=30,
                # Miles already retries at the sample level; a boto retry here would
                # duplicate a trial that is merely slow.
                retries={"max_attempts": 1, "mode": "standard"},
                max_pool_connections=64,
            ),
        )
    return _client


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} must be set for the AgentCore agent function")
    return value


def _session_id_of(base_url: str) -> str:
    """Extract the Miles session id from ``http://<ip>:<port>/sessions/<sid>``."""
    return urlsplit(base_url).path.rstrip("/").rsplit("/", 1)[-1]


def build_proxy_url(base_url: str, session_server_id: str, secret: str, proxy_base: str, ttl: float) -> tuple[str, str]:
    """Return ``(proxy_url, token)`` for this session.

    ``session_server_id`` is the ``ip:port`` of the instance that owns the session, which is
    authoritative when several session-server workers are running -- the port in ``base_url``
    is the same value, but this field is what the generate layer promises.
    """
    session_id = _session_id_of(base_url)
    host, _, port = session_server_id.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"malformed session_server_id: {session_server_id!r}")
    token = sign_token({"sid": session_id, "ip": host, "port": int(port), "exp": time.time() + ttl}, secret)
    return f"{proxy_base.rstrip('/')}/s/{session_id}/v1", token


def build_direct_url(base_url: str) -> tuple[str, str]:
    """Return ``(url, token)`` for the direct path: the session URL itself plus ``/v1``.

    ``base_url`` is ``http://<session-server-ip>:<port>/sessions/<sid>`` and that ip:port is
    already the one owning the session, so nothing has to be rewritten. The token is a
    placeholder: the OpenAI client insists on an api_key, the session server ignores it.
    """
    return f"{base_url.rstrip('/')}/v1", "direct-no-auth"


def _invoke(arn: str, session_id: str, payload: bytes) -> dict:
    """Blocking InvokeAgentRuntime call; the caller runs it off the event loop."""
    response = _runtime_client().invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=session_id,
        payload=payload,
        contentType="application/json",
        accept="application/json",
    )
    body = response["response"].read()
    return json.loads(body) if body else {}


async def _invoke_with_backoff(arn: str, session_id: str, body: bytes, timeout: int) -> dict | None:
    """Invoke with exponential backoff on throttles; None once the trial is genuinely lost."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(asyncio.to_thread(_invoke, arn, session_id, body), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("AgentCore invocation timed out after %ss (session=%s)", timeout, session_id)
            return None
        except Exception as exc:
            name = type(exc).__name__
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            if code not in _RETRYABLE and not any(r in str(exc) for r in _RETRYABLE):
                logger.warning("AgentCore invocation failed (session=%s): %s: %s", session_id, name, exc)
                return None
            if attempt == _MAX_ATTEMPTS:
                logger.warning(
                    "AgentCore still throttled after %s attempts (session=%s): %s", _MAX_ATTEMPTS, session_id, code
                )
                return None
            # Full jitter: synchronised retries across a rollout are what caused the storm.
            delay = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)
            logger.info(
                "AgentCore throttled (%s), attempt %s/%s, sleeping %.1fs", code or name, attempt, _MAX_ATTEMPTS, delay
            )
            await asyncio.sleep(delay)
    return None


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one rollout as one AgentCore invocation."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    arn = _require("AGENTCORE_RUNTIME_ARN")
    proxy_base = os.environ.get("MILES_PROXY_BASE", "")
    timeout = _trial_timeout_s()

    session_server_id = metadata.get("session_server_id")
    if not session_server_id:
        # Set by agentic_tool_call.generate from the tracer; without it we cannot tell the
        # proxy which of the N session-server workers owns this session.
        logger.error("metadata is missing session_server_id; is --use-session-server set?")
        return None

    if proxy_base:
        secret = _require("MILES_PROXY_SECRET")
        try:
            # Outlive the trial ceiling so a token cannot expire mid-episode.
            proxy_url, token = build_proxy_url(base_url, session_server_id, secret, proxy_base, timeout + 120)
        except ValueError as exc:
            logger.error("could not build proxy URL: %s", exc)
            return None
    else:
        proxy_url, token = build_direct_url(base_url)

    payload = {
        "base_url": proxy_url,
        "token": token,
        "prompt": prompt,
        "sampling_params": request_kwargs,
        "instance_id": metadata.get("instance_id"),
        "max_seq_len": metadata.get("max_seq_len"),
    }
    # Reuse the Miles session id so one trajectory maps to exactly one sticky microVM.
    runtime_session_id = _SESSION_PREFIX + _session_id_of(base_url)

    started = time.monotonic()
    _active_sessions.add(runtime_session_id)
    body = json.dumps(payload).encode()
    try:
        async with _get_semaphore():
            result = await _invoke_with_backoff(arn, runtime_session_id, body, timeout)
    except asyncio.CancelledError:
        # Oversampling abort cancels siblings; expected, not an error.
        logger.info("AgentCore invocation cancelled (session=%s)", runtime_session_id)
        return None
    finally:
        _active_sessions.discard(runtime_session_id)
    if result is None:
        return None

    logger.info(
        "trial done in %.1fs: status=%s answer=%r",
        time.monotonic() - started,
        result.get("exit_status"),
        result.get("submitted_answer"),
    )
    # Merged into every sample's metadata by the generate layer, where the reward function
    # reads submitted_answer and compares it against the ground truth.
    return {
        "submitted_answer": result.get("submitted_answer"),
        "exit_status": result.get("exit_status", ""),
        "agent_metrics": result.get("agent_metrics", {}),
    }


async def abort(args) -> None:
    """Stop in-flight AgentCore sessions when Miles aborts oversampling.

    Without this, a cancelled trial keeps looping against SGLang until it hits its own turn
    limit, holding a session slot and burning rollout capacity. The AgentCore analogue of
    Harbor's ``/flush``.
    """
    arn = os.environ.get("AGENTCORE_RUNTIME_ARN", "")
    if not arn or not _active_sessions:
        return
    client = _runtime_client()
    for session_id in list(_active_sessions):
        try:
            await asyncio.to_thread(client.stop_runtime_session, agentRuntimeArn=arn, runtimeSessionId=session_id)
            logger.info("stopped AgentCore session %s", session_id)
        except Exception as exc:
            logger.warning("could not stop AgentCore session %s: %s", session_id, exc)
        finally:
            _active_sessions.discard(session_id)
