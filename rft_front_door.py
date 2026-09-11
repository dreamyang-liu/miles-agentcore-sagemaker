"""In-VPC front door that makes Miles look like the SageMaker RFT runtime to an RFT agent.

Agents built on ``sagemaker.train.rft`` (Strands ``wrap_model`` + ``sagemaker_rft_handler``) do
not take a per-trajectory model URL: they call ``$RFT_RUNTIME_ENDPOINT/v1/chat/completions`` with
the trajectory identified by the ``X-Amzn-SageMaker-Trajectory-Id`` header, and report results
to ``$RFT_RUNTIME_ENDPOINT/complete-rollout`` and ``/update-reward``. Miles, on the other hand,
gives every trajectory its own session URL. This process bridges the two, on the training
head next to the session servers:

* ``POST /miles/register``  -- the Miles agent function maps a trajectory id to its session URL
  before invoking the agent (loopback only in practice).
* ``POST /v1/chat/completions`` -- looked up by the trajectory header and forwarded to
  ``<session_url>/v1/chat/completions`` on this host; streaming (SSE) is passed through
  byte-for-byte, so TITO recording in the session server is untouched. The body is normalised
  first: text-only content blocks are joined without changing their text, the agent's
  model id is replaced with the local policy alias, and
  ``stream_options`` is dropped, because Strands hardcodes
  ``stream_options={"include_usage": true}`` while Miles' session server pops ``stream`` to
  drive the engine itself -- SGLang then rejects the leftover option ("Stream options can only
  be defined when stream=True") and every trajectory dies with a 503 (seen 2026-09-10).
* ``POST /complete-rollout``, ``POST /update-reward`` -- compatibility acknowledgements
  (200 + ``{}``), including duplicate or late notifications. They never complete a Miles
  session or apply a training reward. While a trajectory is registered, its latest feedback
  is available at ``GET /miles/result/<tid>`` for diagnostics; unregistering discards it.
  Miles owns the actual lifecycle and grades on its own side.
* ``GET /health``.

Only ``/v1/chat/completions`` is forwarded. The session server's catch-all reaches the SGLang
control plane, so nothing else is exposed through here.

Usage:
    python rft_front_door.py [--host 0.0.0.0] [--port 30100]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("rft-front-door")

TRAJECTORY_HEADER = "x-amzn-sagemaker-trajectory-id"
# A single trajectory can take many minutes: multi-turn agent, weight sync pauses in between.
UPSTREAM_TIMEOUT_S = 1800.0
# Each streaming model call holds a connection until its response is consumed.
# Leave room for the delegated 128-way rollout fan-out and short overlap.
UPSTREAM_MAX_CONNECTIONS = 256
UPSTREAM_MAX_KEEPALIVE_CONNECTIONS = 128


def make_app(
    *,
    allowed_peers: set[str] | None = None,
    control_peers: set[str] | None = None,
    policy_model: str = "model",
) -> FastAPI:
    app = FastAPI()
    local_peers = {"127.0.0.1", "::1"} | (control_peers or set())
    sessions: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=10.0),
        limits=httpx.Limits(
            max_connections=UPSTREAM_MAX_CONNECTIONS,
            max_keepalive_connections=UPSTREAM_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )
    logger.info(
        "upstream pool max_connections=%d max_keepalive_connections=%d",
        UPSTREAM_MAX_CONNECTIONS, UPSTREAM_MAX_KEEPALIVE_CONNECTIONS,
    )

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    @app.middleware("http")
    async def restrict_peers(request: Request, call_next):
        peer = request.client.host if request.client else ""
        # Runtime agents only need inference and feedback. Session registration and
        # diagnostics belong to the local Miles caller, including on a public listener.
        if request.url.path.startswith("/miles/") and peer not in local_peers:
            return JSONResponse({"error": "local control endpoint"}, status_code=403)
        if allowed_peers is not None and peer not in allowed_peers | local_peers:
            return JSONResponse({"error": "peer not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy", "sessions": len(sessions)}

    @app.post("/miles/register")
    async def register(request: Request) -> dict:
        body = await request.json()
        tid, url = body["trajectory_id"], body["session_url"].rstrip("/")
        sessions[tid] = url
        results.pop(tid, None)
        logger.info("register tid=%s -> %s", tid, url)
        return {"ok": True}

    @app.delete("/miles/register/{tid}")
    async def unregister(tid: str) -> dict:
        sessions.pop(tid, None)
        results.pop(tid, None)
        return {"ok": True}

    @app.get("/miles/result/{tid}")
    async def result(tid: str) -> dict:
        return results.get(tid, {})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        tid = request.headers.get(TRAJECTORY_HEADER, "")
        url = sessions.get(tid)
        if not url:
            logger.warning("unknown trajectory %r from %s", tid, request.client.host if request.client else "?")
            raise HTTPException(status_code=404, detail="unknown trajectory id")
        body = await request.body()
        wants_stream = False
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            wants_stream = bool(parsed.get("stream", False))
            parsed.pop("stream_options", None)
            # Strands represents text as content-block lists. The Qwen3 TITO template
            # renders non-string content as empty, silently dropping the question.
            # Join text verbatim for every role; preserve non-text blocks and all
            # other message fields (tool calls, tool_call_id, reasoning_content).
            messages = parsed.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    if isinstance(content, list) and all(
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                        for block in content
                    ):
                        message["content"] = "".join(block["text"] for block in content)
            # RFT agents can carry a provider model id such as anthropic....:0.
            # SGLang interprets its suffix as a LoRA adapter. Route to the Miles
            # policy instead, as the native agent does; Miles itself owns lora_path.
            parsed["model"] = policy_model
            body = json.dumps(parsed).encode()
        headers = {"content-type": request.headers.get("content-type", "application/json")}
        started = time.monotonic()
        upstream_request = client.build_request("POST", f"{url}/v1/chat/completions", content=body, headers=headers)
        upstream = await client.send(upstream_request, stream=True)
        logger.info(
            "forward tid=%s stream=%s -> %s %s (%.2fs to headers) client=%s",
            tid, wants_stream, url, upstream.status_code, time.monotonic() - started,
            request.client.host if request.client else "?",
        )
        media_type = upstream.headers.get("content-type", "application/json")
        if wants_stream and upstream.status_code == 200:
            async def relay():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
            return StreamingResponse(relay(), status_code=upstream.status_code, media_type=media_type)
        content = await upstream.aread()
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code, media_type=media_type)

    @app.post("/complete-rollout")
    async def complete_rollout(request: Request) -> dict:
        body = await request.json()
        tid = body.get("TrajectoryId", "")
        tracked = tid in sessions
        if tracked:
            results.setdefault(tid, {})["status"] = body.get("Status")
        # Only Miles unregisters a session. SDK completion/error/retry notifications are
        # acknowledged even after it has done so, without recreating retained state.
        logger.info("complete-rollout tid=%s status=%s tracked=%s", tid, body.get("Status"), tracked)
        return {}

    @app.post("/update-reward")
    async def update_reward(request: Request) -> dict:
        body = await request.json()
        tid = body.get("TrajectoryId", "")
        tracked = tid in sessions
        if tracked:
            results.setdefault(tid, {})["rewards"] = body.get("Rewards")
        # Keep SDK feedback diagnostic-only: math_reward computes the training signal.
        logger.info("update-reward tid=%s rewards=%s tracked=%s", tid, body.get("Rewards"), tracked)
        return {}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(path: str) -> JSONResponse:
        logger.warning("rejected /%s", path)
        return JSONResponse({"error": "not exposed"}, status_code=404)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--policy-model", default="model", help="local policy alias sent to the session server")
    parser.add_argument("--ssl-certfile", help="PEM certificate chain for an HTTPS listener")
    parser.add_argument("--ssl-keyfile", help="PEM private key for an HTTPS listener")
    parser.add_argument("--allow-peer", action="append", help="allowed runtime peer IP; repeatable")
    parser.add_argument("--control-peer", action="append", default=[], help="additional local caller IP; repeatable")
    args = parser.parse_args()
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("--ssl-certfile and --ssl-keyfile must be supplied together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        make_app(
            allowed_peers=set(args.allow_peer) if args.allow_peer else None,
            control_peers=set(args.control_peer),
            policy_model=args.policy_model,
        ),
        host=args.host, port=args.port, log_level="info",
        ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile,
        # There is no reverse proxy in this deployment. Do not let forwarded headers
        # change the socket peer used by the access rules.
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
