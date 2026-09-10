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
  first: ``stream_options`` is dropped, because Strands hardcodes
  ``stream_options={"include_usage": true}`` while Miles' session server pops ``stream`` to
  drive the engine itself -- SGLang then rejects the leftover option ("Stream options can only
  be defined when stream=True") and every trajectory dies with a 503 (seen 2026-09-10).
* ``POST /complete-rollout``, ``POST /update-reward`` -- accepted and remembered per
  trajectory (``GET /miles/result/<tid>``); Miles grades on its own side regardless.
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


def make_app() -> FastAPI:
    app = FastAPI()
    sessions: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=10.0))

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

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
            if parsed.pop("stream_options", None) is not None:
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
        results.setdefault(tid, {})["status"] = body.get("Status")
        logger.info("complete-rollout tid=%s status=%s", tid, body.get("Status"))
        return {}

    @app.post("/update-reward")
    async def update_reward(request: Request) -> dict:
        body = await request.json()
        tid = body.get("TrajectoryId", "")
        results.setdefault(tid, {})["rewards"] = body.get("Rewards")
        logger.info("update-reward tid=%s rewards=%s", tid, body.get("Rewards"))
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
