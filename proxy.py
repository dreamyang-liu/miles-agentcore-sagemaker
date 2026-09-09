"""Edge proxy that lets a Bedrock AgentCore agent reach a Miles session server.

Miles' session server has no authentication and a catch-all route that forwards any
path straight to the SGLang router, so it must never be exposed directly. This proxy
is the only thing that faces AgentCore. It does three jobs:

1. **Verify** a short-lived HMAC token minted by the Miles-side agent function. The
   token carries the upstream ``ip``/``port``, so the proxy holds no state at all and
   can be replicated freely -- no Redis, no service discovery.
2. **Whitelist** paths. Only the two chat endpoints are reachable; ``POST /sessions``,
   ``DELETE /sessions/{id}`` and the catch-all SGLang passthrough are not.
3. **Forward** to the exact session-server instance named in the token. Session state
   lives in one process's memory, so routing is deterministic and never balanced.

Usage:
    # serve (reads the shared secret from $MILES_PROXY_SECRET)
    python proxy.py serve --host 0.0.0.0 --port 8080

    # mint a token by hand, for connectivity testing
    python proxy.py sign --sid abc123 --ip 10.0.3.17 --port 30007 --ttl 7200
"""

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import time

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("miles-agentcore-proxy")

SECRET_ENV = "MILES_PROXY_SECRET"

# Never forwarded upstream: framing headers, plus the proxy's own credential -- the
# session server does not authenticate and must not see it.
_DROP_REQUEST_HEADERS = frozenset({"host", "content-length", "transfer-encoding", "authorization", "x-api-key"})
# Dropped from the upstream reply so our ASGI server reframes the body it actually sends.
_DROP_RESPONSE_HEADERS = frozenset({"content-length", "transfer-encoding", "server", "date"})

# The only paths reachable from outside. Anything else is a 404 by construction.
_ALLOWED_SUBPATHS = ("v1/chat/completions", "v1/messages")


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64u_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_token(payload: dict, secret: str) -> str:
    """Mint ``<body>.<sig>``; the body carries sid/ip/port/exp so the proxy stays stateless."""
    body = _b64u_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64u_encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(token: str, secret: str) -> dict:
    """Return the token payload, or raise ValueError. Signature is checked before parsing."""
    body, _, sig = token.partition(".")
    if not body or not sig:
        raise ValueError("malformed token")
    expected = _b64u_encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    # compare_digest: constant time, so a bad token leaks no information about the secret.
    if not hmac.compare_digest(sig, expected):
        raise ValueError("bad signature")
    try:
        payload = json.loads(_b64u_decode(body))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"undecodable payload: {exc}") from exc
    for field in ("sid", "ip", "port", "exp"):
        if field not in payload:
            raise ValueError(f"token missing {field!r}")
    if float(payload["exp"]) < time.time():
        raise ValueError("token expired")
    return payload


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        # Also accept the OpenAI SDK's api_key when a client sends it as x-api-key.
        token = request.headers.get("x-api-key", "")
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    return token.strip()


def _authorize(request: Request, sid: str, secret: str) -> dict:
    try:
        claims = verify_token(_bearer(request), secret)
    except ValueError as exc:
        logger.warning("[reject] sid=%s reason=%s", sid, exc)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    # Binding the token to its session id stops one trial's token from touching another's.
    if claims["sid"] != sid:
        logger.warning("[reject] sid=%s reason=token bound to %s", sid, claims["sid"])
        raise HTTPException(status_code=403, detail="token does not match session")
    return claims


def make_app(secret: str, *, upstream_timeout: float = 900.0) -> FastAPI:
    app = FastAPI(title="miles-agentcore-proxy")
    # No connection-level timeout ceiling below the session server's own (600s default):
    # a long completion must not be cut off here.
    client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        timeout=httpx.Timeout(upstream_timeout, connect=10.0),
    )
    app.router.on_shutdown.append(client.aclose)

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    async def forward(request: Request, sid: str, subpath: str):
        claims = _authorize(request, sid, secret)
        upstream = f"http://{claims['ip']}:{claims['port']}/sessions/{sid}/{subpath}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
        body = await request.body()

        started = time.monotonic()
        upstream_request = client.build_request("POST", upstream, content=body, headers=headers)
        try:
            response = await client.send(upstream_request, stream=True)
        except httpx.TransportError as exc:
            logger.warning("[502] sid=%s upstream=%s %s: %s", sid, upstream, type(exc).__name__, exc)
            return JSONResponse({"error": f"upstream unreachable: {type(exc).__name__}"}, status_code=502)

        logger.info(
            "[%s] sid=%s %s upstream=%s in %.2fs",
            response.status_code,
            sid,
            subpath,
            f"{claims['ip']}:{claims['port']}",
            time.monotonic() - started,
        )
        out = {k: v for k, v in response.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
        # Streamed straight through with aiter_raw (no re-decoding), and x-accel-buffering
        # keeps any intermediate reverse proxy from buffering an SSE reply.
        out["x-accel-buffering"] = "no"
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=out,
            background=BackgroundTask(response.aclose),
        )

    @app.post("/s/{sid}/v1/chat/completions")
    async def chat_completions(request: Request, sid: str):
        return await forward(request, sid, "v1/chat/completions")

    @app.post("/s/{sid}/v1/messages")
    async def anthropic_messages(request: Request, sid: str):
        return await forward(request, sid, "v1/messages")

    return app


def _require_secret() -> str:
    secret = os.environ.get(SECRET_ENV, "")
    if len(secret) < 16:
        raise SystemExit(f"${SECRET_ENV} must be set to at least 16 chars (openssl rand -hex 32)")
    return secret


def _cmd_serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = make_app(_require_secret(), upstream_timeout=args.upstream_timeout)
    logger.info("serving on %s:%s; allowed subpaths=%s", args.host, args.port, list(_ALLOWED_SUBPATHS))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_keep_alive=args.keep_alive)


def _cmd_sign(args: argparse.Namespace) -> None:
    claims = {"sid": args.sid, "ip": args.ip, "port": args.port, "exp": time.time() + args.ttl}
    print(sign_token(claims, _require_secret()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the proxy")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--upstream-timeout",
        type=float,
        default=900.0,
        help="Must exceed the session server's --miles-router-timeout (default 600s).",
    )
    serve.add_argument("--keep-alive", type=int, default=900, help="Idle keep-alive seconds.")
    serve.set_defaults(func=_cmd_serve)

    sign = sub.add_parser("sign", help="mint a token for manual testing")
    sign.add_argument("--sid", required=True)
    sign.add_argument("--ip", required=True)
    sign.add_argument("--port", type=int, required=True)
    sign.add_argument("--ttl", type=float, default=7200.0)
    sign.set_defaults(func=_cmd_sign)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
