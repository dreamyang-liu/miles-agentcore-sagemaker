"""End-to-end check of the proxy leg, with no AWS account and no GPUs.

Starts the fake session server and the proxy, then drives the real agent loop from
``agent/agent.py`` through the proxy exactly as AgentCore would. Verifies the happy path
(multi-turn tool calling, streaming) and, just as importantly, that the path whitelist and
token binding actually reject what they are supposed to.

Run this before touching AgentCore: if it passes, the only remaining unknowns are AWS-side
(ARM64 image, IAM, network mode).

Usage:
    python smoke_test.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSION_PORT = 31907
PROXY_PORT = 18080
SID = "deadbeef" * 4  # 32 hex chars, same shape as uuid4().hex


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _wait_healthy(port: int, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    raise RuntimeError(f"nothing healthy on port {port} after {timeout}s")


def _post(url: str, body: dict, token: str | None) -> tuple[int, str]:
    data = json.dumps(body).encode()
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


class Checks:
    def __init__(self):
        self.failures: list[str] = []

    def expect(self, label: str, actual, wanted) -> None:
        ok = actual == wanted
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {actual!r}, want {wanted!r}")
        if not ok:
            self.failures.append(label)

    def expect_true(self, label: str, value: bool, detail: str = "") -> None:
        print(f"  {'PASS' if value else 'FAIL'}  {label}{f' -- {detail}' if detail else ''}")
        if not value:
            self.failures.append(label)


def main() -> int:
    proxy_mod = _load("acproxy", HERE / "proxy.py")
    agent_mod = _load("acagent", HERE / "agent" / "agent.py")

    secret = secrets.token_hex(32)
    env = {**os.environ, proxy_mod.SECRET_ENV: secret}

    procs = [
        subprocess.Popen(
            [sys.executable, str(HERE / "fake_session_server.py"), "--port", str(SESSION_PORT)],
            env=env,
            stderr=subprocess.PIPE,
        ),
        subprocess.Popen(
            [sys.executable, str(HERE / "proxy.py"), "serve", "--port", str(PROXY_PORT)],
            env=env,
            stderr=subprocess.PIPE,
        ),
    ]
    checks = Checks()
    try:
        _wait_healthy(SESSION_PORT)
        _wait_healthy(PROXY_PORT)
        print("\n[1] happy path -- agent loop through the proxy")
        token = proxy_mod.sign_token(
            {"sid": SID, "ip": "127.0.0.1", "port": SESSION_PORT, "exp": time.time() + 600}, secret
        )
        base_url = f"http://127.0.0.1:{PROXY_PORT}/s/{SID}/v1"
        result = asyncio.run(
            agent_mod.run_episode(
                {
                    "base_url": base_url,
                    "token": token,
                    "prompt": [{"role": "user", "content": "what is 17 * 23 + 4?"}],
                    "sampling_params": {"temperature": 0.7, "max_tokens": 512},
                    "instance_id": "smoke-task-1",
                }
            )
        )
        print(f"      agent result: {json.dumps(result, sort_keys=True)}")
        checks.expect("exit_status", result["exit_status"], "submitted")
        checks.expect("submitted answer", result["submitted_answer"], "395")
        checks.expect("tool_calls executed", result["agent_metrics"]["tool_calls"], 2)
        checks.expect("calculator used", result["agent_metrics"]["calculator_calls"], 1)
        checks.expect("turns", result["agent_metrics"]["turns"], 2)

        print("\n[2] streaming passes through")
        status, body = _post(
            f"{base_url}/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            token,
        )
        checks.expect("stream status", status, 200)
        checks.expect_true("SSE framing preserved", body.startswith("data: ") and "[DONE]" in body)

        print("\n[3] auth rejections")
        url = f"{base_url}/chat/completions"
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        checks.expect("no token", _post(url, payload, None)[0], 401)
        checks.expect("garbage token", _post(url, payload, "not.atoken")[0], 403)
        forged = proxy_mod.sign_token(
            {"sid": SID, "ip": "127.0.0.1", "port": SESSION_PORT, "exp": time.time() + 600},
            "wrong-secret-wrong-secret",
        )
        checks.expect("wrong secret", _post(url, payload, forged)[0], 403)
        expired = proxy_mod.sign_token(
            {"sid": SID, "ip": "127.0.0.1", "port": SESSION_PORT, "exp": time.time() - 5}, secret
        )
        checks.expect("expired token", _post(url, payload, expired)[0], 403)
        other = proxy_mod.sign_token(
            {"sid": "f" * 32, "ip": "127.0.0.1", "port": SESSION_PORT, "exp": time.time() + 600}, secret
        )
        checks.expect("token bound to another session", _post(url, payload, other)[0], 403)

        print("\n[4] path whitelist -- these must never reach the session server")
        root = f"http://127.0.0.1:{PROXY_PORT}"
        checks.expect("POST /sessions (mint)", _post(f"{root}/sessions", {}, token)[0], 404)
        checks.expect("catch-all flush_cache", _post(f"{root}/s/{SID}/flush_cache", {}, token)[0], 404)
        checks.expect("catch-all update_weights", _post(f"{root}/s/{SID}/v1/../update_weights", {}, token)[0], 404)
        checks.expect("samples op", _post(f"{root}/s/{SID}/samples", {}, token)[0], 404)
    finally:
        stderr_blobs = []
        for proc in procs:
            proc.terminate()
            try:
                _, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, err = proc.communicate()
            stderr_blobs.append((err or b"").decode())

    print("\n[5] no BREACH markers in the session server log")
    breaches = [line for blob in stderr_blobs for line in blob.splitlines() if "BREACH" in line]
    checks.expect_true("session server never served a dangerous route", not breaches, "; ".join(breaches))

    print()
    if checks.failures:
        print(f"FAILED: {len(checks.failures)} check(s): {', '.join(checks.failures)}")
        return 1
    print("ALL CHECKS PASSED -- the proxy leg is wired correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
