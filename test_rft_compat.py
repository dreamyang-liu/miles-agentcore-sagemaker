"""Local HTTP contract checks using the RFT SDK from the agent image.

Run: python -m unittest -v test_rft_compat

Requires fastapi, uvicorn, httpx, boto3, requests, and the image's sagemaker-core /
sagemaker-train wheels (see README). No AWS calls, model, or GPU are used. Only token
generation and InvokeAgentRuntime are replaced; the SDK sends its real HTTP callbacks.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
import unittest
import uuid
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sagemaker.train.rft import RolloutFeedbackClient, sagemaker_rft_handler
from sagemaker.train.rft.headers import get_inference_headers

import rft_agent_function
from rft_front_door import TRAJECTORY_HEADER, make_app

SSE = b'data: {"choices":[{"delta":{"content":"395"}}]}\n\ndata: [DONE]\n\n'


@contextmanager
def serve(app: FastAPI):
    """Serve on an already-bound loopback socket, avoiding fixed-port collisions."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("local contract-test server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("local contract-test server did not stop")


class RFTCompatibilityTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.upstream_calls = []
        self.upstream_status = 200
        self.events = []
        self.reject_registration = False
        backend = FastAPI()

        @backend.post("/sessions/{sid}/v1/chat/completions")
        async def completion(sid: str, request: Request):
            body = await request.json()
            self.upstream_calls.append((sid, body))
            if self.upstream_status != 200:
                return JSONResponse({"error": "model unavailable"}, status_code=self.upstream_status)
            if body.get("stream"):
                return Response(SSE, media_type="text/event-stream")
            return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "395"}}]})

        @backend.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
        async def unexpected_upstream(path: str):
            self.upstream_calls.append(("UNEXPECTED", path))
            return JSONResponse({"error": "unexpected backend operation"}, status_code=500)

        self.backend_url = stack.enter_context(serve(backend))
        front = make_app()

        @front.middleware("http")
        async def observe(request: Request, call_next):
            if self.reject_registration and request.url.path == "/miles/register":
                response = JSONResponse({"error": "registration unavailable"}, status_code=503)
            else:
                response = await call_next(request)
            self.events.append((request.method, request.url.path, response.status_code))
            return response

        self.front_url = stack.enter_context(serve(front))
        self.client = stack.enter_context(httpx.Client(base_url=self.front_url, timeout=5, trust_env=False))
        stack.enter_context(patch.dict(os.environ, {
            "AGENTCORE_RUNTIME_ARN": "contract-test-runtime",
            "MILES_RFT_FRONT_DOOR_URL": self.front_url,
            "MILES_RFT_FRONT_DOOR_LOCAL": self.front_url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }))
        stack.enter_context(patch("sagemaker.train.rft.feedback.generate_token", return_value="contract-test-token"))
        self.sid = uuid.uuid4().hex
        self.metadata = {
            "job_arn": "contract-test-job",
            "trajectory_id": self.sid,
            "endpoint": self.front_url,
            "region": "us-west-2",
        }

    def register(self, sid=None):
        sid = sid or self.sid
        self.client.post("/miles/register", json={
            "trajectory_id": sid,
            "session_url": f"{self.backend_url}/sessions/{sid}",
        }).raise_for_status()

    def diagnostics(self, sid=None):
        response = self.client.get(f"/miles/result/{sid or self.sid}")
        response.raise_for_status()
        return response.json()

    def callbacks(self):
        return [(path, status) for method, path, status in self.events
                if method == "POST" and path in ("/complete-rollout", "/update-reward")]

    def inference(self, sid=None, *, stream=False, headers=None):
        return self.client.post("/v1/chat/completions", headers=headers or {
            TRAJECTORY_HEADER: sid or self.sid,
        }, json={
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": "17 * 23 + 4"}],
            "stream": stream,
            "stream_options": {"include_usage": True},
            "temperature": 0.7,
        })

    def test_sdk_lifecycle_through_miles_caller(self):
        """Use the real decorator, header context, and feedback HTTP from one invocation."""
        @sagemaker_rft_handler
        def agent(payload):
            record = json.loads(payload["prompt"])
            self.assertEqual(record["instance"], "17 * 23 + 4")
            self.assertEqual(record["reward_spec"]["ground_truth"], "395")
            self.assertEqual(payload["inferenceParams"], {"temperature": 0.7, "max_tokens": 64})
            headers = get_inference_headers()
            self.assertEqual(headers["X-Amzn-SageMaker-Trajectory-Id"], self.sid)
            response = self.inference(stream=True, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, SSE)
            return {
                "status": "success",
                "agent_answer": "395",
                "reward": [0.2, 1.0],
                "trajectory_reward": -42.0,
                "per_turn_rewards": [0.2, 1.0],
            }

        async def invoke(arn, session_id, body, timeout):
            self.assertEqual(session_id, "miles-rft-" + self.sid)
            result = await asyncio.to_thread(agent, json.loads(body))
            # SDK "success" alone is insufficient: the decorator swallows feedback errors.
            self.assertEqual(self.callbacks(), [("/complete-rollout", 200), ("/update-reward", 200)])
            self.assertEqual(self.diagnostics(), {"status": "ready", "rewards": [0.2, 1.0]})
            # Completing the SDK lifecycle must leave Miles' session usable.
            self.assertEqual(self.inference().status_code, 200)
            return result

        with patch.object(rft_agent_function, "_invoke_with_backoff", side_effect=invoke):
            result = asyncio.run(rft_agent_function.run(
                f"{self.backend_url}/sessions/{self.sid}",
                [{"role": "user", "content": "17 * 23 + 4"}],
                request_kwargs={"temperature": 0.7, "max_tokens": 64},
                metadata={"answer": "395"},
            ))

        self.assertEqual(result["submitted_answer"], "395")
        self.assertEqual(result["exit_status"], "submitted")
        self.assertEqual(result["agent_metrics"]["rft_reward"], -42.0)
        self.assertNotIn("reward", result)  # RFT reward is diagnostic, not Miles' training reward.
        self.assertEqual(self.diagnostics(), {})
        self.assertEqual(self.inference().status_code, 404)
        self.assertEqual([sid for sid, _ in self.upstream_calls], [self.sid, self.sid])

    def test_sdk_scalar_list_and_duplicate_feedback_only_acknowledged(self):
        self.register()
        feedback = RolloutFeedbackClient(self.metadata)
        feedback.report_complete(1.0)
        self.assertEqual(self.diagnostics(), {"status": "ready", "rewards": [1.0]})
        feedback.report_complete([0.2, 0.8])
        feedback.report_complete([0.2, 0.8])
        self.assertEqual(self.diagnostics(), {"status": "ready", "rewards": [0.2, 0.8]})
        self.assertEqual(self.upstream_calls, [])
        self.assertTrue(all(status == 200 for _, status in self.callbacks()))
        self.assertEqual(len(self.callbacks()), 6)
        self.assertEqual(self.inference().status_code, 200)

    def test_sdk_error_feedback_does_not_close_miles_session(self):
        self.register()

        @sagemaker_rft_handler
        def agent(payload):
            return {"status": "error", "error": "synthetic agent failure", "reward": 0.0}

        with self.assertLogs("sagemaker.train.rft", level="WARNING"):
            result = agent({"metadata": self.metadata})
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.callbacks(), [("/complete-rollout", 200), ("/update-reward", 200)])
        self.assertEqual(self.diagnostics(), {"status": "failed", "rewards": [0.0]})
        self.assertEqual(self.upstream_calls, [])
        self.assertEqual(self.inference().status_code, 200)

    def test_sdk_handler_without_reward_only_completes(self):
        self.register()

        @sagemaker_rft_handler
        def agent(payload):
            return {"status": "success", "agent_answer": "395"}

        agent({"metadata": self.metadata})
        self.assertEqual(self.callbacks(), [("/complete-rollout", 200)])
        self.assertEqual(self.diagnostics(), {"status": "ready"})
        self.assertEqual(self.upstream_calls, [])

    def test_late_feedback_acknowledged_without_retaining_finished_trajectory(self):
        self.register()
        feedback = RolloutFeedbackClient(self.metadata)
        feedback.report_complete([0.2, 1.0])
        self.client.delete(f"/miles/register/{self.sid}").raise_for_status()
        self.assertEqual(self.diagnostics(), {})
        feedback.report_complete([0.2, 1.0])
        self.assertEqual(self.callbacks(), [
            ("/complete-rollout", 200), ("/update-reward", 200),
            ("/complete-rollout", 200), ("/update-reward", 200),
        ])
        self.assertEqual(self.diagnostics(), {})
        self.assertEqual(self.upstream_calls, [])
        self.assertEqual(self.inference().status_code, 404)

    def test_unregistered_feedback_acknowledged_without_creating_session(self):
        RolloutFeedbackClient(self.metadata).report_complete(0.0)
        self.assertEqual(self.callbacks(), [("/complete-rollout", 200), ("/update-reward", 200)])
        self.assertEqual(self.diagnostics(), {})
        self.assertEqual(self.client.get("/health").json()["sessions"], 0)
        self.assertEqual(self.upstream_calls, [])

    def test_inference_routes_each_trajectory_and_normalizes_stream_options(self):
        other = uuid.uuid4().hex
        self.register()
        self.register(other)
        response = self.inference(other)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "395")
        response = self.inference(stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, SSE)
        self.assertEqual([sid for sid, _ in self.upstream_calls], [other, self.sid])
        for _, body in self.upstream_calls:
            self.assertNotIn("stream_options", body)
            self.assertEqual(body["model"], "model")
            self.assertEqual(body["temperature"], 0.7)
            self.assertEqual(body["messages"], [{"role": "user", "content": "17 * 23 + 4"}])

    def test_provider_model_id_maps_to_configured_policy_and_preserves_lora_path(self):
        with serve(make_app(policy_model="local-policy")) as url:
            with httpx.Client(base_url=url, timeout=5, trust_env=False) as client:
                client.post("/miles/register", json={
                    "trajectory_id": self.sid,
                    "session_url": f"{self.backend_url}/sessions/{self.sid}",
                }).raise_for_status()
                response = client.post("/v1/chat/completions", headers={TRAJECTORY_HEADER: self.sid}, json={
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [{"role": "user", "content": "hi"}],
                    "lora_path": "miles-owned-adapter",
                })
                self.assertEqual(response.status_code, 200)
                body = self.upstream_calls[-1][1]
                self.assertEqual(body["model"], "local-policy")
                self.assertEqual(body["lora_path"], "miles-owned-adapter")

    def test_text_blocks_preserve_question_history_and_tool_fields(self):
        self.register()
        question = "  Natalia sold 48 clips.\nHow many after selling 24 more?  "
        tool_calls = [{
            "id": "calculator-1", "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression":"48+24"}'},
        }]
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": question[:25]},
                {"type": "text", "text": question[25:]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Let me calculate."}],
             "tool_calls": tool_calls, "reasoning_content": "Preserve this field."},
            {"role": "tool", "content": [{"type": "text", "text": "72"}],
             "tool_call_id": "calculator-1"},
        ]
        response = self.client.post("/v1/chat/completions", headers={TRAJECTORY_HEADER: self.sid},
                                    json={"model": "sdk-model", "messages": messages})
        self.assertEqual(response.status_code, 200)
        forwarded = self.upstream_calls[-1][1]["messages"]
        self.assertEqual(forwarded[0]["content"], question)
        self.assertEqual(forwarded[1]["content"], "Let me calculate.")
        self.assertEqual(forwarded[1]["tool_calls"], tool_calls)
        self.assertEqual(forwarded[1]["reasoning_content"], "Preserve this field.")
        self.assertEqual(forwarded[2]["content"], "72")
        self.assertEqual(forwarded[2]["tool_call_id"], "calculator-1")

    def test_nontext_content_blocks_are_not_discarded(self):
        self.register()
        blocks = [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://image.invalid/example.png"}},
        ]
        response = self.client.post("/v1/chat/completions", headers={TRAJECTORY_HEADER: self.sid},
                                    json={"model": "sdk-model", "messages": [{"role": "user", "content": blocks}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.upstream_calls[-1][1]["messages"][0]["content"], blocks)

    def test_inference_errors_are_not_compatibility_acknowledgements(self):
        self.assertEqual(self.inference().status_code, 404)
        self.register()
        self.upstream_status = 503
        response = self.inference(stream=True)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"error": "model unavailable"})
        self.assertEqual(self.client.post("/update_weights", json={}).status_code, 404)
        self.assertEqual(self.client.post("/sessions", json={}).status_code, 404)
        self.assertEqual(len(self.upstream_calls), 1)

    def test_registration_failure_prevents_runtime_invocation(self):
        self.reject_registration = True
        invoke = AsyncMock(return_value={"status": "success", "agent_answer": "395"})
        with patch.object(rft_agent_function, "_invoke_with_backoff", invoke):
            with self.assertRaises(httpx.HTTPStatusError):
                asyncio.run(rft_agent_function.run(
                    f"{self.backend_url}/sessions/{self.sid}",
                    [{"role": "user", "content": "17 * 23 + 4"}],
                    metadata={"answer": "395"},
                ))
        invoke.assert_not_called()


class PublicListenerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_peer_can_report_but_cannot_register_sessions(self):
        app = make_app(allowed_peers={"192.0.2.10"})
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, client=("192.0.2.10", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://front") as client:
                response = await client.post("/complete-rollout", json={"TrajectoryId": "finished", "Status": "ready"})
                self.assertEqual(response.status_code, 200)
                response = await client.post("/miles/register", json={
                    "trajectory_id": "injected", "session_url": "http://backend/sessions/injected",
                })
                self.assertEqual(response.status_code, 403)
                self.assertEqual((await client.get("/miles/result/finished")).status_code, 403)

    async def test_other_public_peers_are_rejected_but_local_control_works(self):
        app = make_app(allowed_peers={"192.0.2.10"})
        async with app.router.lifespan_context(app):
            public = httpx.ASGITransport(app=app, client=("192.0.2.11", 12345))
            async with httpx.AsyncClient(transport=public, base_url="http://front") as client:
                self.assertEqual((await client.get("/health")).status_code, 403)
            local = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(transport=local, base_url="http://front") as client:
                response = await client.post("/miles/register", json={
                    "trajectory_id": "local", "session_url": "http://backend/sessions/local",
                })
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
