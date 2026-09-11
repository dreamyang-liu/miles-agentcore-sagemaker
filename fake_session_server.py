"""A stand-in for the Miles session server, for wiring up AgentCore before touching GPUs.

Speaks enough of the real contract that an unmodified agent cannot tell the difference:
OpenAI chat completions, streaming or not, driving a two-tool-call episode before it
answers. It does no TITO work -- the point is to prove the network path, not the training
path, so this runs anywhere with no model and no GPU.

It also *deliberately* reproduces the real server's two dangerous behaviours, so a
misconfigured proxy fails loudly instead of silently:

- ``POST /sessions`` mints a session with no authentication at all.
- ``/sessions/{sid}/{anything}`` forwards to the SGLang router, which is a direct line to
  the engine's control plane.

Both log at ERROR with a BREACH marker. If either fires while traffic is coming through
the proxy, the path whitelist is not doing its job.

Usage:
    python fake_session_server.py --host 0.0.0.0 --port 30007

Optional --rft uses the RFT agent's finish tool. Supplying the four --verify-* options
also renders requests with the real tokenizer/template and rejects a missing question.
That check needs transformers and local model files, but still uses no model or GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("fake-session-server")

# A scripted policy: compute, then submit. Exercises both tools the math agent exposes,
# plus the submit-and-stop path, without needing a model.
_SCRIPT = [
    {"name": "calculator", "arguments": '{"expression":"17 * 23 + 4"}'},
    {"name": "submit_answer", "arguments": '{"answer":"395"}'},
]
SCRIPTED_ANSWER = "395"


def _completion(model: str, turn: int, *, rft: bool = False) -> dict:
    """Build turn `turn`'s reply: follow the script, then stop."""
    if turn < len(_SCRIPT):
        call = dict(_SCRIPT[turn])
        if rft and call["name"] == "submit_answer":
            call["name"] = "finish"
        message = {
            "role": "assistant",
            # Real SGLang returns "" rather than null when a tool-call parser fires;
            # the session server rejects null, so mirror the non-null contract here.
            "content": "",
            "tool_calls": [{"id": f"call_{turn}", "type": "function", "function": call}],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": f"Finished after {turn} tool calls."}
        finish_reason = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 128 + turn * 64, "completion_tokens": 32, "total_tokens": 160 + turn * 64},
    }


def _as_sse(completion: dict) -> bytes:
    """Re-render a finished completion as one chunk plus [DONE], like the real server's fake stream."""
    choice = completion["choices"][0]
    message = choice["message"]
    delta = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(message["tool_calls"])]
    chunk = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}],
        "usage": completion["usage"],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"


def make_app(
    *,
    rft: bool = False,
    verify_model_path: str | None = None,
    verify_chat_template: str | None = None,
    verify_expected_text: str | None = None,
    verify_evidence_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title="fake-session-server")
    state: dict[str, int] = {}
    verification = (verify_model_path, verify_chat_template, verify_expected_text, verify_evidence_dir)
    if any(verification) and not all(verification):
        raise ValueError("all four prompt verification settings must be supplied together")
    tokenizer = None
    if verify_model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(verify_model_path, local_files_only=True)
        tokenizer.chat_template = Path(verify_chat_template).read_text()
        Path(verify_evidence_dir).mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "sessions": len(state)})

    @app.post("/sessions")
    async def create_session():
        sid = uuid.uuid4().hex
        state[sid] = 0
        # Unauthenticated by design in the real server. Reachable from outside = anyone can
        # mint sessions and inject fabricated trajectories.
        logger.error("BREACH? unauthenticated POST /sessions served -> %s", sid)
        return JSONResponse({"session_id": sid})

    @app.post("/sessions/{sid}/v1/chat/completions")
    async def chat_completions(request: Request, sid: str):
        body = json.loads(await request.body() or b"{}")
        messages = body.get("messages", [])
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if tokenizer is not None:
            tokens = tokenizer.apply_chat_template(
                messages, tools=body.get("tools"), tokenize=True,
                add_generation_prompt=True, return_dict=False,
            )
            rendered = tokenizer.decode(tokens, skip_special_tokens=False)
            # These recipes use Qwen's ChatML user delimiters. Checking the user turn
            # avoids accepting a question that appears only in a system example.
            user_turns = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rendered, re.DOTALL)
            present = any(verify_expected_text in text for text in user_turns)
            evidence = {
                "session_id": sid,
                "turn": turn,
                "expected_text": verify_expected_text,
                "question_present": present,
                "prompt_tokens": len(tokens),
                "rendered_prompt": rendered,
                "rendered_user_turns": user_turns,
                "message_content_types": [type(m.get("content")).__name__ for m in messages],
            }
            # sid is a single route component; the suffix also prevents "."/".." traversal.
            with (Path(verify_evidence_dir) / f"{sid}.jsonl").open("a") as handle:
                handle.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            logger.info("PROMPT_FIDELITY sid=%s turn=%s question_present=%s", sid, turn, present)
            if not present:
                return JSONResponse(
                    {"error": "source question is missing from the rendered model prompt"}, status_code=400
                )
        state[sid] = turn + 1
        logger.info("sid=%s turn=%s messages=%s stream=%s", sid, turn, len(messages), bool(body.get("stream")))
        completion = _completion(body.get("model", "fake-model"), turn, rft=rft)
        if body.get("stream"):
            return Response(
                content=_as_sse(completion),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )
        return JSONResponse(completion)

    @app.post("/sessions/{sid}/samples")
    async def collect_samples(sid: str):
        # The real one returns safetensors; the shape is irrelevant to a connectivity test.
        return JSONResponse({"note": "stub", "turns_recorded": state.get(sid, 0)})

    @app.delete("/sessions/{sid}")
    async def delete_session(sid: str):
        state.pop(sid, None)
        return Response(status_code=204)

    # Registered last, exactly like the real server: any path at all reaches the router,
    # and the real one does not even check that the session exists.
    @app.api_route("/sessions/{sid}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(sid: str, path: str):
        logger.error("BREACH? catch-all reached: /sessions/%s/%s would hit the SGLang router", sid, path)
        return JSONResponse({"breach": True, "path": path}, status_code=200)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30007)
    parser.add_argument("--rft", action="store_true", help="use the RFT agent's finish tool")
    parser.add_argument("--verify-model-path", help="local tokenizer/model directory")
    parser.add_argument("--verify-chat-template", help="exact training chat-template file")
    parser.add_argument("--verify-expected-text", help="source question that must survive rendering")
    parser.add_argument("--verify-evidence-dir", help="directory for per-session rendered-prompt evidence")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("fake session server on %s:%s (scripted answer %s)", args.host, args.port, SCRIPTED_ANSWER)
    app = make_app(
        rft=args.rft,
        verify_model_path=args.verify_model_path,
        verify_chat_template=args.verify_chat_template,
        verify_expected_text=args.verify_expected_text,
        verify_evidence_dir=args.verify_evidence_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
