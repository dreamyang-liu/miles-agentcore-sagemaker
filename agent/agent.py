"""AgentCore Runtime agent: a two-tool math solver driven by a Miles-served policy.

Implements the Runtime HTTP contract -- ``POST /invocations``, ``GET /ping``, port 8080 --
and nothing else. Miles hands over a per-trajectory session endpoint plus a signed token in
the invocation payload. Every model call uses that session URL, directly over the
VPC or through the optional legacy public proxy. Miles records exact token ids there.

Two tools are exposed to the model:

- ``calculator(expression)`` -- evaluates arithmetic with an AST walker, never ``eval``.
- ``submit_answer(answer)`` -- ends the episode with a final answer.

**This agent does not grade.** It reports the submitted answer and lets Miles score it via
``--custom-rm-path``, so the ground truth never leaves the training cluster and the reward
pathway stays where Miles expects it.

Payload contract (produced by the Miles-side agent function):

    {"base_url": "https://proxy.example.com/s/<sid>/v1",   # already includes /v1
     "token": "<hmac token>",
     "prompt": [{"role": "user", "content": "what is 17 * 23 + 4?"}],
     "sampling_params": {"temperature": 1.0, "max_tokens": 2048},
     "instance_id": "math-42"}

Reply contract (consumed by that agent function, then by the reward function):

    {"submitted_answer": "395", "exit_status": "submitted",
     "agent_metrics": {"turns": 3, "tool_calls": 2, "calculator_calls": 1,
                       "total_tool_time": 0.01}}
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from openai import APIError, AsyncOpenAI
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("miles-agentcore-math-agent")

# A runaway loop must end on its own: AgentCore bills wall-clock and the session has a hard
# maxLifetime, so never rely on the caller to stop us.
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "12"))

# Three things this prompt has to buy, each learned the hard way from a 50-rollout 0.6B run:
#   1. submit_answer is mandatory and is the ONLY scoring channel. An early version said
#      "do not call submit_answer before you have verified the arithmetic", which acted as a
#      brake: submit_rate sat at 0.03 and reward tracked it almost exactly.
#   2. Multi-turn, so the trajectory has several model calls to train on.
#   3. One tool call per response. The previous prompt already asked for "one call per turn"
#      and the policy ignored it -- by rollout 45 89.7% of trajectories emitted calculator
#      and submit_answer together, and only 1.2% ever read a tool result before answering.
#      A prompt cannot fix that alone, because it does not change the reward landscape: while
#      parallel-and-submit still scores, GRPO converges on it. Hence the reject_submit guard
#      in run_episode -- the prompt states the rule, the guard makes breaking it worthless.
SYSTEM_PROMPT = (
    "You are a math assistant that solves problems step by step over multiple turns.\n"
    "\n"
    "HARD RULE: exactly ONE tool call per response. Never emit two tool calls at once.\n"
    "Above all, never send submit_answer in the same response as a calculator call -- such a\n"
    "submit_answer is REJECTED, because you cannot have read that calculator's result yet.\n"
    "\n"
    "How to work:\n"
    "1. Call calculator for the first arithmetic step. Then stop and wait for the result.\n"
    "2. Read the result, then call calculator for the next step. One step per response.\n"
    "3. When a result is the final answer, call submit_answer on its own, with nothing else.\n"
    "\n"
    "submit_answer is the ONLY way to deliver your answer. Stating the answer in your reply "
    "does not count: if you stop without calling submit_answer, the task is scored as failed "
    "no matter how correct your reasoning was. Always finish with submit_answer, alone."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression and return its value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '17 * 23 + 4', 'sqrt(144)', '2**10 / 8'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer, numeric only."}},
                "required": ["answer"],
            },
        },
    },
]

# ---------------------------------------------------------------- calculator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "comb": math.comb,
    "perm": math.perm,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

# Sampling parameters the OpenAI SDK accepts as real keyword arguments. Miles filters its
# sampling params against SGLang's ChatCompletionRequest, which is a superset of OpenAI's --
# top_k, min_p, repetition_penalty, ignore_eos and friends are SGLang extensions. The SDK
# raises TypeError on any unknown kwarg, so everything outside this set has to travel in
# extra_body, which passes it through to SGLang untouched.
_OPENAI_NATIVE_SAMPLING = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "tool_choice",
        "top_logprobs",
        "top_p",
        "user",
    }
)

# Owned by this function, never taken from the caller's sampling params.
_RESERVED_SAMPLING = frozenset({"model", "messages", "tools", "stream", "stream_options"})


def split_sampling_params(sampling: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split sampling params into SDK keyword arguments and extra_body passthrough."""
    native, extra = {}, {}
    for key, value in sampling.items():
        if key in _RESERVED_SAMPLING:
            continue
        (native if key in _OPENAI_NATIVE_SAMPLING else extra)[key] = value
    return native, extra


# Bounds that keep a hostile expression from wedging the microVM. 2**10**9 would otherwise
# allocate until the kernel kills us, and factorial(1e6) is comparably slow.
_MAX_POW_EXPONENT = 4096
_MAX_FACTORIAL = 2048


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"unknown name: {node.id}")
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError(f"exponent too large (limit {_MAX_POW_EXPONENT})")
        return op(left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("only whitelisted math functions may be called")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        args = [_eval_node(a) for a in node.args]
        if node.func.id == "factorial" and (args and args[0] > _MAX_FACTORIAL):
            raise ValueError(f"factorial argument too large (limit {_MAX_FACTORIAL})")
        return _FUNCTIONS[node.func.id](*args)
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, returning the value or a readable error."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"error: could not parse {expression!r} ({exc.msg})"
    try:
        value = _eval_node(tree.body)
    except ZeroDivisionError:
        return "error: division by zero"
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return f"error: {exc}"
    # Render 395.0 as 395 so the model does not learn to submit a spurious decimal.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


# ---------------------------------------------------------------- agent loop


def _tool_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _with_system_prompt(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if prompt and prompt[0].get("role") == "system":
        return list(prompt)
    return [{"role": "system", "content": SYSTEM_PROMPT}, *prompt]


async def run_episode(payload: dict[str, Any]) -> dict[str, Any]:
    """Drive the tool-calling loop against the session endpoint Miles handed us."""
    client = AsyncOpenAI(
        base_url=payload["base_url"],
        api_key=payload["token"],
        # Long completions are normal; the proxy and session server allow ~600s upstream.
        timeout=float(os.environ.get("AGENT_REQUEST_TIMEOUT", "900")),
        max_retries=2,
    )
    native_sampling, extra_sampling = split_sampling_params(dict(payload.get("sampling_params") or {}))
    messages = _with_system_prompt(list(payload["prompt"]))

    submitted: str | None = None
    turns = calculator_calls = tool_calls = rejected_submits = 0
    tool_time = 0.0
    exit_status = "no_submission"

    for turn in range(MAX_TURNS):
        turns = turn + 1
        try:
            # Send the full history every turn: the session server reuses its deepest
            # checkpoint and only tokenizes the appended suffix.
            response = await client.chat.completions.create(
                model=payload.get("model", "model"),
                messages=messages,
                tools=TOOLS,
                # Currently inert on SGLang's qwen25 path (verified: the detector emits a
                # byte-identical structural_tag either way). Kept so the request is correct
                # if/when the engine honours it; the real guard is reject_submit below.
                parallel_tool_calls=False,
                **native_sampling,
                **({"extra_body": extra_sampling} if extra_sampling else {}),
            )
        except APIError as exc:
            logger.warning("model call failed on turn %s: %s", turns, exc)
            exit_status = f"model_error:{type(exc).__name__}"
            break

        message = response.choices[0].message
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        # Thinking models (Qwen3.6 and friends, template preserve_thinking=True) return the
        # chain of thought in a separate `reasoning_content` field, and the session server's
        # strict matcher compares it -- it is one of the four template-relevant keys. Dropping
        # it makes every replayed assistant message mismatch, so the session rolls back to the
        # empty checkpoint on EVERY turn: history grows unbounded (938+ messages, 80k-token
        # prefills, 5s -> 300s trials) and each turn's recorded tokens are discarded.
        # Seen 2026-09-10 on Qwen3.6-27B: 5312 rollbacks, all to `checkpoint -1`.
        if reasoning := getattr(message, "reasoning_content", None):
            assistant["reasoning_content"] = reasoning
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant)

        if not message.tool_calls:
            # No tool call and no submission: the model answered in prose and is done.
            exit_status = "stopped_without_submitting"
            break

        # A submit_answer that rides along in the same response as a calculator was decided
        # BEFORE the calculator returned, so the tool result cannot have informed it. Left
        # alone the policy converges on exactly that -- reward only checks the final answer,
        # so emitting every call in one parallel batch is the cheapest way to score, and
        # grounded tool use collapsed to 1.2% over a 50-rollout run.
        #
        # sampling's parallel_tool_calls=False does NOT prevent this: SGLang's qwen25 detector
        # ignores the flag (it produces a byte-identical structural_tag either way, and only
        # constrains at_least_one). So the invariant is enforced here instead.
        names = [tc.function.name for tc in message.tool_calls]
        reject_submit = "submit_answer" in names and len(message.tool_calls) > 1

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = _tool_arguments(tool_call.function.arguments)
            started = time.monotonic()
            if name == "calculator":
                result = calculate(str(args.get("expression", "")))
                calculator_calls += 1
            elif name == "submit_answer":
                if reject_submit:
                    rejected_submits += 1
                    result = (
                        "error: submit_answer was rejected because it was issued in the same turn as "
                        "another tool call, so you had not seen that tool's result yet. Read the "
                        "results above, then call submit_answer on its own in the next turn."
                    )
                else:
                    submitted = str(args.get("answer", "")).strip()
                    result = f"Answer recorded: {submitted}"
            else:
                result = f"error: unknown tool {name!r}"
            tool_time += time.monotonic() - started
            tool_calls += 1
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        if submitted is not None:
            exit_status = "submitted"
            break
    else:
        exit_status = "max_turns_exceeded"

    return {
        "submitted_answer": submitted,
        "exit_status": exit_status,
        "agent_metrics": {
            "turns": turns,
            "tool_calls": tool_calls,
            "calculator_calls": calculator_calls,
            # total_tool_time lands on Sample.non_generation_time, so Miles' throughput
            # accounting subtracts environment time from generation time.
            "total_tool_time": round(tool_time, 4),
            # Rises when the policy tries to submit without reading a tool result. Should
            # fall as it learns the turn discipline; a flat high value means it never did.
            "rejected_submits": rejected_submits,
        },
    }


# ---------------------------------------------------------------- runtime contract

app = FastAPI(title="miles-agentcore-math-agent")

# AgentCore's idle timer resets on each invocation, but a single long trial can outlast it,
# so report HealthyBusy while work is in flight. time_of_last_update must only move when the
# status actually changes -- advancing it on every ping stops the idle timeout from ever
# firing, and sessions then run to maxLifetime and eat the session quota.
_ping = {"in_flight": 0, "status": "Healthy", "changed_at": time.time()}


def _set_status(status: str) -> None:
    if status != _ping["status"]:
        _ping["status"] = status
        _ping["changed_at"] = time.time()


@app.get("/ping")
async def ping():
    return {"status": _ping["status"], "time_of_last_update": int(_ping["changed_at"])}


@app.post("/invocations")
async def invocations(request: Request):
    session_id = request.headers.get("x-amzn-bedrock-agentcore-runtime-session-id", "?")
    payload = await request.json()
    _ping["in_flight"] += 1
    _set_status("HealthyBusy")
    started = time.monotonic()
    logger.info("invocation session=%s instance=%s", session_id, payload.get("instance_id"))
    try:
        result = await run_episode(payload)
    except Exception as exc:
        # Never surface a 5xx: AgentCore turns it into 424 RuntimeClientError and the
        # trajectory is lost. Report the failure in-band instead.
        logger.exception("episode failed for session %s", session_id)
        result = {
            "submitted_answer": None,
            "exit_status": f"agent_error:{type(exc).__name__}",
            "agent_metrics": {},
            "error": str(exc),
        }
    finally:
        _ping["in_flight"] -= 1
        if _ping["in_flight"] == 0:
            _set_status("Healthy")
    result["agent_metrics"] = {**result.get("agent_metrics", {}), "total_time": round(time.monotonic() - started, 3)}
    logger.info(
        "invocation session=%s done: %s answer=%r", session_id, result["exit_status"], result.get("submitted_answer")
    )
    return JSONResponse(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Host/port are fixed by the Runtime contract.
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
