"""VPC-mode Bedrock AgentCore runtime for the SageMaker recipe, and a direct-path invoker.

``create`` builds (or adopts, by name) a runtime attached to the private subnets of the
``infra.py`` VPC with the ``miles-agentcore-agentcore`` security group. AgentCore VPC mode
only accepts supported AZ IDs: check the current AgentCore VPC documentation before
creating a runtime. The helper filters with its known/overridden list, and if creation
ends ``CREATE_FAILED`` naming unsupported subnets, it retries without them. The
us-west-2 defaults were learned from an earlier deployment.

``invoke`` sends the payload the agent expects, with ``base_url`` pointing straight at the
SageMaker head's VPC IP -- no proxy, no HMAC. Use it against the network smoke
(``launch_smoke.py``) to prove the path before spending GPU time.

Usage:
    python agentcore_runtime.py create --image-uri <ecr>/miles-agentcore-math:latest
    python agentcore_runtime.py invoke --head-ip 10.20.4.84 [--port 30000]
    python agentcore_runtime.py invoke-rft --front-door-url http://miles-head.miles.internal:30100 [--trajectory-id smoke-traj]

Environment (optional):
    AWS_REGION                 default us-west-2
    MILES_AGENTCORE_AZ_IDS     comma-separated AZ IDs known to be supported (default: the us-west-2 set)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
# Learned empirically on 2026-09-04 from a CREATE_FAILED runtime: "Supported availability
# zones are: usw2-az2, usw2-az1, usw2-az3". Other regions: leave unset, the create loop learns.
_KNOWN_AZ_IDS = {"us-west-2": "usw2-az1,usw2-az2,usw2-az3"}
AGENTCORE_AZ_IDS = set(filter(None, os.environ.get("MILES_AGENTCORE_AZ_IDS", _KNOWN_AZ_IDS.get(REGION, "")).split(",")))
DEFAULT_RUNTIME_NAME = "miles_math_agent_vpc"
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
INFRA = json.loads(Path(__file__).with_name(f".infra.{ACCOUNT}.json").read_text())
STATE = Path(__file__).with_name(f".agentcore_runtime.{ACCOUNT}.json")

control = boto3.client("bedrock-agentcore-control", region_name=REGION)
data = boto3.client("bedrock-agentcore", region_name=REGION)


def _find_runtime(name: str) -> dict | None:
    for rt in control.list_agent_runtimes()["agentRuntimes"]:
        if rt["agentRuntimeName"] == name:
            return control.get_agent_runtime(agentRuntimeId=rt["agentRuntimeId"])
    return None


def _wait_settled(runtime_id: str, timeout: int = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rt = control.get_agent_runtime(agentRuntimeId=runtime_id)
        if rt["status"] not in ("CREATING", "UPDATING"):
            return rt
        time.sleep(10)
    raise TimeoutError(f"runtime {runtime_id} still not settled after {timeout}s")


def _delete_runtime(runtime_id: str, name: str) -> None:
    control.delete_agent_runtime(agentRuntimeId=runtime_id)
    while _find_runtime(name):
        time.sleep(5)


def _candidate_subnets() -> list[str]:
    by_id = {s["id"]: s for s in INFRA["subnets"]}
    subnets = INFRA["private_subnet_ids"]
    if AGENTCORE_AZ_IDS:
        subnets = [s for s in subnets if by_id[s]["az_id"] in AGENTCORE_AZ_IDS]
    return subnets


def _unsupported_subnets(failure_reason: str) -> set[str]:
    """Subnet ids named in a ``CREATE_FAILED`` reason about unsupported availability zones."""
    if "unsupported availability zone" not in failure_reason:
        return set()
    return set(re.findall(r"subnet-[0-9a-f]+", failure_reason))


def create(args: argparse.Namespace) -> None:
    role_arn = args.role_arn or INFRA.get("agentcore_role_arn")
    if not role_arn:
        sys.exit("--role-arn is required (or run infra.py first, which records agentcore_role_arn)")
    artifact = {"containerConfiguration": {"containerUri": args.image_uri}}
    sg = INFRA["security_groups"]["miles-agentcore-agentcore"]
    subnets = _candidate_subnets()
    env = dict(kv.split("=", 1) for kv in args.env)
    extra = {"environmentVariables": env} if env else {}

    existing = _find_runtime(args.name)
    if existing and existing["status"].endswith("_FAILED"):
        _delete_runtime(existing["agentRuntimeId"], args.name)  # cannot be repaired in place
        existing = None

    for attempt in range(1, 4):
        print(f"attempt {attempt}: subnets {subnets}", file=sys.stderr)
        network = {"networkMode": "VPC", "networkModeConfig": {"subnets": subnets, "securityGroups": [sg]}}
        if existing:
            resp = control.update_agent_runtime(
                agentRuntimeId=existing["agentRuntimeId"],
                agentRuntimeArtifact=artifact,
                roleArn=role_arn,
                networkConfiguration=network,
                **extra,
            )
        else:
            resp = control.create_agent_runtime(
                agentRuntimeName=args.name,
                agentRuntimeArtifact=artifact,
                roleArn=role_arn,
                networkConfiguration=network,
                description="Miles x SageMaker: VPC mode, direct to the session servers",
                **extra,
            )
        rt = _wait_settled(resp["agentRuntimeId"])
        if rt["status"] == "READY":
            break
        reason = rt.get("failureReason", "")
        print(f"attempt {attempt}: {rt['status']}: {reason}", file=sys.stderr)
        bad = _unsupported_subnets(reason)
        _delete_runtime(rt["agentRuntimeId"], args.name)
        existing = None
        if not bad or not (subnets := [s for s in subnets if s not in bad]):
            sys.exit(f"runtime creation failed: {reason}")
    else:
        sys.exit("gave up creating the runtime")

    summary = {
        "agentRuntimeName": args.name,
        "environmentVariables": env,
        "agentRuntimeId": rt["agentRuntimeId"],
        "agentRuntimeArn": rt["agentRuntimeArn"],
        "status": rt["status"],
        "networkConfiguration": rt["networkConfiguration"],
        "containerUri": args.image_uri,
        "roleArn": role_arn,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    STATE.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def invoke(args: argparse.Namespace) -> None:
    arn = json.loads(STATE.read_text())["agentRuntimeArn"]
    sid = uuid.uuid4().hex
    payload = {
        # Direct path: the head's VPC IP, the recipe's session URL shape, no proxy in between.
        "base_url": f"http://{args.head_ip}:{args.port}/sessions/{sid}/v1",
        "token": "smoke-no-auth",
        "prompt": [{"role": "user", "content": "what is 17 * 23 + 4?"}],
        "sampling_params": {"temperature": 1.0, "max_tokens": 256},
        "instance_id": f"net-smoke-{sid[:8]}",
    }
    print(f"sid={sid} base_url={payload['base_url']}", file=sys.stderr)
    started = time.monotonic()
    resp = data.invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=f"miles-{sid}",  # AgentCore wants >= 33 chars; uuid4().hex is 32
        payload=json.dumps(payload).encode(),
        contentType="application/json",
        accept="application/json",
    )
    body = resp["response"].read()
    print(f"elapsed={time.monotonic() - started:.1f}s status={resp['statusCode']}", file=sys.stderr)
    print(body.decode(errors="replace"))


def invoke_rft(args: argparse.Namespace) -> None:
    """Drive an RFT-contract agent: the trajectory id must already be registered at the front door."""
    arn = args.runtime_arn or json.loads(STATE.read_text())["agentRuntimeArn"]
    record = {
        "instance_id": "smoke-0",
        "data_source": "gsm8k",
        "instance": "What is 17 * 23 + 4?",
        "prompt": [{"role": "user", "content": "What is 17 * 23 + 4?"}],
        "reward_spec": {"ground_truth": "395"},
        "extra_info": {},
    }
    payload = {
        "prompt": json.dumps(record),
        "metadata": {
            "job_arn": "arn:aws:sagemaker:::training-job/miles-smoke",
            "trajectory_id": args.trajectory_id,
            "endpoint": args.front_door_url.rstrip("/"),
            "region": REGION,
        },
        "inferenceParams": {"temperature": 1.0, "max_tokens": 256},
    }
    started = time.monotonic()
    resp = data.invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=f"miles-rft-smoke-{uuid.uuid4().hex}",
        payload=json.dumps(payload).encode(),
        contentType="application/json",
        accept="application/json",
    )
    body = resp["response"].read()
    print(f"elapsed={time.monotonic() - started:.1f}s status={resp['statusCode']}", file=sys.stderr)
    print(body.decode(errors="replace")[:2000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--image-uri", required=True, help="ECR URI of the linux/arm64 agent image")
    c.add_argument("--role-arn", default=None, help="runtime execution role; default from .infra.json")
    c.add_argument("--name", default=DEFAULT_RUNTIME_NAME, help="agentRuntimeName (letters, digits, _)")
    c.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="runtime environment variable; repeatable")
    c.set_defaults(func=create)
    inv = sub.add_parser("invoke")
    inv.add_argument("--head-ip", required=True)
    inv.add_argument("--port", type=int, default=30000)
    inv.set_defaults(func=invoke)
    rft = sub.add_parser("invoke-rft")
    rft.add_argument("--front-door-url", required=True, help="what the agent resolves, e.g. http://miles-head.miles.internal:30100")
    rft.add_argument("--trajectory-id", default="smoke-traj")
    rft.add_argument("--runtime-arn", default=None, help="default: the account's .agentcore_runtime file")
    rft.set_defaults(func=invoke_rft)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
