"""Shared bits for the SageMaker launchers here: account constants, VPC config, log tailing.

Environment (all optional; the defaults are the reference account this was built in):
    AWS_REGION          region of the VPC, ECR repos and jobs         (default us-west-2)
    MILES_SM_ROLE_ARN   SageMaker execution role ARN                  (default: the reference role)
    MILES_SM_BUCKET     S3 bucket for model/data channels + outputs   (default: the reference bucket)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
ECR = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
ROLE = os.environ.get(
    "MILES_SM_ROLE_ARN", f"arn:aws:iam::{ACCOUNT}:role/service-role/AmazonSageMaker-ExecutionRole-20250421T134015"
)
BUCKET = os.environ.get("MILES_SM_BUCKET", "drmyang-sagemaker-us-west-2")
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
INFRA = json.loads(Path(__file__).with_name(".infra.json").read_text())

sm = boto3.client("sagemaker", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)


def vpc_config() -> dict:
    return {
        "Subnets": INFRA["private_subnet_ids"],
        "SecurityGroupIds": [INFRA["security_groups"]["miles-agentcore-train"]],
    }


def job_status(job: str) -> tuple[str, str]:
    d = sm.describe_training_job(TrainingJobName=job)
    return d["TrainingJobStatus"], d.get("SecondaryStatus", "")


def log_lines(job: str, seen: set[str]) -> list[str]:
    """Every not-yet-seen CloudWatch line of the job, prefixed with its host."""
    out = []
    try:
        streams = logs.describe_log_streams(logGroupName=LOG_GROUP, logStreamNamePrefix=f"{job}/")["logStreams"]
    except logs.exceptions.ResourceNotFoundException:
        return out
    for stream in streams:
        host = stream["logStreamName"].split("/")[1].rsplit("-", 1)[0]
        token = None
        while True:
            kwargs = {"logGroupName": LOG_GROUP, "logStreamName": stream["logStreamName"], "startFromHead": True}
            if token:
                kwargs["nextToken"] = token
            page = logs.get_log_events(**kwargs)
            for event in page["events"]:
                key = f"{stream['logStreamName']}:{event['timestamp']}:{event['message']}"
                if key not in seen:
                    seen.add(key)
                    out.append(f"[{host}] {event['message']}")
            if page["nextForwardToken"] == token:
                break
            token = page["nextForwardToken"]
    return out


def watch(job: str, markers: re.Pattern, *, stop_when: re.Pattern | None = None, interval: int = 20) -> None:
    """Print marker lines as they appear; return on a terminal status or when ``stop_when`` matches."""
    seen: set[str] = set()
    while True:
        status, secondary = job_status(job)
        hit_stop = False
        for line in log_lines(job, seen):
            if markers.search(line):
                print(line[:400], flush=True)
            if stop_when and stop_when.search(line):
                hit_stop = True
        print(f"-- {time.strftime('%H:%M:%S')} {status}/{secondary}", file=sys.stderr, flush=True)
        if status in ("Completed", "Failed", "Stopped") or hit_stop:
            if status == "Failed":
                print("FAILURE:", sm.describe_training_job(TrainingJobName=job).get("FailureReason"), flush=True)
            return
        time.sleep(interval)


def dump_logs(job: str) -> None:
    for line in log_lines(job, set()):
        print(line)


def stop(job: str) -> None:
    sm.stop_training_job(TrainingJobName=job)
    print("stop requested")
