"""Stage 2: the real Miles recipe as a multi-instance SageMaker training job in the VPC.

Submits the image built from ``Dockerfile.train``; ``entrypoint.py`` inside it does the Ray
head/worker dance and runs ``run_qwen3_agentcore_math.py --agent-mode agentcore`` on the head.
The AgentCore runtime (VPC mode, see ``agentcore_runtime.py``) connects straight to the head's
session servers.

Usage:
    python launch_train.py start [--mode smoke] [--instance-type ml.g6e.12xlarge] [--count 2]
    python launch_train.py watch <job-name>
    python launch_train.py logs  <job-name>
    python launch_train.py stop  <job-name>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import sm_common as C

# Built from Dockerfile.train; override to point at another repo/tag.
IMAGE = os.environ.get("MILES_TRAIN_IMAGE", f"{C.ECR}/miles:sagemaker-agentcore")
RUNTIME = json.loads(Path(__file__).with_name(".agentcore_runtime.json").read_text())

_MARKERS = re.compile(
    r"(HOST_REPORT|HEAD_READY|RAY_NODES|WORKER_RESOLVED|WORKER_JOINED|WORKER_DONE|LAUNCHER_EXIT|DONE role"
    r"|Session servers launched|trial done|rollout/raw_reward|agent metrics for rollout"
    r"|Traceback|Error|error:|pidfd|OutOfMemory|CUDA out of memory|Killed)"
)


def start(args: argparse.Namespace) -> None:
    name = f"miles-train-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    s3 = f"s3://{C.BUCKET}/miles"
    C.sm.create_training_job(
        TrainingJobName=name,
        RoleArn=C.ROLE,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": args.count, "VolumeSizeInGB": 200},
        VpcConfig=C.vpc_config(),
        InputDataConfig=[
            {
                "ChannelName": "model",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"{s3}/models/{args.model_name}/", "S3DataDistributionType": "FullyReplicated"}},
                "InputMode": "File",
            },
            {
                "ChannelName": "data",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"{s3}/data/", "S3DataDistributionType": "FullyReplicated"}},
                "InputMode": "File",
            },
        ],
        OutputDataConfig={"S3OutputPath": f"{s3}/output/"},
        CheckpointConfig={"S3Uri": f"{s3}/checkpoints/{name}/", "LocalPath": "/opt/ml/checkpoints"},
        StoppingCondition={"MaxRuntimeInSeconds": args.max_runtime},
        Environment={
            "MILES_SM_MODE": args.mode,
            "MILES_SM_MODEL_NAME": args.model_name,
            "MILES_SM_DATASET": args.dataset,
            "MILES_SM_EXTRA_ARGS": args.extra_args,
            "AGENTCORE_RUNTIME_ARN": RUNTIME["agentRuntimeArn"],
            "AWS_REGION": C.REGION,
            "AGENTCORE_MAX_CONCURRENT": str(args.agentcore_max_concurrent),
        },
        Tags=[{"Key": "project", "Value": "miles-agentcore"}],
    )
    print(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start")
    s.add_argument("--mode", choices=["smoke", "normal"], default="smoke")
    s.add_argument("--instance-type", default="ml.g6e.12xlarge")
    s.add_argument("--count", type=int, default=2)
    s.add_argument("--model-name", default="Qwen3-0.6B")
    s.add_argument("--dataset", default="gsm-hard")
    s.add_argument("--extra-args", default="")
    s.add_argument("--agentcore-max-concurrent", type=int, default=8)
    s.add_argument("--max-runtime", type=int, default=3 * 3600)
    s.set_defaults(func=start)
    w = sub.add_parser("watch")
    w.add_argument("job")
    w.set_defaults(func=lambda a: C.watch(a.job, _MARKERS))
    l = sub.add_parser("logs")
    l.add_argument("job")
    l.set_defaults(func=lambda a: C.dump_logs(a.job))
    st = sub.add_parser("stop")
    st.add_argument("job")
    st.set_defaults(func=lambda a: C.stop(a.job))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
