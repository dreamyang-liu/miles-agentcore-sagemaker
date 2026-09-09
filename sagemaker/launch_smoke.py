"""Stage-1 network smoke: a 2-host SageMaker training job serving a fake session server.

Submits ``miles-sagemaker-smoke`` (see ``smoke/``) into the VPC built by ``infra.py``, then
reads the hosts' ``HOST_REPORT`` / ``HEAD_READY`` / ``WORKER_PROBE`` lines out of CloudWatch.
The head's VPC IP printed here is what ``agentcore_runtime.py invoke`` hands to the agent.

Usage:
    python launch_smoke.py start [--instance-type ml.m6i.large] [--count 2] [--duration 1200]
    python launch_smoke.py watch <job-name>     # poll status, print the marker lines, stop at HEAD_READY
    python launch_smoke.py logs  <job-name>     # dump every log line seen so far
    python launch_smoke.py stop  <job-name>
"""

from __future__ import annotations

import argparse
import re
import time

import sm_common as C

IMAGE = f"{C.ECR}/miles-sagemaker-smoke:latest"
_MARKERS = re.compile(r"(HOST_REPORT|HEAD_READY|WORKER_RESOLVED|WORKER_PROBE|DONE|POST /sessions|BREACH)")


def start(args: argparse.Namespace) -> None:
    name = f"miles-net-smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    C.sm.create_training_job(
        TrainingJobName=name,
        RoleArn=C.ROLE,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": args.count, "VolumeSizeInGB": 30},
        VpcConfig=C.vpc_config(),
        OutputDataConfig={"S3OutputPath": f"s3://{C.BUCKET}/miles-agentcore-smoke/"},
        StoppingCondition={"MaxRuntimeInSeconds": args.duration + 900},
        Environment={"SMOKE_PORT": str(args.port), "SMOKE_DURATION_S": str(args.duration)},
        Tags=[{"Key": "project", "Value": "miles-agentcore"}],
    )
    print(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start")
    s.add_argument("--instance-type", default="ml.m6i.large")
    s.add_argument("--count", type=int, default=2)
    s.add_argument("--duration", type=int, default=1200)
    s.add_argument("--port", type=int, default=30000)
    s.set_defaults(func=start)
    w = sub.add_parser("watch")
    w.add_argument("job")
    w.add_argument("--until-done", action="store_true", help="keep polling after the head is ready")
    w.set_defaults(func=lambda a: C.watch(a.job, _MARKERS, stop_when=None if a.until_done else re.compile("HEAD_READY")))
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
