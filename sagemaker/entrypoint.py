"""SageMaker training-job entrypoint for the AgentCore math recipe, single- or multi-instance.

Runs inside the Miles image on every host of the job. Reads
``/opt/ml/input/config/resourceconfig.json`` and takes one of two roles:

* ``hosts[0]`` (head): starts the Ray head on its VPC IP, waits until every host has joined,
  then runs ``run_qwen3_agentcore_math.py`` with ``MILES_SCRIPT_EXTERNAL_RAY=1`` so the
  launcher only submits. The session servers are spawned by the Ray driver, i.e. here, on
  ports ``30000..`` of this VPC IP -- the address the AgentCore runtime connects to directly.
  When the launcher returns, the head stops Ray so the workers can exit too.
* every other host (worker): ``ray start --address=<head>:6379`` (retrying until the head is
  up), then blocks while the head's Ray port answers, and exits when it stops.

Inputs are SageMaker channels: ``model`` (an HF checkpoint directory, linked to
``/root/models/<MILES_SM_MODEL_NAME>``) and ``data`` (``<dataset>_{train,eval}.jsonl``).
Outputs go under ``/opt/ml/checkpoints`` so ``CheckpointConfig`` streams them to S3.

Environment (set by ``launch_train.py``):
    MILES_SM_MODE            smoke | normal            (default smoke)
    MILES_SM_MODEL_NAME      e.g. Qwen3-0.6B
    MILES_SM_DATASET         gsm8k | gsm-hard | rft-gsm8k ...
    MILES_SM_AGENT_MODE      agentcore | rft           (default agentcore)
    MILES_SM_EXTRA_ARGS      appended to the launcher command line
    MILES_RFT_FRONT_DOOR     "1": start rft_front_door.py on the head and publish its address as
                             MILES_HEAD_DNS (Route 53 zone MILES_ROUTE53_ZONE_ID) : MILES_RFT_FRONT_DOOR_PORT
    AGENTCORE_RUNTIME_ARN    passed through to the agent function
    AWS_REGION, AGENTCORE_MAX_CONCURRENT, WANDB_API_KEY  passed through if present
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s sm-entrypoint %(message)s")
logger = logging.getLogger("sm-entrypoint")

REPO = Path("/root/miles")
LAUNCHER = REPO / "examples/experimental/agentcore/run_qwen3_agentcore_math.py"
RESOURCE_CONFIG = Path("/opt/ml/input/config/resourceconfig.json")
MODEL_CHANNEL = Path("/opt/ml/input/data/model")
DATA_CHANNEL = Path("/opt/ml/input/data/data")
MODEL_DIR = Path("/root/models")
OUTPUT_DIR = Path("/opt/ml/checkpoints")
RAY_PORT = 6379
JOIN_TIMEOUT_S = 900
FRONT_DOOR = REPO / "examples/experimental/agentcore/rft_front_door.py"


def _iface_ipv4(iface: str) -> str | None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", iface[:15].encode()))
        except OSError:
            return None
    return socket.inet_ntoa(packed[20:24])


def _resolve(host: str, timeout_s: int = 600) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.gethostbyname(host)
        except socket.gaierror:
            if time.monotonic() > deadline:
                raise
            time.sleep(3)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def _gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def _sh(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    logger.info("$ %s", cmd)
    return subprocess.run(cmd, shell=True, check=False, **kwargs)


def _host_report(cfg: dict, own_ip: str, gpus: int) -> None:
    ifaces = {p.name: _iface_ipv4(p.name) for p in sorted(Path("/sys/class/net").iterdir())}
    shm = shutil.disk_usage("/dev/shm").total // 2**20
    report = {
        "current_host": cfg["current_host"],
        "hosts": cfg["hosts"],
        "network_interface_name": cfg.get("network_interface_name"),
        "vpc_ip": own_ip,
        "ifaces": ifaces,
        "gpus": gpus,
        "shm_mib": shm,
        "training_job": os.environ.get("TRAINING_JOB_NAME"),
    }
    logger.info("HOST_REPORT %s", json.dumps(report, sort_keys=True))


def _alive_ray_nodes() -> int:
    code = "import ray; ray.init(address='auto', logging_level='ERROR'); print(sum(n['Alive'] for n in ray.nodes()))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False).stdout.strip()
    return int(out) if out.isdigit() else 0


def _link_model(model_name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / model_name
    if not target.exists():
        target.symlink_to(MODEL_CHANNEL)
    if not (target / "config.json").exists():
        raise SystemExit(f"model channel at {MODEL_CHANNEL} has no config.json")


def _upsert_head_dns(zone_id: str, name: str, ip: str) -> None:
    """Point the private name at this head so a runtime's fixed endpoint follows the job."""
    import boto3

    route53 = boto3.client("route53")
    change = route53.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            "Comment": os.environ.get("TRAINING_JOB_NAME", "miles"),
            "Changes": [
                {
                    "Action": "UPSERT",
                    "ResourceRecordSet": {"Name": name, "Type": "A", "TTL": 30, "ResourceRecords": [{"Value": ip}]},
                }
            ],
        },
    )["ChangeInfo"]
    logger.info("HEAD_DNS %s -> %s (%s)", name, ip, change["Status"])


def _start_front_door(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(FRONT_DOOR), "--host", "0.0.0.0", "--port", str(port)],
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"},
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=2) as client:
                if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                    logger.info("FRONT_DOOR_READY port=%s pid=%s", port, proc.pid)
                    return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("rft_front_door did not become healthy")


def run_head(cfg: dict, own_ip: str, gpus: int) -> int:
    hosts = cfg["hosts"]
    model_name = os.environ.get("MILES_SM_MODEL_NAME", "Qwen3-0.6B")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    front_door = None
    front_door_env: dict[str, str] = {}
    if os.environ.get("MILES_RFT_FRONT_DOOR") == "1":
        port = int(os.environ.get("MILES_RFT_FRONT_DOOR_PORT", "30100"))
        head_dns = os.environ["MILES_HEAD_DNS"]
        _upsert_head_dns(os.environ["MILES_ROUTE53_ZONE_ID"], head_dns, own_ip)
        front_door = _start_front_door(port)
        front_door_env = {
            "MILES_RFT_FRONT_DOOR_URL": f"http://{head_dns}:{port}",
            "MILES_RFT_FRONT_DOOR_LOCAL": f"http://127.0.0.1:{port}",
        }

    _sh("ray stop --force >/dev/null 2>&1; true")
    rc = _sh(
        f"export PYTHONUNBUFFERED=1 && ray start --head --node-ip-address {own_ip} --port {RAY_PORT} "
        f"--num-gpus {gpus} --dashboard-host 127.0.0.1 --disable-usage-stats"
    ).returncode
    if rc != 0:
        return rc
    logger.info("HEAD_READY ip=%s ray_port=%s", own_ip, RAY_PORT)

    deadline = time.monotonic() + JOIN_TIMEOUT_S
    while (alive := _alive_ray_nodes()) < len(hosts):
        logger.info("RAY_NODES %s/%s joined", alive, len(hosts))
        if time.monotonic() > deadline:
            logger.error("only %s/%s ray nodes joined within %ss", alive, len(hosts), JOIN_TIMEOUT_S)
            return 1
        time.sleep(10)
    logger.info("RAY_NODES %s/%s joined", alive, len(hosts))

    env = {
        **os.environ,
        **front_door_env,
        "MILES_SCRIPT_EXTERNAL_RAY": "1",
        "MASTER_ADDR": own_ip,
        "SLURM_JOB_NUM_NODES": str(len(hosts)),
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",
    }
    env.pop("RAY_ADDRESS", None)  # execute_train then submits to the local dashboard
    # SageMaker's container runtime denies pidfd_getfd, which torch needs to share
    # expandable-segment CUDA allocations with the colocated SGLang engines during weight sync
    # ("RuntimeError: pidfd_getfd: Operation not permitted" in every engine at the first update).
    # Classic CUDA IPC handles do not need it, so the trainer runs without expandable segments.
    train_env_vars = json.dumps({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"})
    cmd = (
        f"python3 {LAUNCHER} --mode {os.environ.get('MILES_SM_MODE', 'smoke')} "
        f"--agent-mode {os.environ.get('MILES_SM_AGENT_MODE', 'agentcore')} "
        f"--skip-prepare --model-name {model_name} --dataset {os.environ.get('MILES_SM_DATASET', 'gsm-hard')} "
        f"--model-dir {MODEL_DIR} --data-dir {DATA_CHANNEL} --output-dir {OUTPUT_DIR} "
        f"--num-gpus-per-node {gpus} --train-env-vars '{train_env_vars}' "
        f"{os.environ.get('MILES_SM_EXTRA_ARGS', '')}"
    )
    rc = _sh(cmd, env=env, cwd=LAUNCHER.parent).returncode
    logger.info("LAUNCHER_EXIT rc=%s", rc)
    if front_door is not None:
        front_door.terminate()
    _sh("ray stop --force; true")
    return rc


def run_worker(cfg: dict, own_ip: str, gpus: int) -> int:
    head_ip = _resolve(cfg["hosts"][0])
    logger.info("WORKER_RESOLVED %s -> %s", cfg["hosts"][0], head_ip)
    deadline = time.monotonic() + JOIN_TIMEOUT_S
    while True:
        if _port_open(head_ip, RAY_PORT):
            rc = _sh(
                f"export PYTHONUNBUFFERED=1 && ray start --address={head_ip}:{RAY_PORT} --num-gpus {gpus} "
                f"--node-ip-address {own_ip} --disable-usage-stats"
            ).returncode
            if rc == 0:
                break
        if time.monotonic() > deadline:
            logger.error("could not join ray head %s:%s within %ss", head_ip, RAY_PORT, JOIN_TIMEOUT_S)
            return 1
        time.sleep(10)
    logger.info("WORKER_JOINED head=%s:%s", head_ip, RAY_PORT)
    # Hold the node until the head tears Ray down (LAUNCHER_EXIT on its side).
    misses = 0
    while misses < 3:
        time.sleep(30)
        misses = 0 if _port_open(head_ip, RAY_PORT) else misses + 1
    logger.info("WORKER_DONE head gone")
    _sh("ray stop --force; true")
    return 0


def main() -> None:
    cfg = json.loads(RESOURCE_CONFIG.read_text())
    iface = cfg.get("network_interface_name", "eth0")
    own_ip = _iface_ipv4(iface) or socket.gethostbyname(socket.gethostname())
    gpus = _gpu_count()
    _host_report(cfg, own_ip, gpus)
    # Every host: SGLang engines are placed on workers too and open the HF checkpoint by
    # this path (the `model` channel is FullyReplicated, so it exists on each host).
    _link_model(os.environ.get("MILES_SM_MODEL_NAME", "Qwen3-0.6B"))
    # NCCL/Gloo must use the VPC interface; execute_train forwards these into the Ray runtime env.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", iface)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
    os.environ.setdefault("MILES_HOST_IP", own_ip)
    is_head = cfg["current_host"] == cfg["hosts"][0]
    rc = run_head(cfg, own_ip, gpus) if is_head else run_worker(cfg, own_ip, gpus)
    logger.info("DONE role=%s rc=%s", "head" if is_head else "worker", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
