"""Entrypoint of the SageMaker network-smoke job.

Reads ``/opt/ml/input/config/resourceconfig.json`` the way the real launcher will, then:

* on ``hosts[0]`` (the future Ray head) serves ``fake_session_server`` on ``0.0.0.0:PORT``
  and logs every request with its client address -- that log line is the evidence that an
  AgentCore runtime reached the container over the VPC;
* on every other host, resolves the head by name and by VPC IP, probes
  ``http://<head>:PORT/health`` and logs ``WORKER_PROBE ok|fail`` -- the evidence that
  intra-job traffic (Ray's 6379 later) works on the same path.

Both roles print ``HOST_REPORT {...}`` with every interface/IP they see, so the launcher can
read the head's VPC IP straight out of CloudWatch and hand it to ``InvokeAgentRuntime``.

Environment:
    SMOKE_PORT        port the head serves on (default 30000, the recipe's first session-server port)
    SMOKE_DURATION_S  how long the job stays up for probing (default 1200)
    MILES_RFT_FRONT_DOOR      "1": also run rft_front_door.py on MILES_RFT_FRONT_DOOR_PORT (30100), publish
                              MILES_HEAD_DNS in Route 53 zone MILES_ROUTE53_ZONE_ID, and pre-register the
                              trajectory id SMOKE_TRAJECTORY_ID ("smoke-traj") to a fake session -- so an
                              RFT-contract agent can be invoked from outside with that trajectory id.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import struct
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from fake_session_server import make_app
from rft_front_door import make_app as make_front_door

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("smoke")

RESOURCE_CONFIG = Path("/opt/ml/input/config/resourceconfig.json")
PORT = int(os.environ.get("SMOKE_PORT", "30000"))
DURATION_S = int(os.environ.get("SMOKE_DURATION_S", "1200"))


def _iface_ipv4(iface: str) -> str | None:
    """IPv4 of one interface via SIOCGIFADDR; no iproute2 needed in the image."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", iface[:15].encode()))
        except OSError:
            return None
    return socket.inet_ntoa(packed[20:24])


def _all_ifaces() -> dict[str, str | None]:
    names = sorted(p.name for p in Path("/sys/class/net").iterdir())
    return {name: _iface_ipv4(name) for name in names}


def _default_route_ip() -> str | None:
    """What ``miles.utils.http_utils.get_host_info`` would pick: the UDP-probe source address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _resolve(host: str, attempts: int = 60) -> str | None:
    for _ in range(attempts):
        try:
            return socket.gethostbyname(host)
        except socket.gaierror:
            time.sleep(2)
    return None


def _read_resource_config() -> dict:
    if RESOURCE_CONFIG.exists():
        return json.loads(RESOURCE_CONFIG.read_text())
    # Local docker run: behave as a single head.
    return {"current_host": socket.gethostname(), "hosts": [socket.gethostname()], "network_interface_name": "eth0"}


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", access_log=True))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def _publish_head_dns(vpc_ip: str) -> None:
    import boto3

    name = os.environ["MILES_HEAD_DNS"]
    change = boto3.client("route53").change_resource_record_sets(
        HostedZoneId=os.environ["MILES_ROUTE53_ZONE_ID"],
        ChangeBatch={"Changes": [{"Action": "UPSERT", "ResourceRecordSet": {
            "Name": name, "Type": "A", "TTL": 30, "ResourceRecords": [{"Value": vpc_ip}]}}]},
    )["ChangeInfo"]
    logger.info("HEAD_DNS %s -> %s (%s)", name, vpc_ip, change["Status"])


def _serve_head(vpc_ip: str | None) -> None:
    rft = os.environ.get("MILES_RFT_FRONT_DOOR") == "1"
    servers = [_serve(make_app(rft=rft), PORT)]
    if rft:
        fd_port = int(os.environ.get("MILES_RFT_FRONT_DOOR_PORT", "30100"))
        servers.append(_serve(make_front_door(), fd_port))
        _publish_head_dns(vpc_ip or "127.0.0.1")
        # Pre-register one trajectory -> a fresh fake session, so the RFT agent can be driven
        # from outside the VPC with just that trajectory id.
        tid = os.environ.get("SMOKE_TRAJECTORY_ID", "smoke-traj")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                sid = httpx.post(f"http://127.0.0.1:{PORT}/sessions", timeout=5).json()["session_id"]
                httpx.post(f"http://127.0.0.1:{fd_port}/miles/register", timeout=5,
                           json={"trajectory_id": tid, "session_url": f"http://127.0.0.1:{PORT}/sessions/{sid}"}).raise_for_status()
                break
            except (httpx.HTTPError, KeyError):
                time.sleep(1)
        else:
            raise RuntimeError("RFT smoke trajectory could not be registered; head is not ready")
        logger.info("FRONT_DOOR_READY port=%s trajectory_id=%s session_id=%s", fd_port, tid, sid)
    logger.info("HEAD_READY ip=%s port=%s duration_s=%s", vpc_ip, PORT, DURATION_S)
    time.sleep(DURATION_S)
    for server in servers:
        server.should_exit = True


def _probe_worker(head_host: str) -> None:
    head_ip = _resolve(head_host)
    logger.info("WORKER_RESOLVED %s -> %s", head_host, head_ip)
    targets = [t for t in (head_host, head_ip) if t]
    deadline = time.monotonic() + 600
    results: dict[str, str] = {}
    while time.monotonic() < deadline and len(results) < len(targets):
        for target in targets:
            if target in results:
                continue
            try:
                r = httpx.get(f"http://{target}:{PORT}/health", timeout=5)
                results[target] = f"ok status={r.status_code}"
            except httpx.HTTPError as exc:
                logger.info("probe %s:%s not yet: %s", target, PORT, type(exc).__name__)
        if len(results) < len(targets):
            time.sleep(5)
    for target in targets:
        logger.info("WORKER_PROBE %s target=%s:%s", results.get(target, "fail"), target, PORT)
    # Stay up so the job does not end before the head has been exercised.
    time.sleep(max(0, DURATION_S - 60))


def main() -> None:
    cfg = _read_resource_config()
    current, hosts, iface = cfg["current_host"], cfg["hosts"], cfg.get("network_interface_name", "eth0")
    ifaces = _all_ifaces()
    vpc_ip = ifaces.get(iface)
    report = {
        "current_host": current,
        "hosts": hosts,
        "network_interface_name": iface,
        "vpc_ip": vpc_ip,
        "default_route_ip": _default_route_ip(),
        "ifaces": ifaces,
        "training_job": os.environ.get("TRAINING_JOB_NAME"),
    }
    logger.info("HOST_REPORT %s", json.dumps(report, sort_keys=True))
    if current == hosts[0]:
        _serve_head(vpc_ip)
    else:
        _probe_worker(hosts[0])
    logger.info("DONE role=%s", "head" if current == hosts[0] else "worker")


if __name__ == "__main__":
    main()
