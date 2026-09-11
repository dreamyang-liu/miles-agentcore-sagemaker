"""Offline checks for documented SageMaker arguments and runtime selection."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def test_multiline_extra_args_preserve_json_and_literal_text(self):
        entrypoint = load_file("example_entrypoint", ROOT / "sagemaker/entrypoint.py")
        extras = '--num-rollout 3\n--extra-env-vars \'{"VALUE":"$(literal) with spaces"}\''
        with patch.dict(os.environ, {"MILES_SM_EXTRA_ARGS": extras, "MILES_SM_AGENT_MODE": "rft"}):
            command = entrypoint._launcher_command("model with spaces", 8)
        self.assertEqual(command[command.index("--model-name") + 1], "model with spaces")
        encoded = command[command.index("--extra-env-vars") + 1]
        self.assertEqual(json.loads(encoded), {"VALUE": "$(literal) with spaces"})
        self.assertEqual(command[command.index("--num-rollout") + 1], "3")

    def test_explicit_runtime_does_not_require_default_runtime_file(self):
        with tempfile.TemporaryDirectory() as temp:
            common = ModuleType("sm_common")
            common.ECR = "example-registry"
            common.RUNTIME_STATE = Path(temp) / "missing-runtime.json"
            common.REGION = "us-west-2"
            common.ROLE = "test-execution-role"
            common.BUCKET = "test-input-bucket"
            common.INFRA = {
                "front_door_port": 30100, "head_dns": "head.example.internal",
                "route53_zone_id": "ZEXAMPLE",
            }
            common.vpc_config = lambda: {"Subnets": ["test-subnet"], "SecurityGroupIds": ["test-sg"]}
            common.sm = Mock()
            with patch.dict(sys.modules, {"sm_common": common}):
                launcher = load_file("example_launch_train", ROOT / "sagemaker/launch_train.py")
                args = argparse.Namespace(
                    runtime_arn="explicit-runtime-arn", mode="normal",
                    instance_type="ml.p5.48xlarge", count=1, model_name="Qwen3.6-27B",
                    dataset="test-data", max_runtime=7200, agent_mode="rft",
                    extra_args="--num-rollout 3", agentcore_max_concurrent=96,
                )
                with redirect_stdout(io.StringIO()):
                    launcher.start(args)
            request = common.sm.create_training_job.call_args.kwargs
            self.assertEqual(request["Environment"]["AGENTCORE_RUNTIME_ARN"], "explicit-runtime-arn")
            self.assertEqual(request["Environment"]["MILES_RFT_FRONT_DOOR"], "1")
            self.assertEqual(request["Environment"]["AGENTCORE_MAX_CONCURRENT"], "96")
            self.assertFalse(common.RUNTIME_STATE.exists())

    def test_args_file_helper_preserves_json_and_overrides(self):
        process = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_with_args.py"),
            "--args-file", str(ROOT / "configs/qwen36-rft-3step.args"), "--print-command",
            "--", "--dataset", "name with spaces",
        ], text=True, capture_output=True, check=True)
        command = shlex.split(process.stdout)
        self.assertEqual(command[-2:], ["--dataset", "name with spaces"])
        self.assertEqual(
            json.loads(command[command.index("--extra-env-vars") + 1]),
            {"RFT_STOP_SESSION_ON_FINISH": "1"},
        )
        self.assertEqual(command[command.index("--max-tokens-per-gpu") + 1], "20000")


if __name__ == "__main__":
    unittest.main()
