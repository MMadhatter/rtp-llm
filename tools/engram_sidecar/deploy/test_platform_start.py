# SPDX-License-Identifier: Apache-2.0
"""Exercise launch argv with a recording runtime; never starts a container."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class PlatformLaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record = self.root / "argv.json"
        binary = self.root / "pouch"
        binary.write_text(
            "#!/usr/bin/env python3\nimport json,os,sys\nif sys.argv[1]=='inspect': sys.exit(1)\nassert sys.argv[1]=='run'\nopen(os.environ['RECORD'],'w').write(json.dumps(sys.argv[1:]))\n"
        )
        binary.chmod(0o700)
        binary = self.root / "nvidia-smi"
        binary.write_text("#!/usr/bin/env bash\nexit 0\n")
        binary.chmod(0o700)
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "ca.crt",
            "owner.crt",
            "owner.key",
            "consumer.crt",
            "consumer.key",
            "model.env",
        ):
            (self.root / name).write_text("{}")
        self.env = {
            **os.environ,
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "RECORD": str(self.record),
        }
        self.config = dict(
            IMAGE="example/test:assignment",
            NAME="test",
            CHECKPOINT=str(self.root),
            STATE_DIR=str(self.root / "state"),
            TLS_DIR=str(self.root),
            OWNER_IP="192.0.2.1",
            OWNER_PORT="19780",
            GPU_UUID="GPU-example",
            NUMA="0",
            MODEL_REVISION="rev1",
            SERVICE_ID="service1",
            MEMORY_LIMIT="4g",
            CONTAINER_UID="123",
            CONTAINER_GID="123",
            DISCOVERY="platform",
            OWNER_ENDPOINT="192.0.2.1:19780",
            AUTHORIZATION_GROUP="tenant-a",
            FABRIC_DOMAIN="fabric-a",
            INSTANCE_ID="owner-a",
            OWNER_INSTANCE_ID="owner-a",
            OWNER_EPOCH="01234567-1234-1234-1234-0123456789ab",
            DATASET_ID="sha256:" + "a" * 64,
            PLATFORM_IMEX_CHANNEL="7",
            SIDECAR_ROLE="producer",
            RESERVATION_ID="reserved",
            RESERVED_MEMORY_BYTES="1024",
            REQUIRED_MEMORY_BYTES="512",
            ALLOWED_CLIENT_NAMES="owner-a consumer-b",
            MODEL_ENV=str(self.root / "model.env"),
            GPU_UUIDS="GPU-example",
            SHM_SIZE="1g",
        )

    def run_script(self, script, *args, expect=0):
        path = self.root / "config.sh"
        path.write_text(
            "\n".join(k + "=" + shlex.quote(v) for k, v in self.config.items())
        )
        run = subprocess.run(
            ["bash", str(ROOT / script), str(path), *args],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, expect, run.stderr)
        return json.loads(self.record.read_text()) if self.record.exists() else []

    def test_producer_uses_assignment_and_exact_channel(self):
        args = self.run_script("sidecar-start.sh")
        self.assertIn("RTP_ENGRAM_DISCOVERY=platform", args)
        self.assertIn("RTP_ENGRAM_SIDECAR_ROLE=producer", args)
        self.assertIn("--device=/dev/nvidia-caps-imex-channels/channel7", args)
        self.assertNotIn("auto", args)
        self.assertFalse(any("REGISTRY_ENDPOINT" in value for value in args))
        self.assertIn("consumer-b", args)

    def test_consumer_sidecar_is_cpu_observer(self):
        self.config.update(SIDECAR_ROLE="consumer", INSTANCE_ID="consumer-b")
        args = self.run_script("sidecar-start.sh")
        self.assertIn("RTP_ENGRAM_OWNER_ENDPOINT=192.0.2.1:19780", args)
        self.assertIn("RTP_ENGRAM_INSTANCE_ID=consumer-b", args)
        self.assertFalse(
            any(value.startswith(("--device", "--nvidia")) for value in args)
        )
        self.assertFalse(any("REGISTRY_ENDPOINT" in value for value in args))

    def test_rtp_shared_and_local_ab(self):
        self.config.update(INSTANCE_ID="consumer-b")
        args = self.run_script("consumer-start.sh", "shared")
        self.assertIn("--device=/dev/nvidia-caps-imex-channels/channel7", args)
        self.assertIn("RTP_ENGRAM_OWNER_INSTANCE_ID=owner-a", args)
        args = self.run_script("consumer-start.sh", "local")
        self.assertFalse(any("RTP_ENGRAM_OWNER_" in value for value in args))
        self.assertFalse(any("--device=" in value for value in args))

    def test_wrong_producer_and_auto_rejected_before_run(self):
        self.config.update(INSTANCE_ID="unselected")
        self.run_script("sidecar-start.sh", expect=2)
        self.assertFalse(self.record.exists())
        self.config.update(SIDECAR_ROLE="auto")
        self.run_script("sidecar-start.sh", expect=2)
        self.assertFalse(self.record.exists())

    def test_registry_ambiguity_rejected(self):
        self.config["REGISTRY_ENDPOINT"] = "192.0.2.2:19781"
        self.run_script("sidecar-start.sh", expect=2)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
