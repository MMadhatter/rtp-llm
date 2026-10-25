# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c2_sim", HERE / "c2_sim.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


class C2SimulationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = json.loads((HERE / "c2-sim.example.json").read_text())
        self.data["common"]["dataset_id"] = "sha256:" + "a" * 64
        self.data["common"]["image"] = "example/platform:latest"
        for m in self.data["instances"]:
            m["hostname"] = (
                socket.gethostname() if m["rtp_role"] == "PREFILL" else "remote-host"
            )
        self.source = self.root / "spec.json"
        self.source.write_text(json.dumps(self.data))
        self.out = self.root / "rendered"
        self.spec = sim.render(self.source, self.out)
        self.member = self.spec["instances"][0]

    def test_pairing_and_single_owner(self):
        cfgs = [sim.config_for(self.spec, m, True) for m in self.spec["instances"]]
        self.assertEqual([c["SIDECAR_ROLE"] for c in cfgs], ["producer", "consumer"])
        for m in self.spec["instances"]:
            side = sim.config_for(self.spec, m, True)
            rtp = sim.config_for(self.spec, m, False)
            for k in (
                "INSTANCE_ID",
                "OWNER_INSTANCE_ID",
                "OWNER_EPOCH",
                "OWNER_ENDPOINT",
                "AUTHORIZATION_GROUP",
                "SERVICE_ID",
                "DATASET_ID",
            ):
                self.assertEqual(side[k], rtp[k])
            self.assertNotIn("SIDECAR_ROLE", rtp)
        self.assertEqual(cfgs[0]["ALLOWED_CLIENT_NAMES"], "prefill-a decode-b")
        self.assertEqual((self.out / "assignment.json").stat().st_mode & 0o777, 0o600)

    def test_rerender_never_overwrites_or_rotates_existing_epoch(self):
        before = (self.out / "assignment.json").read_text()
        with self.assertRaises(FileExistsError):
            sim.render(self.source, self.out)
        self.assertEqual(before, (self.out / "assignment.json").read_text())
        with self.assertRaisesRegex(ValueError, "input must not supply"):
            sim.render(self.out / "assignment.json", self.root / "other")

    def test_reject_ambiguous_members_resources_and_reservation(self):
        cases = []
        x = copy.deepcopy(self.data)
        x["common"]["owner_instance_id"] = "other-tenant"
        cases.append(x)
        x = copy.deepcopy(self.data)
        x["instances"][1]["instance_id"] = "prefill-a"
        cases.append(x)
        x = copy.deepcopy(self.data)
        x["instances"][1]["hostname"] = x["instances"][0]["hostname"]
        x["instances"][1]["gpu_uuids"] = x["instances"][0]["gpu_uuids"]
        cases.append(x)
        x = copy.deepcopy(self.data)
        x["owner"]["reserved_memory_bytes"] = 1
        cases.append(x)
        x = copy.deepcopy(self.data)
        x["owner"]["port"] = x["instances"][0]["http_port"]
        cases.append(x)
        x = copy.deepcopy(self.data)
        x["instances"][0]["uid"] = 0
        cases.append(x)
        for x in cases:
            with self.subTest(x=x["common"]["owner_instance_id"]), self.assertRaises(
                ValueError
            ):
                sim.validate(x)

    def test_shell_quote_prevents_command_injection(self):
        p = self.root / "quoted.sh"
        p.write_text(sim.shell_config({"VALUE": "$(touch BAD); `whoami` ' line"}))
        text = subprocess.check_output(
            ["bash", "-c", 'source "$1"; printf "%s" "$VALUE"', "bash", str(p)],
            cwd=self.root,
            text=True,
        )
        self.assertEqual(text, "$(touch BAD); `whoami` ' line")
        self.assertFalse((self.root / "BAD").exists())

    def test_wrong_host_blocks_runtime_access(self):
        with patch.object(
            sim, "run", side_effect=AssertionError("must not access runtime")
        ):
            with self.assertRaisesRegex(ValueError, "wrong host"):
                sim.main(
                    ["status", "--run-dir", str(self.out), "--instance", "decode-b"]
                )

    def test_modified_config_is_rejected_before_start(self):
        p = self.out / "prefill-a" / "sidecar.sh"
        p.write_text(p.read_text() + "OWNER_EPOCH=wrong\n")
        with patch.object(sim, "run", side_effect=AssertionError("no runtime")):
            with self.assertRaisesRegex(ValueError, "modified"):
                sim.start(self.out, self.spec, self.member, "sidecar")

    def container(self, part="rtp", running=True):
        cfg = sim.config_for(self.spec, self.member, part == "sidecar")
        env = [
            "RTP_ENGRAM_" + k + "=" + cfg[k]
            for k in (
                "INSTANCE_ID",
                "OWNER_EPOCH",
                "OWNER_ENDPOINT",
                "SERVICE_ID",
                "AUTHORIZATION_GROUP",
                "OWNER_INSTANCE_ID",
                "FABRIC_DOMAIN",
                "MODEL_REVISION",
                "DATASET_ID",
            )
        ]
        return {"Id": "f" * 64, "Config": {"Env": env}, "State": {"Running": running}}

    def receipt(self, part="rtp"):
        sim.write_private(
            sim.receipt_path(self.out, self.member, part),
            json.dumps({"epoch": self.spec["owner_epoch"], "container_id": "f" * 64}),
        )

    def test_idempotent_start_checks_receipt_and_does_not_restart(self):
        self.receipt("sidecar")
        with patch.object(
            sim, "inspect", return_value=self.container("sidecar")
        ), patch.object(sim, "run", side_effect=AssertionError("no start")):
            result = sim.start(self.out, self.spec, self.member, "sidecar")
            self.assertEqual(result["state"], "ALREADY_RUNNING")

    def test_stale_or_foreign_container_is_rejected(self):
        self.receipt()
        obj = self.container()
        obj["Config"]["Env"][0] = "RTP_ENGRAM_INSTANCE_ID=other"
        with patch.object(sim, "inspect", return_value=obj), self.assertRaisesRegex(
            ValueError, "assignment mismatch"
        ):
            sim.checked_container(self.out, self.spec, self.member, "rtp")

    def test_old_image_preflight_does_not_start_sidecar(self):
        failure = subprocess.CalledProcessError(1, ["pouch", "run"])
        with patch.object(sim, "preflight"), patch.object(
            sim, "run", side_effect=failure
        ) as run:
            with self.assertRaisesRegex(ValueError, "780bd26a4"):
                sim.start(self.out, self.spec, self.member, "sidecar")
        self.assertEqual(run.call_count, 1)
        self.assertFalse(sim.receipt_path(self.out, self.member, "sidecar").exists())

    def test_rtp_waits_for_owner_and_start_uses_shared_config(self):
        with patch.object(sim, "preflight"), patch.object(
            sim, "owner_ready", side_effect=ValueError("not ready")
        ), patch.object(sim, "run") as run:
            with self.assertRaisesRegex(ValueError, "not ready"):
                sim.start(self.out, self.spec, self.member, "rtp")
            run.assert_not_called()
        with patch.object(sim, "preflight"), patch.object(
            sim, "owner_ready", return_value={}
        ), patch.object(sim, "run", return_value=""), patch.object(
            sim, "inspect", return_value=self.container()
        ):
            sim.start(self.out, self.spec, self.member, "rtp")
        self.assertTrue(sim.receipt_path(self.out, self.member, "rtp").exists())

    def test_stop_only_targets_recorded_local_rtp(self):
        self.receipt()
        with patch.object(
            sim,
            "inspect",
            side_effect=[self.container(), self.container(running=False)],
        ), patch.object(sim, "run", return_value="") as run, contextlib.redirect_stdout(
            io.StringIO()
        ) as output:
            sim.main(
                ["stop-rtp", "--run-dir", str(self.out), "--instance", "prefill-a"]
            )
        run.assert_called_once_with(
            ["pouch", "stop", "-t", "300", "f" * 64], timeout=330
        )
        self.assertTrue(json.loads(output.getvalue())["owner_retained"])

    def test_chat_is_inside_prefill_and_decode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefill frontend"):
            sim.http_probe(
                self.out, self.spec, self.spec["instances"][1], "chat", "model"
            )
        with patch.object(
            sim, "checked_container", return_value=self.container()
        ), patch.object(sim, "execute", return_value='{"state":"CHAT_OK"}') as execute:
            sim.http_probe(self.out, self.spec, self.member, "chat", "model")
        self.assertIn("python", execute.call_args.args)
        self.assertIn("18640", execute.call_args.args)
        self.assertIn("127.0.0.1", sim.HTTP_PROBE)

    def test_readiness_auth_failure_does_not_wait(self):
        for details in ("PERMISSION_DENIED", b"PERMISSION_DENIED"):
            failure = subprocess.CalledProcessError(1, ["probe"], stderr=details)
            with patch.object(
                sim.time, "sleep", side_effect=AssertionError("must not retry")
            ):
                with self.assertRaisesRegex(ValueError, "authorization"):
                    sim.wait_for(lambda: (_ for _ in ()).throw(failure), 300)

    def test_real_pki_binds_instance_cn_and_owner_san(self):
        previous = os.umask(0o077)
        try:
            sim.pki(self.spec, self.root / "pki")
        finally:
            os.umask(previous)
        for m in self.spec["instances"]:
            directory = self.root / "pki" / m["instance_id"]
            subject = sim.run(
                [
                    "openssl",
                    "x509",
                    "-in",
                    str(directory / "consumer.crt"),
                    "-noout",
                    "-subject",
                    "-nameopt",
                    "RFC2253",
                ]
            )
            self.assertIn("CN=" + m["instance_id"], subject)
            self.assertFalse((directory / "ca.key").exists())
            self.assertEqual((directory / "consumer.key").stat().st_mode & 0o777, 0o600)
            sim.run(
                [
                    "openssl",
                    "verify",
                    "-CAfile",
                    str(directory / "ca.crt"),
                    str(directory / "consumer.crt"),
                ]
            )
        owner = self.root / "pki" / "prefill-a"
        text = sim.run(
            ["openssl", "x509", "-in", str(owner / "owner.crt"), "-noout", "-text"]
        )
        self.assertIn("IP Address:192.0.2.10", text)
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        member = {**self.member, "checkpoint": str(checkpoint), "tls_dir": str(owner)}
        with patch.object(sim.os, "geteuid", return_value=member["uid"]):
            sim.preflight(member, {}, "sidecar")


if __name__ == "__main__":
    unittest.main()
