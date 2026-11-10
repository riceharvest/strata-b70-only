"""Saved read-only connections work in both source and portable terminal clients."""

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_cli import connections
from strata_cli.client import ClientError, Endpoint, server_url
from strata_cli.render import render_view


ROOT = Path(__file__).resolve().parents[2]


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-connections-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "connections"
        env = mock.patch.dict(os.environ, {"STRATA_CONNECTION_DIR": str(self.directory)}, clear=False)
        env.start()
        self.addCleanup(env.stop)

    def _cli(self, *argv):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(list(argv))
        return code, json.loads(output.getvalue())

    def test_create_edit_precedence_and_private_storage(self):
        code, created = self._cli("connection", "create", "edge", "--url", "https://Example.COM:443",
                                  "--prefix", "/proxy/strata", "--credential-env", "STRATA_READ_KEY",
                                  "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(created["connection"]["url"], "https://example.com:443")
        self.assertEqual(created["connection"]["prefix"], "/proxy/strata")
        saved = self.directory / "edge.json"
        if os.name == "posix":
            self.assertEqual(self.directory.stat().st_mode & 0o077, 0)
            self.assertEqual(saved.stat().st_mode & 0o077, 0)
        self.assertNotIn("test-secret", saved.read_text())
        revision = created["revision"]
        code, plan = self._cli("connection", "set", "edge", "prefix=/new", "--plan",
                                "--if-revision", revision, "--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(plan["would_save"])
        self.assertEqual(connections.load("edge")[1], revision)
        code, changed = self._cli("connection", "set", "edge", "prefix=/new",
                                   "--if-revision", revision, "--output", "json")
        self.assertEqual(code, 0)
        self.assertNotEqual(changed["revision"], revision)
        code, stale = self._cli("connection", "set", "edge", "prefix=/other",
                                 "--if-revision", revision, "--output", "json")
        self.assertEqual((code, stale["error"]["code"]), (6, 6))
        self.assertEqual(connections.load("edge")[0]["prefix"], "/new")
        with mock.patch.dict(os.environ, {"STRATA_CONNECTION": "edge", "STRATA_URL": "https://ignored.example",
                                       "STRATA_READ_KEY": "test-secret"}):
            code, effective = self._cli("config", "show", "--effective", "--output", "json")
            self.assertEqual(code, 0)
            self.assertEqual(effective["effective"]["url"], "https://example.com:443")
            self.assertEqual(effective["effective"]["prefix"], "/new")
            self.assertEqual(effective["effective"]["base_url"], "https://example.com:443/new")
            self.assertEqual(effective["effective"]["origins"]["url"], "connection")
            self.assertEqual(effective["effective"]["timeout_s"], 3.0)
            self.assertEqual(effective["effective"]["origins"]["timeout_s"], "default")
            self.assertTrue(effective["effective"]["credential_present"])
            self.assertNotIn("test-secret", json.dumps(effective))
            code, override = self._cli("config", "show", "--url", "https://other.example",
                                       "--credential-env", "OTHER_KEY", "--timeout", "4.5",
                                       "--output", "json")
            self.assertEqual(code, 0)
            self.assertEqual(override["effective"]["url"], "https://other.example")
            self.assertEqual(override["effective"]["prefix"], "")
            self.assertEqual(override["effective"]["credential_env"], "OTHER_KEY")
            self.assertEqual(override["effective"]["origins"]["url"], "flag")
            self.assertEqual(override["effective"]["origins"]["timeout_s"], "flag")
            self.assertEqual(override["effective"]["timeout_s"], 4.5)
            code, conflict = self._cli("config", "show", "--url", "https://other.example",
                                       "--connection", "edge", "--output", "json")
            self.assertEqual((code, conflict["error"]["code"]), (2, 2))

    def test_connection_and_config_json_aliases(self):
        code, listed = self._cli("connection", "list", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(listed["connections"], [])
        code, effective = self._cli("config", "show", "--url", "http://127.0.0.1:8095", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(effective["ok"])
        self.assertEqual(effective["effective"]["url"], "http://127.0.0.1:8095")

    def test_quiet_connection_and_status_keep_machine_records(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["connection", "list", "--quiet"]), 0)
        self.assertEqual(output.getvalue(), "")
        code, listed = self._cli("connection", "list", "--quiet", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(listed["connections"], [])
        with mock.patch("strata_cli.monitor_commands.Endpoint") as endpoint:
            endpoint.return_value.snapshot.return_value = {"schema_version": 1}
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["status", "--url", "http://127.0.0.1:8095", "--quiet"]), 0)
            self.assertEqual(output.getvalue(), "")
            with redirect_stdout(output):
                self.assertEqual(main(["status", "--url", "http://127.0.0.1:8095",
                                       "--quiet", "--json"]), 0)
            self.assertTrue(json.loads(output.getvalue())["ok"])

    def test_invalid_prefix_credentials_and_file_permissions_fail_closed(self):
        for prefix in ("//proxy", "/a/../b", "/a?key=secret", "/a//b", "https://host/path"):
            with self.assertRaises(connections.ConnectionError):
                connections.prefix_path(prefix)
        with self.assertRaises(ValueError):
            server_url("http://@127.0.0.1:8095")
        with self.assertRaises(connections.ConnectionError):
            connections.credential_name("BAD-NAME")
        connections.create("local", "http://127.0.0.1:8095")
        with self.assertRaises(connections.ConnectionError):
            connections.create("local", "http://127.0.0.1:8095")
        if os.name == "posix":
            record = self.directory / "local.json"
            record.chmod(0o644)                          # our own file with drifted modes is put back, not refused
            connections.load("local")
            self.assertEqual(stat.S_IMODE(record.stat().st_mode), 0o600)
            with mock.patch("os.getuid", return_value=os.getuid() + 1):      # a file that belongs to another account is refused
                with self.assertRaisesRegex(connections.ConnectionError, "belongs to .*sudo chown"):
                    connections.load("local")
        record = self.directory / "local.json"
        corrupted = json.loads(record.read_text())
        corrupted["url"] = 42
        record.write_text(json.dumps(corrupted))
        code, response = self._cli("config", "show", "--connection", "local", "--output", "json")
        self.assertEqual((code, response["error"]["code"]), (5, 5))


if __name__ == "__main__":
    unittest.main()
