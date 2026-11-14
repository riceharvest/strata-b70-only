"""The packaged Terminal (#4188 part 3): `strata install` handing setup its options.
Nothing here starts an engine, downloads anything or reads the real profile."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import cli


class InstallForwardsSetupOptions(unittest.TestCase):
    """#4188: `strata install --source modelscope --family qwen ...` was refused by strata's own parser."""

    def test_the_options_of_setup_go_behind_a_double_dash_and_update_stays_strata_s(self):
        self.assertEqual(cli._install_argv(["install", "--source", "modelscope", "--yes"]), ["install", "--", "--source", "modelscope", "--yes"])
        self.assertEqual(cli._install_argv(["install", "--family", "qwen", "--update"]), ["install", "--update", "--", "--family", "qwen"])
        self.assertEqual(cli._install_argv(["install"]), ["install", "--"])
        self.assertEqual(cli._install_argv(["install", "-h"]), ["install", "-h"])
        self.assertEqual(cli._install_argv(["status", "--source", "x"]), ["status", "--source", "x"])
        self.assertEqual(cli._install_argv([]), [])

    def test_the_parser_hands_all_of_it_to_setup_in_order(self):
        ap = cli.parser()
        for argv in (["install", "--source", "modelscope", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32k", "--yes"],
                     ["install", "--family", "qwen", "--yes"], ["install", "--yes"]):
            a, extras = ap.parse_known_args(cli._install_argv(argv))
            self.assertEqual(extras, [], argv)
            self.assertEqual([x for x in a.setup_args if x != "--"], argv[1:], argv)

    def test_main_runs_setup_with_every_option_and_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "setup.py").write_text("# stand-in\n", encoding="utf-8")
            calls = []
            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": tmp}), \
                    mock.patch.object(subprocess, "call", side_effect=lambda argv, **kw: calls.append(argv) or 0):
                code = cli.main(["install", "--source", "modelscope", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32k", "--yes"])
                self.assertEqual(code, 0)
                code = cli.main(["install", "--update", "--yes"])
                self.assertEqual(code, 0)
        self.assertEqual(calls[0][2:], ["--setup", "--source", "modelscope", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32k", "--yes"])
        self.assertEqual(calls[1][2:], ["--update", "--yes"])


if __name__ == "__main__":
    unittest.main()
