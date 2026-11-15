"""The update's test start picks a saved model's run config: a config it cannot read is still tried, as before."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import models, updater


class DiscoverRunConfigCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-discover-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, name: str, **fields) -> str:
        path = self.root / f"strata-{name}.json"
        path.write_text(json.dumps(fields), encoding="utf-8")
        return str(path)

    def discover(self, *entries):
        listed = {"models": [{"id": i, "component_status": "present", "source_config": c} for i, c in entries]}
        with mock.patch.object(models, "list_installed", return_value=listed), \
                mock.patch("strata_control.profiles.list_profiles", return_value=[]):
            return updater.discover_run_config()

    def test_other_and_unreadable_configs_are_still_used(self):
        qwen = self.config("qwen", exe="C:/e/engine/strata.exe")
        broken = self.root / "strata-broken.json"
        broken.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.discover(("a-broken", str(broken))), broken)
        self.assertEqual(self.discover(("b-qwen", qwen)), Path(qwen))


if __name__ == "__main__":
    unittest.main()
