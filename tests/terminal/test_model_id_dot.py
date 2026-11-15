"""Qwen3.8-Flash-Next's model id has a dot ("qwen3.8-flash-next-iq3_xxs"). The app lists installed models through Control, which
used to drop any id with a dot: the install finished and the model never appeared."""
import json
import tempfile
import unittest
from pathlib import Path

from strata_control import models


class DotInModelId(unittest.TestCase):
    def test_ids_with_a_dot_are_accepted_and_odd_ones_are_not(self):
        for good in ("qwen3.8-flash-next-iq3_xxs", "q2_0", "qwen36-ud-q4_k_xl"):
            self.assertTrue(models.MODEL_ID.fullmatch(good), good)
        for bad in ("a..b", ".a", "../x", "a/b", "A"):
            self.assertFalse(models.MODEL_ID.fullmatch(bad), bad)

    def test_a_run_config_with_a_dotted_name_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "strata-qwen3.8-flash-next-iq3_xxs.json").write_text(json.dumps(
                {"exe": "/x/strata", "tokenizer": str(folder), "args": ["--max-context", "8192"], "model_name": "qwen3.8-flash-next"}))
            listed = models.list_installed(str(folder))
            self.assertEqual([m["id"] for m in listed["models"]], ["qwen3.8-flash-next-iq3_xxs"])


if __name__ == "__main__":
    unittest.main()
