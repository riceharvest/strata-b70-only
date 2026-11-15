"""The config's "engine_ready_s": a big model read from a slow disk needs more than the default 900 s before READY.

    python -m unittest serve.test_engine_ready_config -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ReadyTimeout(unittest.TestCase):
    def test_the_config_sets_it_and_a_bad_value_is_said(self):
        from serve import server
        before = server.ENGINE_READY_S
        self.addCleanup(setattr, server, "ENGINE_READY_S", before)
        server.apply_ready_timeout({})
        self.assertEqual(server.ENGINE_READY_S, before)
        server.apply_ready_timeout({"engine_ready_s": 3600})
        self.assertEqual(server.ENGINE_READY_S, 3600.0)
        for bad in (0, -5, "soon", True):
            with self.assertRaises(ValueError):
                server.apply_ready_timeout({"engine_ready_s": bad})


if __name__ == "__main__":
    unittest.main()
