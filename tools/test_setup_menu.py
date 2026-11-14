"""Tests for setup.py's model menu (0.1.44): the family order, the Unsloth sizes in order with images on.

    python -m unittest tools.test_setup_menu
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class FamilyOrder(unittest.TestCase):
    def test_the_menu_order(self):
        fams = setup.menu_families(True)
        self.assertEqual(fams, ["qwen", "swift", "coder", "unsloth", "gyro"])

    def test_the_default_stays_the_first_family(self):
        self.assertEqual(setup.menu_families(False)[0], "qwen")
        self.assertIn("GSQ-RCO", setup.FAMILIES["qwen"]["by"])

    def test_the_unsloth_line_names_the_alternative_on_amd_and_intel(self):
        self.assertIn("UD-IQ4_XS", setup.family_menu_note("unsloth", True, None))
        self.assertEqual(setup.family_menu_note("unsloth", False, {"arch": 86}), "")
        self.assertEqual(setup.family_menu_note("coder", True, None), "")


class UnslothSizes(unittest.TestCase):
    def test_the_sizes_in_menu_order(self):
        names = [m for m in setup.MODELS if "unsloth" in setup.MODELS[m].get("families", ())]
        self.assertEqual(names, ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0", "UD-IQ1_S"])

    def test_nothing_is_experimental_and_images_are_on(self):
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0", "UD-IQ1_S"):
            d = setup.MODELS[m]
            self.assertFalse(d.get("experimental"), m)
            self.assertFalse(d.get("nvidia_only"), m)
            self.assertIs(d.get("vision"), True, m)
            self.assertIn("download_gb", d)
            self.assertIn("ram_gb", d)
        self.assertNotIn("experimental", setup.MODELS["UD-Q4_K_XL"]["about"].lower())
        self.assertTrue(setup.FAMILIES["unsloth"]["vision"])

    def test_the_family_line_lists_every_size(self):
        about = setup.FAMILIES["unsloth"]["about"]
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0", "UD-IQ1_S"):
            self.assertIn(m, about)


class RecommendedSizes(unittest.TestCase):
    """0.1.44 (owner): IQ3_S, UD-Q4_K_XL and UD-IQ4_XS are the recommended sizes; recommend, never force."""

    def names(self, family):
        return [m for m in setup.MODELS if family in setup.MODELS[m].get("families", ("qwen", "swift"))]

    def test_qwen_and_swift_default_to_iq3_s_when_it_fits(self):
        for fam in ("qwen", "swift"):
            n = self.names(fam)
            self.assertIn("IQ3_S", n)
            self.assertEqual(setup.recommended_size(fam, n, 67.7), "IQ3_S")
            self.assertEqual(setup.recommended_size(fam, n, 191.0), "IQ3_S")      # not the largest: IQ3_S stays the pick
            self.assertEqual([m for m in n if setup.size_tag(fam, m)], ["IQ3_S"])  # the others are choices, unmarked

    def test_without_the_ram_the_largest_that_fits_and_iq3_s_stays_marked(self):
        n = self.names("qwen")
        self.assertEqual(setup.recommended_size("qwen", n, 61.0), "IQ3_XXS")      # IQ3_S needs 62 GB
        self.assertEqual(setup.recommended_size("qwen", n, 52.0), "IQ3_XXS")      # 48 GB tier (RAM + VRAM - 5 GB)
        self.assertEqual(setup.recommended_size("qwen", n, 40.0, 11.0), "IQ3_XXS")  # 40 + 11 - 5 = 46 GB together
        self.assertEqual(setup.recommended_size("qwen", n, 32.0, 12.0), "IQ2_XS")  # 44 - 5 = 39 GB: the 40 GB tier
        self.assertEqual(setup.recommended_size("swift", self.names("swift"), 32.0, 12.0), "IQ2_XS")
        self.assertEqual(setup.recommended_size("qwen", n, 32.0, 35.0), "IQ3_S")   # 67 - 5 = 62 GB together
        self.assertEqual(setup.size_tag("qwen", "IQ3_S"), " (recommended)")

    def test_3425_the_tiers_leave_5_gb_for_windows(self):
        # the reporter: RTX 5080 (16 GB) + 32 GB of RAM summed to 48 GB and got IQ3_XXS, which read ~11 GB from the SSD
        n = self.names("qwen")
        self.assertEqual(setup.TIER_SYSTEM_RESERVE_GB, 5.0)
        self.assertEqual(setup.recommended_size("qwen", n, 31.9, 15.9), "IQ2_XS")
        self.assertEqual(setup.tier_pick(31.9, 15.9), ("swift", "IQ2_XS"))
        # boundaries: floors 38 / 46 / 62 plus the 5 GB
        self.assertEqual(setup.tier_pick(43.0), ("swift", "IQ2_XS"))
        self.assertIsNone(setup.tier_pick(42.9))
        self.assertEqual(setup.tier_pick(51.0), ("qwen", "IQ3_XXS"))
        self.assertEqual(setup.tier_pick(50.9), ("swift", "IQ2_XS"))
        self.assertEqual(setup.tier_pick(67.0), ("qwen", "IQ3_S"))
        self.assertEqual(setup.tier_pick(66.9), ("qwen", "IQ3_XXS"))
        self.assertEqual(setup.tier_pick(40.0, 3.0), ("swift", "IQ2_XS"))      # VRAM counts the same as RAM

    def test_coder_has_nothing_marked(self):
        n = self.names("coder")
        self.assertEqual([setup.size_tag("coder", m) for m in n], [""] * len(n))
        self.assertEqual(setup.recommended_size("coder", n, 64.0), n[0])


class UnslothDefaultEnv(unittest.TestCase):
    """0.1.44 (owner): the env setup writes for Unsloth's files (review-quants.md section 5)."""

    def test_q5_and_q6_turn_the_k_quant_prompt_kernels_on_q4_does_not(self):
        f = setup.unsloth_default_env
        self.assertEqual(f("unsloth", "UD-Q5_K_XL", False, False)["STRATA_MMQ_KQ"], "1")
        self.assertEqual(f("unsloth", "UD-Q6_K_XL", True, True)["STRATA_MMQ_KQ"], "1")
        for m in ("UD-Q4_K_XL", "UD-IQ4_XS", "Q8_0"):
            self.assertNotIn("STRATA_MMQ_KQ", f("unsloth", m, False, False), m)

    def test_hc_q8_on_cuda_for_the_ud_files_only(self):
        f = setup.unsloth_default_env
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL"):
            self.assertEqual(f("unsloth", m, False, True)["STRATA_HC_Q8"], "1", m)
            self.assertNotIn("STRATA_HC_Q8", f("unsloth", m, True, True), m)          # not on AMD
        self.assertNotIn("STRATA_HC_Q8", f("unsloth", "Q8_0", False, True))

    def test_tail_skip_off_only_when_every_expert_is_in_ram(self):
        f = setup.unsloth_default_env
        self.assertEqual(f("unsloth", "UD-Q4_K_XL", False, False)["STRATA_ROUTE_TAIL_SKIP"], "0")
        self.assertNotIn("STRATA_ROUTE_TAIL_SKIP", f("unsloth", "UD-Q4_K_XL", False, True))   # budget mode keeps the skip

    def test_other_families_are_untouched(self):
        for fam, m in (("qwen", "IQ3_S"), ("swift", "Q2_0"), ("coder", "IQ1_M")):
            self.assertEqual(setup.unsloth_default_env(fam, m, False, False), {}, fam)


class EnginePlaceholders(unittest.TestCase):
    def test_every_engine_minimum_is_this_release(self):
        self.assertEqual(setup.MIN_ENGINE, (0, 1, 46))
        for m in ("UD-Q5_K_XL", "UD-Q6_K_XL"):
            self.assertEqual(setup.MODELS[m]["engine"], (0, 1, 44), m)
        # No minimum may name an engine newer than this tree (a placeholder for a release that does not exist yet would make setup
        # ask for an engine nobody can download). The tree's version is read from CMakeLists.txt, so this does not go stale each release.
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        tree = tuple(int(n) for n in re.search(r"project\(strata VERSION ([0-9.]+)", text).group(1).split("."))
        for what, minimum in [("MIN_ENGINE", setup.MIN_ENGINE)] + [(m, setup.MODELS[m]["engine"]) for m in setup.MODELS if "engine" in setup.MODELS[m]]:
            self.assertLessEqual(tuple(minimum), tree[:len(minimum)], f"{what} {minimum} is ahead of the tree {tree}")


if __name__ == "__main__":
    unittest.main()
