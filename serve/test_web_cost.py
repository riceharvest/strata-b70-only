"""Tests for the Cost Saved feature in the web app.

Runs pricing, cost calculations, and HTML checks with Node and unittest:
    python3 -m unittest serve.test_web_cost -v
"""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

WEB = Path(__file__).with_name("web")
APP = WEB / "app.js"
INDEX = WEB / "index.html"
CSS = WEB / "dashboard.css"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebCostSaved(unittest.TestCase):
    def setUp(self):
        self.app_src = APP.read_text(encoding="utf-8")
        self.index_html = INDEX.read_text(encoding="utf-8")
        self.dash_css = CSS.read_text(encoding="utf-8")

    def run_js(self, expr):
        js = f"""
const fs = require('fs');
const src = fs.readFileSync('{APP}', 'utf-8');
const sub = src.slice(src.indexOf('const QWEN_FLASH_PRICING'), src.indexOf('const store ='));
const fmtDef = 'const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {{maximumFractionDigits: d, minimumFractionDigits: d}}));';
const out = eval(fmtDef + ';' + sub + ';' + {json.dumps(expr)});
console.log(JSON.stringify(out));
"""
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_qwen_flash_pricing_rates(self):
        pricing = self.run_js("QWEN_FLASH_PRICING")
        self.assertEqual(pricing["provider"], "Alibaba Cloud Int.")
        self.assertEqual(pricing["model"], "Qwen-Flash")
        self.assertAlmostEqual(pricing["inputPerM"], 0.15)
        self.assertAlmostEqual(pricing["outputPerM"], 0.47)
        self.assertAlmostEqual(pricing["cacheReadPerM"], 0.016)
        self.assertAlmostEqual(pricing["cacheCreate5mPerM"], 0.20)
        self.assertAlmostEqual(pricing["cacheRead5mPerM"], 0.016)

    def test_calc_cost_and_fmt_cost(self):
        # 0 tokens
        self.assertEqual(self.run_js("fmtCost(calcCost(0, 0, 0))"), "$0.00")
        # 1,000 fresh prompt tokens + 200 out
        self.assertEqual(self.run_js("fmtCost(calcCost(1000, 0, 200))"), "$0.0002")
        # 5,000 prompt tokens (2,000 reused, 3,000 fresh) + 1,000 out:
        # fresh: 3000 * 0.15/1e6 = 0.00045
        # reused: 2000 * 0.016/1e6 = 0.000032
        # out: 1000 * 0.47/1e6 = 0.00047
        # sum = 0.000952 -> $0.0010
        self.assertEqual(self.run_js("fmtCost(calcCost(5000, 2000, 1000))"), "$0.0010")
        # 1M prompt + 500k out = 0.15 + 0.235 = 0.385 -> $0.39
        self.assertEqual(self.run_js("fmtCost(calcCost(1000000, 0, 500000))"), "$0.39")
        # 10M prompt + 5M out = 1.50 + 2.35 = 3.85 -> $3.85
        self.assertEqual(self.run_js("fmtCost(calcCost(10000000, 0, 5000000))"), "$3.85")

    def test_dashboard_tile_and_tables(self):
        self.assertIn('tile("Cost saved"', self.app_src)
        self.assertIn('title="Estimated cost saved vs Alibaba Cloud Qwen-Flash', self.index_html)
        self.assertIn('Saved</th>', self.index_html)
        self.assertIn('facts-cost', self.index_html)
        self.assertIn('data-tone="cost"', self.dash_css)


if __name__ == "__main__":
    unittest.main()
