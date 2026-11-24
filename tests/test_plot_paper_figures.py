"""Plotting smoke/validation tests. Synthetic inputs are NOT study outputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
module_path = ROOT / "6_plot_figures/plot_paper_figures.py"
spec = importlib.util.spec_from_file_location("figures", module_path)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class PlottingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        gate = []
        for year in figures.YEARS:
            gate.append({
                "year": year,
                "survivor_test_rate_weighted_all_survivors": .015 + (year-2016)*.0002,
                "fatal_test_rate_weighted_all_fatal": .3 + (year-2016)*.001,
                "q_test_BAC_gt0_weighted": .8 - (year-2016)*.04,
            })
        self.gate = self.base / "gate.csv"
        pd.DataFrame(gate).to_csv(self.gate, index=False)
        metrics = []
        for experiment, eval_names, arms in [
            ("premise", ["pooled_test"], figures.ARMS),
            ("temporal", ["source_test"] + [f"year_{y}" for y in range(2020, 2025)], ("M0", "M1")),
        ]:
            for eval_name in eval_names:
                for tier in figures.TIERS:
                    for family in figures.FAMS:
                        for seed in figures.SEEDS:
                            for arm in arms:
                                metrics.append({"experiment": experiment, "eval_set": eval_name,
                                   "tier": tier, "family": family, "seed": seed, "arm": arm,
                                   "pr_auc": .10 + {"M0":0, "M1":.04, "M2":.045}[arm]
                                           + .001*(seed == figures.SEEDS[-1])})
        self.metrics = self.base / "ml.csv"
        pd.DataFrame(metrics).to_csv(self.metrics, index=False)
        annual = []
        for year in figures.YEARS:
            for fam in figures.FAMS:
                annual.append({"year": year, "family": fam, "q_test": .7-(year-2016)*.02,
                               "mean_delta_pr_auc": .015+ .001*(year-2016)+{"LR":0,"XGB":.005,"CAT":.004}[fam]})
        self.annual = self.base / "annual.csv"
        pd.DataFrame(annual).to_csv(self.annual, index=False)

    def call(self, *extra):
        return figures.main(["--gate-csv", str(self.gate), "--ml-csv", str(self.metrics),
                             "--annual-csv", str(self.annual), "--output-dir", str(self.base / "figs"),
                             "--dpi", "60", *extra])

    def test_all_five_figures(self):
        paths = self.call()
        self.assertEqual(len(paths), 5)
        self.assertTrue(all(p.exists() and p.stat().st_size > 500 for p in paths))
        self.assertTrue(all("PARTIAL" not in p.name for p in paths))

    def test_incomplete_years_are_rejected(self):
        d = pd.read_csv(self.gate).iloc[:-1]
        d.to_csv(self.gate, index=False)
        with self.assertRaisesRegex(ValueError, "required records missing"):
            self.call("--figures", "2")
        paths = self.call("--figures", "2", "--allow-partial")
        self.assertIn("PARTIAL", paths[0].name)

    def test_missing_model_seed_is_rejected(self):
        d = pd.read_csv(self.metrics).iloc[:-1]
        d.to_csv(self.metrics, index=False)
        with self.assertRaisesRegex(ValueError, "required records missing"):
            self.call("--figures", "6")

    def test_missing_input_is_not_silently_generated(self):
        with self.assertRaises(FileNotFoundError):
            figures.main(["--figures", "7", "--annual-csv", str(self.base / "missing.csv")])


if __name__ == "__main__":
    unittest.main()
