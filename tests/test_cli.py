import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from quantscale.cli import main
from quantscale.research import CommandBackend


class CliTests(unittest.TestCase):
    def test_command_adapter_round_trip(self):
        script = ('import json,sys; r=json.load(sys.stdin); '
                  'print(json.dumps({"response":{"echo":r["agent"]},'
                  '"usage":{"total_tokens":11},"provenance":{"model":"fixture"}}))')
        response, tokens = CommandBackend([sys.executable, "-c", script]).invoke({"agent": 3})
        self.assertEqual(tokens, 11)
        self.assertEqual(response["echo"], 3)
        self.assertEqual(response["_provenance"]["model"], "fixture")

    def test_matrix_budget_modes_and_single_size(self):
        from unittest.mock import patch
        from quantscale.config import Config
        for mode in ("fixed-total", "fixed-per-agent"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = Config()
                (root / "config.json").write_text(json.dumps(config.to_dict()))
                (root / "matrix.json").write_text(json.dumps({"budget_mode": mode,
                    "topologies": ["single", "independent"], "fleet_sizes": [2, 4],
                    "backtest_budgets": [3], "seeds": [7]}))
                observed = []
                def research(data, settings, output, topology, size, backend, budget_mode):
                    observed.append((topology, size, settings.backtests, settings.tokens, budget_mode))
                    return {"winner": {"validation": {"sharpe": 0}}, "process": {}}
                with patch("quantscale.cli.run_research", side_effect=research):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        main(["scaling", "--data", "unused.csv", "--config", str(root / "config.json"),
                              "--matrix", str(root / "matrix.json"), "--output", str(root / "output"), "--backend", "demo"])
                self.assertEqual(len(observed), 3)
                self.assertEqual(observed[0][:3], ("single", 1, 3))
                self.assertEqual(observed[2][2], 12 if mode == "fixed-per-agent" else 3)
                self.assertEqual(observed[2][3], config.tokens * (4 if mode == "fixed-per-agent" else 1))
                self.assertTrue((root / "output/research_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
