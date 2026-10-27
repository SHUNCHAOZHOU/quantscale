import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantscale.config import Config
from quantscale.pilot import run_pilot
from quantscale.research import normalize_worker_ids


class PilotFailureTests(unittest.TestCase):
    def test_manager_id_aliases_preserve_tasks_without_changing_content(self):
        self.assertEqual(normalize_worker_ids({"worker_0": "a", "worker_1": "b"}, 2), {"0": "a", "1": "b"})
        self.assertEqual(normalize_worker_ids({"worker_1": "a", "worker_2": "b"}, 2), {"0": "a", "1": "b"})
        self.assertEqual(normalize_worker_ids({"1": "a", "2": "b"}, 2), {"0": "a", "1": "b"})
        malformed = {"worker_0": "a", "worker_3": "b"}
        self.assertEqual(normalize_worker_ids(malformed, 2), malformed)

    def test_missing_submission_does_not_hide_other_condition_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = Config(strategy_mode="rules", backtests=1)
            (root / "config.json").write_text(json.dumps(config.to_dict()))
            attempts = []
            def research(data, settings, output, topology, size, backend):
                output.mkdir()
                attempts.append(topology)
                if topology == "hybrid":
                    raise RuntimeError("fixture no valid strategy")
                (output / "selection.json").write_text(json.dumps({"strategy": {"family": "momentum"}}))
                (output / "frozen.json").write_text(json.dumps({"artifact_sha256": topology}))
                return {"winner": {"validation": {"sharpe": 0.1}, "strategy_id": topology},
                        "process": {"total_tokens": 10, "total_backtests": 1,
                                    "duplicate_variant_rate": 0.0}}
            def evaluation(*args):
                self.assertEqual(len(attempts), 5)
                return {"validation_test_sharpe_gap": 0.2, "excluded_data_sessions": [],
                        "hidden": {"sharpe": -0.1, "annualized_return": -0.01, "max_drawdown": 0.03}}
            with patch("quantscale.pilot.DeepSeekBackend", return_value=object()), \
                 patch("quantscale.pilot.prepare_partitions", return_value=("unused", "unused")), \
                 patch("quantscale.pilot.run_research", side_effect=research), \
                 patch("quantscale.pilot.evaluate", side_effect=evaluation), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_pilot("unused", root / "config.json", root / "pilot", seeds=(7,), workers=1)
            result = json.loads((root / "pilot/completed.json").read_text())
            self.assertEqual(result["planned_runs"], 5)
            self.assertEqual(result["runs"], 4)
            self.assertEqual(len(result["research_failures"]), 1)
            roster = json.loads((root / "pilot/all-strategies-frozen.json").read_text())
            self.assertTrue(roster["all_scheduled_research_attempts_finished"])
            self.assertEqual(len(roster["no_submission"]), 1)


if __name__ == "__main__":
    unittest.main()
