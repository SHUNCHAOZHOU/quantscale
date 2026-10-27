import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from quantscale.config import Config, Period
from quantscale.data import synthetic_data
from quantscale.manager_protocol import (ManagerProtocolError, parse_manager,
                                         repair_reference, strict_json_object)
from quantscale.research import run_research


def broken(content='{"directions":{"0":"momentum" "1":"reversal"}}'):
    return dict(adapter_error='missing comma', adapter_error_kind='json_parse',
                finish_reason='stop', raw_content=content)


class ProtocolTests(unittest.TestCase):
    def test_shapes_preserve_assignments_and_shared_constraints(self):
        parsed, actions = parse_manager({'worker_1': 'alpha', 'worker_2': 'beta'}, 2)
        self.assertEqual(parsed, {'0': 'alpha', '1': 'beta'})
        self.assertIn('added_directions_wrapper', actions)
        parsed, actions = parse_manager({'directions': {
            '0': {'assignment': 'alpha', 'constraints_note': 'keep turnover low', 'theme': 'liquidity'},
            '1': {'task': 'beta'}, 'cross_cutting_guidance': {'risk': 'avoid concentration'}},
            'type': 'json_object'}, 2)
        self.assertIn('keep turnover low', parsed['0'])
        self.assertIn('liquidity', parsed['0'])
        self.assertIn('avoid concentration', parsed['0'])
        self.assertIn('avoid concentration', parsed['1'])
        self.assertIn('preserved_shared_guidance', actions)

    def test_ambiguities_are_not_guessed(self):
        bad = [{'directions': {'0': 'a'}}, {'0': 'a', 'worker_0': 'b'},
               {'0': 'a', '2': 'b'}, {'0': '', '1': 'b'},
               {'0': {'assignment': 'a', 'task': 'different'}, '1': 'b'},
               {'directions': {'0': 'a', '1': 'b'}, '0': 'different'},
               {'0': {'assignment': 'a', 'unknown': 'b'}, '1': 'c'},
               {0: 'a', 1: 'b'}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ManagerProtocolError):
                parse_manager(value, 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            strict_json_object('{"0":"a","0":"b"}')

    def test_repair_only_has_a_reference_for_complete_verbatim_tasks(self):
        self.assertEqual(repair_reference(broken(), 2), {'0': 'momentum', '1': 'reversal'})
        for value in [broken('{"0":"a"}'), broken('{"0":"a","0":"b","1":"c"}'),
                      broken('{"0":"a" "1":"b","cross_cutting_guidance":"keep risk low"}'),
                      dict(broken(), finish_reason='length'), dict(broken(), adapter_error_kind='incomplete')]:
            self.assertIsNone(repair_reference(value, 2))


class ScriptedBackend:
    name = 'protocol-fixture'
    token_accounting = 'fixture: 50 tokens per call, no inference'

    def __init__(self, managers):
        self.managers = iter(managers)
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        if request['role'] == 'manager':
            return next(self.managers), 50
        return {'hypothesis': 'fixture momentum', 'strategy': {
            'family': 'momentum', 'lookback': 5, 'skip': 0,
            'fraction': .5, 'volatility_scale': False}}, 50


class RepairIntegrationTests(unittest.TestCase):
    def run_fixture(self, managers, backtests=2, tokens_override=10000, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        data = root / 'research.csv'
        synthetic_data('2018-01-01', '2018-03-30', 8).to_csv(data, index=False)
        config = Config(strategy_mode='rules', universe_mode='fixed', universe_size=8,
                        train=Period('2018-01-01', '2018-02-28'),
                        validation=Period('2018-03-01', '2018-03-30'),
                        test=Period('2018-04-01', '2018-04-30'),
                        min_price=0, min_volume=0, max_weight=.25,
                        backtests=backtests, tokens=tokens_override, **overrides)
        backend = ScriptedBackend(managers)
        with contextlib.redirect_stderr(io.StringIO()):
            result = run_research(data, config, root / 'run', 'centralized', 2, backend)
        events = [json.loads(x) for x in (root / 'run/trajectory.jsonl').read_text().splitlines()]
        return result, backend, events

    def test_repair_is_logged_charged_and_does_not_consume_candidate_quota(self):
        result, backend, events = self.run_fixture([broken(), {'directions': {'0': 'momentum', '1': 'reversal'}}])
        self.assertEqual(result['process']['total_tokens'], 200)
        self.assertEqual(result['process']['total_backtests'], 2)
        self.assertEqual(result['process']['communication_calls'], 2)
        self.assertEqual(backend.requests[1]['format_repair']['original_response'], broken()['raw_content'])
        broadcast = next(x for x in events if x['kind'] == 'manager_broadcast')
        self.assertEqual(broadcast['format_repair_attempts'], 1)

    def test_changing_tasks_during_repair_is_rejected_then_next_round_can_recover(self):
        result, backend, events = self.run_fixture([
            broken(), {'directions': {'0': 'reversal', '1': 'momentum'}},
            {'0': 'momentum', '1': 'reversal'}])
        self.assertEqual(result['process']['total_tokens'], 250)
        self.assertTrue(any('altered' in x.get('error', '') for x in events))
        self.assertEqual(sum(x['kind'] == 'manager_round_failed' for x in events), 1)

    def test_exhausted_manager_formats_keep_earlier_valid_candidates(self):
        result, backend, events = self.run_fixture([
            {'0': 'momentum', '1': 'reversal'}, {}, {}], backtests=4)
        self.assertEqual(result['process']['total_backtests'], 2)
        self.assertEqual(result['process']['total_tokens'], 250)
        self.assertTrue(any(x['kind'] == 'manager_protocol_exhausted' for x in events))
        self.assertEqual(events[-1]['kind'], 'frozen')

    def test_zero_repairs_and_consecutive_failure_limit_are_enforced(self):
        result, backend, events = self.run_fixture([
            {'0': 'momentum', '1': 'reversal'}, broken()], backtests=4,
            manager_max_format_repairs=0, manager_max_failed_rounds=1)
        self.assertEqual(len(backend.requests), 4)
        self.assertEqual(result['process']['total_tokens'], 200)
        self.assertFalse(any('format_repair' in r for r in backend.requests))

    def test_no_budget_means_no_unpaid_format_repair_or_candidate(self):
        with self.assertRaisesRegex(RuntimeError, 'No valid strategy'):
            # First response consumes the full global allowance. invoke refuses
            # the repair locally, before entering the provider fixture again.
            self.run_fixture([broken()], tokens_override=50)


if __name__ == '__main__':
    unittest.main()
