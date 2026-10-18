import json
from pathlib import Path
import tempfile
import unittest

from edison.fixture_trial import ResponseTable, trial, validate_report

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/structural_stalker/cable_v1'


class FixtureTrialTests(unittest.TestCase):
    def test_unknown_resource_and_unlisted_request_do_not_change_state(self):
        table = ResponseTable(FIXTURE)
        self.assertEqual(table.call('set_temperature', {'target_ref': 'V04', 'temperature_C': 40})['status'],
                         'fixture_not_specified')
        self.assertEqual(table.state, 'state-0')
        self.assertNotIn('V04', table.known)
        self.assertEqual(table.calls, 1)

    def test_states_return_distinct_evidence_and_preserve_identity(self):
        table = ResponseTable(FIXTURE)
        table.call('observe', {'target_ref': 'S17', 'method': 'overview', 'parameters': {}})
        self.assertEqual(table.call('pulse_trace', {'target_ref': 'S17'})['record_id'], 'O08')
        table.call('set_temperature', {'target_ref': 'V04', 'temperature_C': 40})
        self.assertEqual(table.call('pulse_trace', {'target_ref': 'S17'})['record_id'], 'O09')
        table.call('set_temperature', {'target_ref': 'V04', 'temperature_C': 20})
        self.assertEqual(table.call('pulse_trace', {'target_ref': 'S17'})['record_id'], 'O10')
        self.assertIn('S17', table.known)

    def test_source_revision_refresh_and_budget_are_enforced(self):
        table = ResponseTable(FIXTURE)
        a = table.call('read', {'resource_ref': 'catalog:root'})
        table.new_source()
        b = table.call('read', {'resource_ref': 'catalog:root'})
        self.assertEqual(a['source_ref'], b['source_ref'])
        self.assertNotEqual(a['source_revision'], b['source_revision'])
        table.call('read', {'resource_ref': 'doc:batch-S17@1'})
        table.call('read', {'resource_ref': 'catalog:root'})
        with self.assertRaises(ValueError):
            table.call('read', {'resource_ref': 'catalog:root'})

    def test_actor_gets_only_public_data_and_base_report_is_frozen_before_new_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            packets = []
            base_bytes = []
            def model(request, review=False):
                packets.append(request)
                if request['phase'] == 'new_source':
                    path = Path(temp) / 'base-report.json'
                    self.assertTrue(path.exists())
                    base_bytes.append(path.read_bytes())
                if review:
                    return {'entities': [], 'assertions': [], 'open_questions': ['What remains unknown?'],
                            'summary': 'No structural claim made.'}
                return {'question': 'Inspect the available catalogue.', 'tool': 'read',
                        'arguments_json': '{"resource_ref":"catalog:root"}'}
            log = trial(FIXTURE, temp, model=model, progress=lambda *a, **kw: None)
            for packet in packets:
                serialized = json.dumps(packet)
                for private in ('world.json', 'expectations.json', 'manual-example', 'w:core', 'oracle_choice'):
                    self.assertNotIn(private, serialized)
                self.assertNotIn('when', packet)
            self.assertEqual(log['status'], 'completed')
            self.assertTrue(all(b == (Path(temp) / 'base-report.json').read_bytes() for b in base_bytes))
            self.assertEqual(len(log['history']), 21)

    def test_report_cannot_cite_unreturned_evidence(self):
        report = {'entities': [{'id': 'x', 'description': 'a part', 'evidence_refs': ['O02']}],
                  'assertions': [], 'open_questions': [], 'summary': 'part'}
        with self.assertRaisesRegex(ValueError, 'unknown evidence'):
            validate_report(report, [], {'S17'})

    def test_failed_base_report_never_opens_new_source(self):
        with tempfile.TemporaryDirectory() as temp:
            phases = []
            def model(request, review=False):
                phases.append(request['phase'])
                if review:
                    return {'entities': [{'id': 'invented', 'description': 'a part', 'evidence_refs': ['O02']}],
                            'assertions': [], 'open_questions': [], 'summary': 'Unsupported.'}
                return {'question': 'Stop.', 'tool': 'finish', 'arguments_json': '{}'}
            log = trial(FIXTURE, temp, model=model, progress=lambda *a, **kw: None)
            self.assertEqual(log['status'], 'partial')
            self.assertEqual(phases, ['base', 'base'])
            self.assertNotIn('new_source', log['reports'])


if __name__ == '__main__':
    unittest.main()
