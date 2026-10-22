from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edison.structural_graph import ResearchGraph, empty_delta
from edison.structural_trial import FixtureSource, evidence_index, replay_session, run

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/structural_stalker/cable_v2'


def reply(packet, tool='finish', arguments=None, delta=None):
    return {'graph_delta': delta or empty_delta(), 'question_id': packet.get('required_question_id') or 'QROOT',
            'question': 'Investigate the public task.', 'tool': tool,
            'arguments_json': json.dumps(arguments or {}), 'summary': 'No completeness claim.'}


class StructuralGraphTests(unittest.TestCase):
    def setUp(self):
        self.source = FixtureSource(FIXTURE)
        self.graph = ResearchGraph(self.source.start)
        self.history = [self.source.call('observe', {
            'target_ref': 'S17', 'method': 'radial_profile', 'parameters': {'x_mm': 0}})]
        self.evidence = evidence_index(self.source.start, self.history, [])

    def apply(self, delta):
        self.graph.apply(delta, evidence=self.evidence, known_refs=self.source.known, turn=1)

    def structural_delta(self):
        delta = empty_delta()
        delta['entities'] = [{'id': 'candidate-region', 'description': 'A spatially delimited region.',
                             'identity_basis': 'Selected intervals of a returned profile.',
                             'evidence_refs': ['O02']}]
        delta['relation_types'] = [{'id': 'part_of', 'description': 'A candidate physical part of a whole.'}]
        delta['assertions'] = [{'id': 'a1', 'statement': 'A candidate region belongs to S17.',
            'relation': 'part_of', 'participants': [{'role': 'part', 'entity_id': 'candidate-region'},
                {'role': 'whole', 'entity_id': 'S17'}], 'evidence_refs': ['O02'],
            'origin': 'hypothesis', 'scope': 'Only the measured section, not the whole length.',
            'state_refs': ['state-0'], 'supersedes': []}]
        return delta

    def test_delta_can_be_empty_and_unknown_entities_need_no_executable_capability(self):
        self.apply(empty_delta())
        self.apply(self.structural_delta())
        self.assertIn('candidate-region', self.graph.data['entities'])
        self.assertNotIn('candidate-region', self.source.known)
        self.assertEqual(self.graph.data['revision'], 2)

    def test_corrections_and_conflicting_decompositions_preserve_previous_records(self):
        self.apply(self.structural_delta())
        delta = empty_delta()
        claim = deepcopy(self.graph.data['assertions']['a1'])
        claim.update(id='a2', statement='A revised decomposition remains a hypothesis.', supersedes=['a1'])
        delta['assertions'] = [claim]
        self.apply(delta)
        self.assertEqual(list(self.graph.data['assertions']), ['a1', 'a2'])
        self.assertEqual(self.graph.data['assertions']['a1']['supersedes'], [])

    def test_redefinitions_and_unknown_evidence_are_rejected_atomically(self):
        self.apply(self.structural_delta())
        for change in ('description', 'evidence'):
            original = deepcopy(self.graph.data)
            delta = self.structural_delta()
            if change == 'description':
                delta['entities'][0]['description'] = 'Different identity under an old ID.'
            else:
                delta['assertions'][0]['evidence_refs'] = ['O10']
            with self.assertRaises(ValueError):
                self.apply(delta)
            self.assertEqual(self.graph.data, original)

    def test_undeclared_participants_relations_and_unobserved_states_are_rejected(self):
        for field, value in (('state_refs', ['state-1']), ('relation', 'undefined_relation')):
            delta = self.structural_delta()
            delta['assertions'][0][field] = value
            with self.assertRaises(ValueError):
                self.apply(delta)
        delta = self.structural_delta()
        delta['assertions'][0]['participants'][0]['entity_id'] = 'secret-part'
        with self.assertRaises(ValueError):
            self.apply(delta)

    def test_document_is_not_promoted_to_direct_observation(self):
        self.source.call('read', {'resource_ref': 'catalog:root'})
        self.history.append(self.source.call('read', {'resource_ref': 'doc:model-M7@1'}))
        self.evidence = evidence_index(self.source.start, self.history, [])
        delta = self.structural_delta()
        delta['assertions'][0].update(evidence_refs=['D03'], origin='observation')
        with self.assertRaisesRegex(ValueError, 'not a physical observation'):
            self.apply(delta)
        delta['assertions'][0]['origin'] = 'source_report'
        self.apply(delta)

    def test_call_id_can_cite_its_returned_instrument_record(self):
        delta = self.structural_delta()
        delta['assertions'][0].update(evidence_refs=['C0001'], origin='observation')
        self.apply(delta)
        self.assertEqual(self.graph.data['assertions']['a1']['origin'], 'observation')

    def test_questions_branch_reopen_and_keep_fifo_without_erasing_blocked_work(self):
        delta = empty_delta()
        delta['questions'] = [{'id': 'q-child', 'text': 'What explains the profile?', 'targets': ['S17'],
                               'parent_ids': ['QROOT'], 'evidence_refs': ['O02']}]
        delta['question_updates'] = [{'id': 'u1', 'question_id': 'QROOT', 'status': 'blocked',
                                      'reason': 'Needs additional data.', 'evidence_refs': ['O02']}]
        self.apply(delta)
        self.assertEqual(self.graph.fifo_question(), 'q-child')
        delta = empty_delta()
        delta['question_updates'] = [{'id': 'u2', 'question_id': 'QROOT', 'status': 'open',
                                      'reason': 'Revisit after evidence.', 'evidence_refs': ['O02']}]
        self.apply(delta)
        self.assertEqual(self.graph.fifo_question(), 'q-child')  # Umbrella does not monopolize FIFO.
        self.assertEqual(self.graph.question_statuses()['QROOT'], 'open')
        self.assertEqual(len(self.graph.data['question_updates']), 2)

    def test_question_cycles_and_cross_type_ids_are_rejected(self):
        delta = empty_delta()
        delta['questions'] = [{'id': 'cycle', 'text': 'Question', 'targets': ['S17'],
                               'parent_ids': ['cycle'], 'evidence_refs': []}]
        with self.assertRaisesRegex(ValueError, 'cycle'):
            self.apply(delta)
        delta = self.structural_delta()
        delta['assertions'][0]['id'] = 'candidate-region'
        with self.assertRaises(ValueError):
            self.apply(delta)

    def test_ablation_withholds_claim_content_but_keeps_identity_and_schedule(self):
        self.apply(self.structural_delta())
        view = self.graph.view('registry')
        self.assertEqual(view['assertion_ids'], ['a1'])
        self.assertNotIn('assertions', view)
        self.assertNotIn('A candidate region belongs to S17.', json.dumps(view))
        self.assertIn('candidate-region', view['entities'])
        self.assertEqual(view['questions'], self.graph.view()['questions'])


class StructuralTrialTests(unittest.TestCase):
    def test_contract_reads_all_published_material_references(self):
        source = FixtureSource(FIXTURE)
        source.call('read', {'resource_ref': 'catalog:root'})
        response = source.call('read', {'resource_ref': 'doc:references@1'})['response']
        for material in response['payload']['references']:
            self.assertEqual(source.call('read', {'resource_ref': material['ref']})['response']['status'], 'ok')

    def test_attempt_ids_are_unique_even_for_errors_and_repeated_records(self):
        source = FixtureSource(FIXTURE)
        arguments = {'target_ref': 'S17'}
        first = source.call('pulse_trace', arguments)
        repeat = source.call('pulse_trace', arguments)
        failure = source.call('read', {'resource_ref': ['bad type']})
        self.assertEqual(first['response'], repeat['response'])
        self.assertEqual([first['call_id'], repeat['call_id'], failure['call_id']], ['C0001', 'C0002', 'C0003'])
        self.assertEqual(failure['response']['status'], 'invalid_arguments')
        self.assertEqual(source.calls, 3)

    def test_hot_observation_is_explicitly_unavailable_and_return_has_new_record(self):
        source = FixtureSource(FIXTURE)
        arguments = {'target_ref': 'S17', 'method': 'overview', 'parameters': {}}
        cold = source.call('observe', arguments)
        source.call('set_temperature', {'target_ref': 'V04', 'temperature_C': 40})
        hot = source.call('observe', arguments)
        self.assertEqual(hot['response']['status'], 'state_not_available')
        source.call('set_temperature', {'target_ref': 'V04', 'temperature_C': 20})
        returned = source.call('observe', arguments)
        self.assertNotEqual(cold['response']['record_id'], returned['response']['record_id'])
        self.assertEqual(returned['response']['payload']['context']['state_ref'], 'state-2')

    def test_transfer_controls_change_raw_measurements_without_telling_actor_part_count(self):
        results = []
        for name in ('cable_v2', 'homogeneous_v2', 'shifted_v2'):
            source = FixtureSource(ROOT / 'fixtures/structural_stalker' / name)
            results.append(source.call('observe', {'target_ref': 'S17', 'method': 'radial_profile',
                                                  'parameters': {'x_mm': 0}})['response']['payload']['contrast_units'])
        self.assertEqual(len({tuple(x) for x in results}), 3)
        self.assertEqual(len(set(results[1][:9])), 1)

    def test_runtime_reads_only_two_data_files_and_freezes_base_before_later_source(self):
        reads, packets, frozen = [], [], []
        original_text, original_bytes = Path.read_text, Path.read_bytes
        def read_text(path, *a, **kw):
            reads.append(path)
            return original_text(path, *a, **kw)
        def read_bytes(path, *a, **kw):
            reads.append(path)
            return original_bytes(path, *a, **kw)
        with tempfile.TemporaryDirectory() as temp:
            def model(packet):
                packets.append(deepcopy(packet))
                if packet['phase'] == 'new_source':
                    frozen.append(original_bytes(Path(temp) / 'base-graph.json'))
                return reply(packet)
            with patch.object(Path, 'read_text', read_text), patch.object(Path, 'read_bytes', read_bytes):
                log = run(FIXTURE, temp, model=model, progress=lambda *a, **kw: None)
            self.assertEqual(set(reads), {FIXTURE / 'public/start.json', FIXTURE / 'private/responses.json'})
            self.assertEqual(log['status'], 'completed')
            self.assertTrue(all(b == (Path(temp) / 'base-graph.json').read_bytes() for b in frozen))
            self.assertNotIn('expectations.json', json.dumps(packets))
            self.assertNotIn('world.json', json.dumps(packets))
            with self.assertRaises(ValueError):
                run(FIXTURE, temp, model=model)

    def test_rejected_update_does_not_execute_tool_or_corrupt_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            def model(packet):
                delta = empty_delta()
                delta['entities'] = [{'id': 'bad', 'description': 'Nonexistent evidence.',
                                      'identity_basis': 'invented', 'evidence_refs': ['secret']}]
                return reply(packet, 'pulse_trace', {'target_ref': 'S17'}, delta)
            log = run(FIXTURE, temp, model=model, progress=lambda *a, **kw: None)
            self.assertEqual(log['status'], 'partial')
            self.assertEqual(log['history'], [])
            graph = json.loads((Path(temp) / 'graph.json').read_text())
            self.assertNotIn('bad', graph['entities'])
            self.assertEqual(len(log['attempts']), 5)  # Four rejected actions, one consolidation.

    def test_budget_bound_and_final_delta_processes_last_tool_result(self):
        with tempfile.TemporaryDirectory() as temp:
            def model(packet):
                if packet['consolidation_only']:
                    delta = empty_delta()
                    delta['questions'] = [{'id': 'q-' + packet['phase'], 'text': 'What remains?',
                        'targets': ['S17'], 'parent_ids': ['QROOT'],
                        'evidence_refs': [packet['history'][-1]['call_id']]}]
                    return reply(packet, delta=delta)
                return reply(packet, 'pulse_trace', {'target_ref': 'S17'})
            log = run(FIXTURE, temp, model=model, progress=lambda *a, **kw: None)
            self.assertEqual(len(log['history']), 21)
            self.assertEqual(len(log['attempts']), 23)
            graph = json.loads((Path(temp) / 'graph.json').read_text())
            self.assertEqual(graph['questions']['q-new_source']['evidence_refs'], ['C0021'])
            self.assertEqual(log['status'], 'completed')
            self.assertEqual(replay_session(json.loads(json.dumps(log))).data, graph)


if __name__ == '__main__':
    unittest.main()
