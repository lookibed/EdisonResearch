import json
import os
import unittest

from edison.experiments import ExperimentExecutor
from edison.infopole import InfoField
from edison.composition import CompositionWorld
from edison.investigator import ResearchScientist, validate_response
from edison.lab import LabBackend
from edison.llm_scientist import LLMComponentScientist
from edison.scientist import KnowledgeBase


def hypotheses():
    return [{'id': 'a', 'statement': 'Order matters', 'prediction': 'Reversed inputs differ',
             'falsification': 'A tested pair agrees'},
            {'id': 'b', 'statement': 'Order does not matter', 'prediction': 'A permutation agrees',
             'falsification': 'A tested pair differs'}]


class ScriptedModel:
    def __init__(self, programs=None):
        self.programs = programs or [
            "for values in [[1,2],[2,1]]:\n    emit(call('evaluate', {'values': values})['result'])",
            "emit(call('evaluate', {'values': [1,1]})['result'])"]
        self.requests = []
        self.index = 0

    def respond(self, request, review=False):
        self.requests.append(request)
        claims = []
        if request['observations']:
            claims = [{'claim': 'Recorded responses were obtained', 'evidence_ids': ['E1'],
                       'scope': 'Only the recorded inputs', 'uncertainties': ['Untested inputs']}]
        if review:
            return {'interpretations': claims, 'remaining_unknowns': ['General law'],
                    'suggested_next_question': 'Which new input could falsify this?'}
        program = self.programs[self.index % len(self.programs)]
        self.index += 1
        return {'question': 'Does permutation or duplication change the output?',
                'hypotheses': hypotheses(), 'experiment': program, 'interpretations': claims}


class ProtocolTests(unittest.TestCase):
    def test_interpretation_cannot_invent_evidence(self):
        response = {'question': 'question', 'hypotheses': hypotheses(), 'experiment': 'emit(1)',
                    'interpretations': [{'claim': 'unsupported', 'evidence_ids': ['E99'],
                                         'scope': 'some', 'uncertainties': []}]}
        with self.assertRaises(ValueError):
            validate_response(response, {'E1': 'real-observation'})
        response['interpretations'][0]['evidence_ids'] = ['E1']
        claims = validate_response(response, {'E1': 'real-observation'})
        self.assertEqual(claims[0]['status'], 'llm_interpretation_unverified')
        self.assertEqual(claims[0]['evidence_ids'], ['real-observation'])

    def test_domains_expose_interfaces_not_hidden_laws(self):
        for domain in LabBackend.DOMAINS:
            backend = LabBackend(domain, 0)
            self.assertEqual(set(backend.entity), {'id', 'source', 'revision', 'interface'})
            text = json.dumps(backend.tools).lower()
            for private_name in ('hamming', 'mass', 'drag', 'sum of squares'):
                self.assertNotIn(private_name, text)

    def test_transport_adapters_record_actual_distinct_behaviors(self):
        a, b = LabBackend('function', 0), LabBackend('function', 1)
        self.assertNotEqual(a.invoke('evaluate', {'values': [2, 3]})['result'],
                            b.invoke('evaluate', {'values': [2, 3]})['result'])
        physical = LabBackend('physics', 0)
        positive = physical.invoke('observe', {'force': 1, 'duration': 1})['result']
        negative = physical.invoke('observe', {'force': -1, 'duration': 1})['result']
        self.assertAlmostEqual(positive['position'], -negative['position'])


@unittest.skipUnless(os.environ.get('EDISON_TEST_SANDBOX') == '1',
                     'Set EDISON_TEST_SANDBOX=1 in a host supporting bubblewrap user namespaces')
class IsolatedResearchTests(unittest.TestCase):
    def test_rpc_and_program_computation(self):
        backend = LabBackend('function', 0)
        outcome = ExperimentExecutor().run(
            "r=call('evaluate', {'values':[2,3]}); emit({'twice':r['result']['value']*2})",
            backend.tools, backend.invoke)
        self.assertEqual(outcome['status'], 'ok')
        self.assertEqual(outcome['emissions'], [{'twice': 26}])
        self.assertEqual(len(outcome['evidence']), 1)

    def test_no_secrets_project_files_network_or_child_processes(self):
        code = """import os, socket
result = {'secret': 'OPENROUTER_API_KEY' in os.environ}
try:
    open('/workspace/EdisonResearch/README.md').read()
    result['project_readable'] = True
except OSError:
    result['project_readable'] = False
try:
    os.fork()
    result['fork_allowed'] = True
except OSError:
    result['fork_allowed'] = False
try:
    socket.create_connection(('198.51.100.1', 443), timeout=.2)
    result['network'] = True
except OSError:
    result['network'] = False
emit(result)
"""
        outcome = ExperimentExecutor().run(code, [], lambda *_: None)
        self.assertEqual(outcome['status'], 'ok')
        self.assertEqual(outcome['emissions'], [{'secret': False, 'project_readable': False,
                                               'fork_allowed': False, 'network': False}])

    def test_timeout_and_tool_budget(self):
        result = ExperimentExecutor(timeout=.4).run('while True: pass', [], lambda *_: None)
        self.assertEqual(result['status'], 'timeout')
        backend = LabBackend('function', 0)
        result = ExperimentExecutor(max_calls=2).run(
            "for i in range(5): call('evaluate', {'values':[i]})", backend.tools, backend.invoke)
        self.assertEqual(result['status'], 'tool_budget_exhausted')
        self.assertEqual(len(result['evidence']), 2)

    def test_emissions_cannot_forge_authoritative_evidence(self):
        result = ExperimentExecutor().run("emit({'evidence_id':'forged','result':42})", [], lambda *_: None)
        self.assertEqual(result['evidence'], [])
        self.assertEqual(len(result['emissions']), 1)

    def test_model_gets_feedback_revises_and_keeps_interpretations_separate(self):
        backend, model = LabBackend('function', 0), ScriptedModel()
        report = ResearchScientist(model=model).investigate(backend.entity, backend.tools, backend.invoke)
        self.assertEqual(report['status'], 'observed')
        self.assertEqual(len(report['evidence']), 3)
        self.assertEqual(len(model.requests[1]['observations']), 2)
        self.assertEqual(len(model.requests[2]['observations']), 3)
        self.assertTrue(all(s['execution_status'] == 'ok' for s in report['rounds']))
        self.assertTrue(all(c['status'] == 'llm_interpretation_unverified' for c in report['interpretations']))
        self.assertFalse(report['claims_automatically_verified'])

    def test_existing_evidence_is_reused_without_tool_invocation(self):
        backend = LabBackend('function', 0)
        first = ResearchScientist(model=ScriptedModel()).investigate(backend.entity, backend.tools, backend.invoke)
        def forbidden(*_):
            self.fail('Previously observed requests should not be executed again')
        second = ResearchScientist(model=ScriptedModel()).investigate(
            backend.entity, backend.tools, forbidden, initial_evidence=first['evidence'])
        self.assertEqual(second['unique_tool_attempts'], 0)
        self.assertEqual(second['duplicate_requests'], 3)
        self.assertEqual(len(second['evidence']), 3)

    def test_component_scientist_uses_general_loop_and_serializes_real_evidence(self):
        world = CompositionWorld('combined', 0)
        field = InfoField(world)
        token = world.primitives.transforms[0]
        field.activate(token)
        entity = next(e for e in field.scan(0, 16)['entries'] if e['id'] == token)
        programs = [
            "emit(call('probe', {'input':[1,0,1,1], 'flip_positions':[]}))",
            "emit(call('probe', {'input':[1,0,1,1], 'flip_positions':[2]}))"]
        researcher = ResearchScientist(model=ScriptedModel(programs))
        scientist = LLMComponentScientist(field.probe, researcher=researcher)
        scientist.study(entity)
        self.assertEqual(len(scientist.knowledge.evidence), 2)
        self.assertEqual({f['property'] for f in scientist.knowledge.facts.values()}, {'observed_response'})
        self.assertTrue(scientist.knowledge.interpretations)
        loaded = KnowledgeBase.from_report(scientist.knowledge.report())
        self.assertEqual(len(loaded.evidence), 2)
        self.assertTrue(all(c['status'] == 'llm_interpretation_unverified' for c in loaded.interpretations))


if __name__ == '__main__':
    unittest.main()
