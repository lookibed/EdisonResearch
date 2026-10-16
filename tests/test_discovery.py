import unittest

from edison.composition import CompositionWorld
from edison.evaluator import EvaluatorProcess
from edison.ideas import Branches, Construction, Primitives
from edison.infopole import InfoField
from edison.pipeline import run
from edison.proposer import Batch, request_for
from edison.scientist import KnowledgeBase, Scientist
from edison.stalker import Stalker


class LocalSource:
    """Same public boundary as the remote evaluator, for deterministic component tests."""
    def __init__(self, field):
        self.field = field

    @property
    def primitives(self):
        return self.field.primitives

    def scan_entities(self, cursor, limit):
        return self.field.scan(cursor, limit)

    def activate_entity(self, token):
        return self.field.activate(token)


class UnknownReverseAdapter:
    """A new adapter absent from the scientist's code and default source."""
    def encoded_length(self, length):
        return length

    def encode(self, bits):
        return list(reversed(bits))

    def decode(self, bits, original_length):
        return list(reversed(bits))[:original_length]


def discover_all(field):
    stalker = Stalker(LocalSource(field))
    while not stalker.finished:
        stalker.poll()
    return stalker


class DiscoveryTests(unittest.TestCase):
    def test_broken_new_decoder_cannot_pass_by_returning_no_bits(self):
        class BrokenAdapter(UnknownReverseAdapter):
            def decode(self, bits, original_length):
                return []
        world = CompositionWorld('combined', 0)
        field = InfoField(world)
        field.publish_transform('broken', BrokenAdapter())
        discover_all(field)
        outcome = world.measure(Construction(transforms=('broken',)), split='probe', messages=32)
        self.assertIsNone(outcome.measurement)
        self.assertEqual(outcome.reason, 'invalid_adapter_output')
        self.assertIn('broken', field.primitives.transforms)

    def test_start_empty_descriptors_do_not_reveal_properties_and_refresh_deduplicates(self):
        field = InfoField(CompositionWorld('combined', 0))
        self.assertEqual(field.primitives, Primitives((), ()))
        first = field.scan()['entries'][0]
        self.assertEqual(set(first), {'id', 'kind', 'interface', 'source', 'revision'})
        with self.assertRaises(ValueError):
            field.probe(first['id'], {'input': [0], 'flip_positions': []})
        stalker = discover_all(field)
        self.assertEqual(len(stalker.entities), 4)
        self.assertEqual(len(field.primitives.nodes), 2)
        self.assertEqual(len(field.primitives.transforms), 2)
        stalker.refresh()
        while not stalker.finished:
            self.assertEqual(stalker.poll(), [])
        self.assertEqual(len(stalker.events), 4)

    def test_new_adapter_after_initial_scan_is_discovered_studied_and_usable(self):
        world = CompositionWorld('combined', 8)
        field = InfoField(world)
        stalker = discover_all(field)
        token = 'previously-absent-adapter'
        field.publish_transform(token, UnknownReverseAdapter())
        self.assertNotIn(token, field.primitives.transforms)
        stalker.refresh()
        found = []
        while not stalker.finished:
            found.extend(stalker.poll())
        self.assertEqual([e['id'] for e in found], [token])
        scientist = Scientist(field.probe)
        self.assertEqual(scientist.study(found[0])['status'], 'complete_on_declared_domain')
        facts = {f['property']: f['value'] for f in scientist.knowledge.facts.values()}
        self.assertEqual(facts['roundtrip_without_noise'], {'recovered': 16, 'tested': 16})
        self.assertFalse(facts['encoded_equals_input'])
        self.assertEqual(facts['single_bit_flip_recovery'], {'recovered': 0, 'tested': 64})
        world._links = {key: (0., False) for key in world._links}
        result = world.measure(Construction(transforms=(token,)), split='holdout', messages=32)
        self.assertEqual(result.measurement.bit_errors, 0)

    def test_late_primitive_grows_old_nodes_without_discarding_pending_branches(self):
        branches = Branches(Primitives(('a',), ()))
        first, second = branches.next(), branches.next()
        pending = set(branches.pending)
        branches.update_primitives(Primitives(('a',), ('new',)))
        self.assertTrue(pending.issubset(branches.pending))
        root_child = next(n for n in branches.graph.nodes if n.construction == Construction(transforms=('new',)))
        old_child = next(n for n in branches.graph.nodes if n.construction == Construction(('a',), ('new',)))
        self.assertTrue(any(e.parent == first.id and e.child == root_child.id for e in branches.graph.edges))
        self.assertTrue(any(e.parent == second.id and e.child == old_child.id for e in branches.graph.edges))
        seen = {branches.next().id for _ in range(20)}
        self.assertTrue({root_child.id, old_child.id}.issubset(seen))
        with self.assertRaises(ValueError):
            branches.update_primitives(Primitives((), ('new',)))

    def test_remote_catalog_gates_access_to_undiscovered_entities(self):
        with EvaluatorProcess('combined', 0, 1, 32, 128, discovery=True) as evaluator:
            self.assertEqual(evaluator.primitives, Primitives((), ()))
            entry = evaluator.scan_entities()['entries'][0]
            c = Construction(path=(entry['id'],)) if entry['kind'] == 'node' else Construction(transforms=(entry['id'],))
            with self.assertRaises(RuntimeError):
                evaluator.probe(c)
            evaluator.activate_entity(entry['id'])
            evaluator.probe(c)
            report = evaluator.finish()
            self.assertEqual(len(report['observations']), 1)


class ScientistTests(unittest.TestCase):
    def setUp(self):
        self.world = CompositionWorld('combined', 0)
        self.field = InfoField(self.world)
        self.stalker = discover_all(self.field)

    def study(self, token):
        scientist = Scientist(self.field.probe)
        entry = next(e for e in self.stalker.entities.values() if e['id'] == token)
        scientist.study(entry)
        return scientist, entry

    def test_properties_are_measured_and_every_fact_has_replayable_evidence(self):
        token = next(t for t, mode in self.world._transforms.items() if mode == 'hamming')
        scientist, entry = self.study(token)
        facts = {f['property']: f['value'] for f in scientist.knowledge.facts.values()}
        self.assertEqual(facts['encoded_lengths_for_4_bits'], [7])
        self.assertEqual(facts['single_bit_flip_recovery'], {'recovered': 112, 'tested': 112})
        before = self.field.accounting()['unique_probes']
        self.assertEqual(scientist.study(entry)['status'], 'already_studied')
        self.assertEqual(self.field.accounting()['unique_probes'], before)
        for fact in scientist.knowledge.facts.values():
            self.assertTrue(fact['scope'])
            for evidence_id in fact['evidence_ids']:
                observation = scientist.knowledge.evidence[evidence_id]
                self.assertEqual(observation, self.field.probe(token, observation['experiment']))

    def test_node_observations_distinguish_noise_from_dominant_inversion(self):
        token = next(n for n in self.world.primitives.nodes if self.world._links['source', n][0] == .02)
        scientist, _ = self.study(token)
        facts = {f['property']: f['value'] for f in scientist.knowledge.facts.values()}
        incoming, outgoing = facts['incoming_link_observation'], facts['outgoing_link_observation']
        self.assertEqual(incoming['dominant_mapping'], 'preserving')
        self.assertEqual(outgoing['dominant_mapping'], 'inverted')
        self.assertGreater(outgoing['wilson_95'][0], .5)
        self.assertEqual(incoming['bits'], 1024)
        evidence = list(scientist.knowledge.evidence.values())
        self.assertEqual(len(evidence), 4)
        self.assertNotEqual(evidence[0]['evidence_id'], evidence[1]['evidence_id'])

    def test_budget_exhaustion_does_not_create_unmeasured_recovery_claim(self):
        self.field.probe_budget = 24
        token = next(t for t, mode in self.world._transforms.items() if mode == 'hamming')
        scientist, entry = self.study(token)
        self.assertEqual(scientist.knowledge.studies[-1]['status'], 'deferred')
        self.assertNotIn('single_bit_flip_recovery', {f['property'] for f in scientist.knowledge.facts.values()})
        count = len(scientist.knowledge.evidence)
        scientist.study(entry)
        self.assertEqual(len(scientist.knowledge.evidence), count)
        self.field.probe_budget = 512
        self.assertEqual(scientist.study(entry)['status'], 'complete_on_declared_domain')
        self.assertIn('single_bit_flip_recovery', {f['property'] for f in scientist.knowledge.facts.values()})

    def test_facts_without_evidence_are_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeBase().add_fact({'id': 'x'}, 'unmeasured', True, 'any', [])

    def test_saved_knowledge_avoids_repeat_probes_and_rejects_wrong_context(self):
        token = next(t for t, mode in self.world._transforms.items() if mode == 'hamming')
        scientist, entry = self.study(token)
        loaded = KnowledgeBase.from_report(scientist.knowledge.report())
        def forbidden_probe(*args):
            self.fail('A completed study must not repeat component experiments')
        restored = Scientist(forbidden_probe, loaded)
        self.assertEqual(restored.study(entry)['status'], 'already_studied')
        evidence_id = next(iter(loaded.evidence))
        with self.assertRaises(ValueError):
            loaded.add_fact(dict(entry, revision='different'), 'wrong', True, 'any', [evidence_id])


class RecordingProposer:
    def __init__(self):
        self.requests = []

    def propose(self, parent, primitives):
        self.requests.append(request_for(parent, primitives))
        return Batch((), {})


class DiscoveryPipelineTests(unittest.TestCase):
    def test_full_pipeline_starts_empty_learns_properties_and_solves(self):
        proposer = RecordingProposer()
        report = run('combined', 0, 192, proposer, 4, 512, 4096, discovery=True)
        d = report['discovery']
        self.assertEqual(d['initial_catalog'], {'nodes': (), 'transforms': ()})
        self.assertEqual(len(d['entities']), 4)
        self.assertEqual([e['before_trial'] for e in d['events']], [1, 3, 5, 7])
        self.assertEqual(report['component_accounting']['unique_probes'], 218)
        self.assertEqual(len(d['knowledge']['facts']), 14)
        self.assertTrue(report['success'])
        self.assertTrue(report['graph']['pending'])
        self.assertFalse(d['knowledge_sent_to_generator'])
        self.assertEqual(len(proposer.requests), 4)
        for request in proposer.requests:
            self.assertNotIn('knowledge', request)
            self.assertNotIn('facts', request)
            self.assertNotIn('observations', request)
        self.assertTrue(any(o.get('component_fact_refs') for o in report['observations']))

    def test_no_component_budget_keeps_entities_and_search_running(self):
        report = run('combined', 2, 10, probe_messages=32, holdout_messages=128,
                     discovery=True, scientist_budget=0)
        self.assertEqual(len(report['discovery']['entities']), 4)
        self.assertEqual(report['discovery']['knowledge']['facts'], [])
        self.assertTrue(all(s['status'] == 'deferred' for s in report['discovery']['knowledge']['studies']))
        self.assertEqual(len(report['observations']), 10)


if __name__ == '__main__':
    unittest.main()
