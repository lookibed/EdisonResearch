"""Black-box component experiments. No access to simulator classes or private truth."""
import hashlib
from itertools import product
import json
import math


class StudyPaused(Exception):
    pass


class KnowledgeBase:
    def __init__(self):
        self.evidence = {}
        self.facts = {}
        self.studies = []

    def add_fact(self, entity, property_name, value, scope, evidence_ids):
        if not evidence_ids or any(i not in self.evidence for i in evidence_ids):
            raise ValueError('A fact requires recorded experimental evidence')
        context = (entity['id'], entity['source'], entity['revision'])
        if any((self.evidence[i]['entity'], self.evidence[i]['source'], self.evidence[i]['revision']) != context
               for i in evidence_ids):
            raise ValueError('Evidence belongs to a different entity or source revision')
        record = {'entity': entity['id'], 'source': entity['source'], 'revision': entity['revision'],
                  'property': property_name, 'value': value, 'scope': scope,
                  'status': 'supported_on_tested_domain', 'evidence_ids': list(evidence_ids)}
        key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        self.facts[key] = dict(record, id=key)
        return key

    def references_for(self, construction, contexts=None):
        entities = set(construction.path + construction.transforms)
        return [key for key, fact in self.facts.items() if fact['entity'] in entities
                and (contexts is None or (fact['entity'], fact['source'], fact['revision']) in contexts)]

    def report(self):
        return {'facts': list(self.facts.values()), 'evidence': list(self.evidence.values()),
                'studies': list(self.studies)}

    @classmethod
    def from_report(cls, report):
        result = cls()
        for observation in report['evidence']:
            if observation['status'] != 'ok':
                raise ValueError('Knowledge contains an unsuccessful observation')
            result.evidence[observation['evidence_id']] = observation
        for fact in report['facts']:
            entity = {'id': fact['entity'], 'source': fact['source'], 'revision': fact['revision']}
            key = result.add_fact(entity, fact['property'], fact['value'], fact['scope'], fact['evidence_ids'])
            if key != fact['id']:
                raise ValueError('Fact integrity check failed')
        result.studies = list(report['studies'])
        return result


def wilson_interval(errors, trials):
    """95% Wilson interval for a binomial mismatch fraction, not a universal law."""
    z = 1.959963984540054
    p = errors / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return [max(0., center - radius), min(1., center + radius)]


class Scientist:
    def __init__(self, probe_entity, knowledge=None):
        self.probe_entity = probe_entity
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()
        self._cache = {json.dumps([o['source'], o['revision'], o['entity'], o['experiment']], sort_keys=True): o
                       for o in self.knowledge.evidence.values()}
        self._complete = {(s['source'], s['revision'], s['entity']) for s in self.knowledge.studies
                          if s['status'] == 'complete_on_declared_domain'}

    def _observe(self, entity, experiment):
        key = json.dumps([entity['source'], entity['revision'], entity['id'], experiment], sort_keys=True)
        if key not in self._cache:
            observation = self.probe_entity(entity['id'], experiment)
            if observation['status'] != 'ok':
                raise StudyPaused(observation['status'])
            if ((observation['source'], observation['revision'], observation['entity'])
                    != (entity['source'], entity['revision'], entity['id'])
                    or observation['experiment'] != experiment):
                raise StudyPaused('evidence_context_mismatch')
            self.knowledge.evidence[observation['evidence_id']] = observation
            self._cache[key] = observation
        return self._cache[key]

    def study(self, entity):
        key = (entity['source'], entity['revision'], entity['id'])
        if key in self._complete:
            return {'entity': entity['id'], 'status': 'already_studied'}
        try:
            if entity['interface'] == 'bit-transform-v1':
                self._study_transform(entity)
            elif entity['interface'] == 'bit-link-v1':
                self._study_node(entity)
            else:
                raise StudyPaused('unsupported_interface')
            result = {'entity': entity['id'], 'status': 'complete_on_declared_domain'}
            self._complete.add(key)
        except StudyPaused as exc:
            result = {'entity': entity['id'], 'status': 'deferred', 'reason': str(exc)}
        except (ValueError, RuntimeError, TypeError):
            result = {'entity': entity['id'], 'status': 'deferred', 'reason': 'probe_failed'}
        result.update(source=entity['source'], revision=entity['revision'])
        self.knowledge.studies.append(result)
        return result

    def _fact(self, entity, name, value, scope, observations):
        return self.knowledge.add_fact(entity, name, value, scope,
                                       [o['evidence_id'] for o in observations])

    def _study_transform(self, entity):
        words = [list(bits) for bits in product((0, 1), repeat=4)]
        baseline = [self._observe(entity, {'input': bits, 'flip_positions': []}) for bits in words]
        scope = 'All 16 four-bit inputs, no channel noise; adapter encoder/decoder only.'
        lengths = sorted({len(o['result']['encoded']) for o in baseline})
        self._fact(entity, 'encoded_lengths_for_4_bits', lengths, scope, baseline)
        self._fact(entity, 'roundtrip_without_noise',
                   {'recovered': sum(o['result']['decoded'] == bits for o, bits in zip(baseline, words)),
                    'tested': len(words)}, scope, baseline)
        self._fact(entity, 'encoded_equals_input',
                   all(o['result']['encoded'] == bits for o, bits in zip(baseline, words)), scope, baseline)

        # A second input length is an observation, not a universal scaling law.
        again = self._observe(entity, {'input': [0, 1, 0, 1, 1, 0, 1, 0], 'flip_positions': []})
        self._fact(entity, 'encoded_length_for_8_bit_example', len(again['result']['encoded']),
                   'Only the eight-bit input 01011010; no general scaling claim.', [again])
        perturbations = []
        restored = 0
        for bits, original in zip(words, baseline):
            for position in range(len(original['result']['encoded'])):
                observation = self._observe(entity, {'input': bits, 'flip_positions': [position]})
                perturbations.append(observation)
                restored += observation['result']['decoded'] == bits
        if not perturbations:
            raise StudyPaused('no_encoded_bits_to_perturb')
        self._fact(entity, 'single_bit_flip_recovery',
                   {'recovered': restored, 'tested': len(perturbations)},
                   'All 16 four-bit inputs and every single encoded-bit flip; no other noise.',
                   perturbations)

    def _study_node(self, entity):
        # Two distinct balanced patterns and separate replicate streams per direction.
        patterns = [[0, 1] * 256, [0] * 256 + [1] * 256]
        for direction in ('incoming', 'outgoing'):
            observations = [self._observe(entity, {'input': bits, 'direction': direction, 'replicate': i})
                            for i, bits in enumerate(patterns)]
            if any(len(o['result']['received']) != len(o['experiment']['input']) for o in observations):
                raise StudyPaused('unexpected_link_output_length')
            errors = sum(sum(a != b for a, b in zip(o['experiment']['input'], o['result']['received']))
                         for o in observations)
            trials = sum(len(o['experiment']['input']) for o in observations)
            interval = wilson_interval(errors, trials)
            orientation = 'inverted' if interval[0] > .5 else 'preserving' if interval[1] < .5 else 'undetermined'
            self._fact(entity, direction + '_link_observation',
                       {'mismatches': errors, 'bits': trials, 'mismatch_fraction': errors / trials,
                        'wilson_95': interval, 'dominant_mapping': orientation},
                       'Source→node or node→sink; two balanced 512-bit patterns, independent replicates. '
                       'Other neighbours, time variation and other inputs are untested.', observations)
