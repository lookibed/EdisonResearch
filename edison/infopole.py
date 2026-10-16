"""Private local information-field provider. It reveals interfaces, never properties.

This finite provider is a runnable abstraction of discovery, not an Internet crawler.
Trusted providers can publish additional adapters without editing the scientist.
"""
from dataclasses import asdict
import hashlib
import json
from .ideas import Primitives
from .world import rng_for
from .transforms import valid_bits


class InfoField:
    def __init__(self, world, probe_budget=512):
        if probe_budget < 0:
            raise ValueError('Probe budget must be nonnegative')
        self._world = world
        self.probe_budget = probe_budget
        self._entries = []
        self._active = set()
        self._evidence = {}
        self.duplicate_probes = 0
        self.source = 'local-bit-field-v1'
        self.revision = hashlib.sha256(f'{world.profile}:{world.seed}:field-v1'.encode()).hexdigest()[:20]
        for token in world.primitives.nodes:
            self._publish(token, 'node', 'bit-link-v1')
        for token in world.primitives.transforms:
            self._publish(token, 'transform', 'bit-transform-v1')
        rng_for('field-order', world.seed, world.profile).shuffle(self._entries)

    def _publish(self, token, kind, interface):
        self._entries.append({'id': token, 'kind': kind, 'interface': interface,
                              'source': self.source, 'revision': self.revision})

    def publish_transform(self, token, adapter):
        self._world.register_transform(token, adapter)
        self._publish(token, 'transform', 'bit-transform-v1')

    @property
    def primitives(self):
        return Primitives(tuple(e['id'] for e in self._entries
                                if e['kind'] == 'node' and e['id'] in self._active),
                          tuple(e['id'] for e in self._entries
                                if e['kind'] == 'transform' and e['id'] in self._active))

    def scan(self, cursor=0, limit=1):
        if (type(cursor) is not int or type(limit) is not int or not 0 <= cursor <= len(self._entries)
                or not 1 <= limit <= 16):
            raise ValueError('Invalid scan cursor or page size')
        page = self._entries[cursor:cursor + limit]
        following = cursor + len(page)
        return {'entries': [dict(e) for e in page],
                'next_cursor': following if following < len(self._entries) else None}

    def activate(self, token):
        if not any(e['id'] == token for e in self._entries):
            raise ValueError('Unknown entity')
        self._active.add(token)
        return asdict(self.primitives)

    def probe(self, token, experiment):
        if token not in self._active:
            raise ValueError('Entity must be discovered and activated first')
        entry = next(e for e in self._entries if e['id'] == token)
        if type(experiment) is not dict:
            raise ValueError('Expected experiment object')
        key = hashlib.sha256(json.dumps([self.source, self.revision, token, experiment],
                                       sort_keys=True).encode()).hexdigest()
        if key in self._evidence:
            self.duplicate_probes += 1
            return self._evidence[key]
        if len(self._evidence) >= self.probe_budget:
            return {'status': 'budget_exhausted'}
        bits = experiment.get('input')
        if (type(bits) is not list or not 1 <= len(bits) <= 1024
                or any(type(b) is not int or b not in (0, 1) for b in bits)):
            raise ValueError('Expected 1..1024 input bits')
        if entry['kind'] == 'transform':
            if set(experiment) != {'input', 'flip_positions'}:
                raise ValueError('Invalid transform experiment')
            adapter = self._world._adapters[token]
            encoded = adapter.encode(bits)
            if not valid_bits(encoded) or len(encoded) != adapter.encoded_length(len(bits)):
                return {'status': 'invalid_adapter_output'}
            positions = experiment['flip_positions']
            if (type(positions) is not list or len(set(positions)) != len(positions)
                    or any(type(i) is not int or not 0 <= i < len(encoded) for i in positions)):
                raise ValueError('Invalid perturbation positions')
            if len(encoded) > 8192:
                return {'status': 'resource_limit'}
            received = list(encoded)
            for position in positions:
                received[position] ^= 1
            decoded = adapter.decode(received, len(bits))
            if not valid_bits(decoded):
                return {'status': 'invalid_adapter_output'}
            result = {'encoded': encoded, 'received': received, 'decoded': decoded}
            work = len(encoded)
        else:
            if (set(experiment) != {'input', 'direction', 'replicate'}
                    or experiment['direction'] not in ('incoming', 'outgoing')
                    or type(experiment['replicate']) is not int or not 0 <= experiment['replicate'] <= 1024):
                raise ValueError('Invalid link experiment')
            link = ('source', token) if experiment['direction'] == 'incoming' else (token, 'sink')
            probability, inverted = self._world._links[link]
            noise = rng_for('entity-probe', self.revision, token, experiment)
            received = [b ^ int(inverted) ^ int(noise.random() < probability) for b in bits]
            result = {'received': received}
            work = len(bits)
        evidence = {'status': 'ok', 'evidence_id': key, 'entity': token,
                    'source': self.source, 'revision': self.revision,
                    'experiment': experiment, 'result': result, 'processed_bits': work}
        self._evidence[key] = evidence
        return evidence

    def accounting(self):
        return {'unique_probes': len(self._evidence), 'budget': self.probe_budget,
                'duplicate_probes': self.duplicate_probes,
                'processed_bits': sum(e['processed_bits'] for e in self._evidence.values())}
