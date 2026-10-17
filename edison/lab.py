"""Cross-domain LLM research: python -m edison.lab --domain function --rounds 2.

Domains specify tool contracts and hidden laws, never the scientist's experiments.
"""
import argparse
import hashlib
import json
import math
import os
from .investigator import ResearchScientist
from .transforms import HammingTransform, IdentityTransform, valid_bits
from .world import rng_for


class LabBackend:
    DOMAINS = ('bits', 'function', 'physics')

    def __init__(self, domain, seed=0):
        if domain not in self.DOMAINS:
            raise ValueError('Unknown domain')
        self.domain, self.seed = domain, seed
        self.entity = {'id': 'object', 'source': 'local-research-lab-v1',
                       'revision': hashlib.sha256(f'{domain}:{seed}:v1'.encode()).hexdigest()[:20],
                       'interface': 'json-tools-v1'}
        self._codec = HammingTransform() if seed % 2 == 0 else IdentityTransform()
        rng = rng_for('lab-physics', seed)
        self._mass, self._drag = rng.uniform(1, 4), rng.uniform(.2, .8)
        arrays = {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': 64}
        bit_arrays = dict(arrays, items={'type': 'integer', 'enum': [0, 1]})
        if domain == 'bits':
            self.tools = [self._tool('encode', {'bits': bit_arrays}, 'Returns a wire bit vector as result.bits.'),
                          self._tool('decode', {'bits': dict(bit_arrays, maxItems=448)},
                                     'Returns a decoded bit vector as result.bits; invalid frames may be rejected.')]
        elif domain == 'function':
            self.tools = [self._tool('evaluate', {'values': arrays},
                                     'Unknown function of a list of integers; returns a number as result.value.')]
        else:
            self.tools = [self._tool('observe', {
                'force': {'type': 'number', 'minimum': -10, 'maximum': 10},
                'duration': {'type': 'number', 'minimum': 0, 'maximum': 5}},
                'Apply constant force in N for duration in s from rest at position zero. '
                'Each call resets the object. Returns position in m and velocity in m/s.')]

    @staticmethod
    def _tool(name, properties, description):
        return {'name': name, 'description': description, 'arguments': {
            'type': 'object', 'additionalProperties': False, 'properties': properties,
            'required': list(properties)}}

    def invoke(self, tool, arguments):
        descriptor = next((t for t in self.tools if t['name'] == tool), None)
        if descriptor is None or set(arguments) != set(descriptor['arguments']['required']):
            raise ValueError('Invalid tool or arguments')
        if self.domain == 'function':
            values = arguments['values']
            if (type(values) is not list or len(values) > 64
                    or any(type(x) is not int or abs(x) > 10000 for x in values)):
                raise ValueError('Expected up to 64 bounded integers')
            mode = self.seed % 3
            value = sum(x * x for x in values) if mode == 0 else sum(values) if mode == 1 else len(set(values))
            result = {'value': value}
        elif self.domain == 'bits':
            bits = arguments['bits']
            if not valid_bits(bits) or len(bits) > (64 if tool == 'encode' else 448):
                raise ValueError('Invalid bits')
            if tool == 'encode':
                output = self._codec.encode(bits)
            else:
                if isinstance(self._codec, HammingTransform) and len(bits) % 7:
                    raise ValueError('Invalid frame')
                length = len(bits) // 7 * 4 if isinstance(self._codec, HammingTransform) else len(bits)
                output = self._codec.decode(bits, length)
            result = {'bits': output}
        else:
            force, duration = arguments['force'], arguments['duration']
            if (type(force) not in (int, float) or type(duration) not in (int, float)
                    or not math.isfinite(force) or not math.isfinite(duration)
                    or abs(force) > 10 or not 0 <= duration <= 5):
                raise ValueError('Force or duration out of bounds')
            gamma = self._drag / self._mass
            factor = -math.expm1(-gamma * duration)
            result = {'position': round(force / self._drag * (duration - factor / gamma), 9),
                      'velocity': round(force / self._drag * factor, 9)}
        experiment = {'tool': tool, 'arguments': arguments}
        context = [self.entity, experiment, result]
        return {'status': 'ok', 'entity': self.entity['id'], 'source': self.entity['source'],
                'revision': self.entity['revision'], 'experiment': experiment, 'result': result,
                'evidence_id': hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=LabBackend.DOMAINS, default='function')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--tool-budget', type=int, default=24)
    args = parser.parse_args()
    if not os.environ.get('OPENROUTER_API_KEY'):
        parser.error('Configure OPENROUTER_API_KEY securely')
    backend = LabBackend(args.domain, args.seed)
    scientist = ResearchScientist(rounds=args.rounds, call_budget=args.tool_budget)
    result = scientist.investigate(backend.entity, backend.tools, backend.invoke)
    result.update(domain=args.domain, seed=args.seed)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
