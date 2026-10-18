"""Build the standalone, data-level replay from the recorded bits research run."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edison.lab import LabBackend
from edison.world import decode, encode


def decoder_trace(bits):
    word = list(bits)
    checks = []
    syndrome = 0
    for mask in (1, 2, 4):
        parity = 0
        operations = []
        for position, bit in enumerate(word, 1):
            if position & mask:
                before = parity
                parity ^= bit
                operations.append({'position': position, 'bit': bit, 'before': before, 'after': parity})
        if parity:
            syndrome |= mask
        checks.append({'mask': mask, 'operations': operations, 'parity': parity, 'syndrome': syndrome})
    if syndrome:
        word[syndrome - 1] ^= 1
    result = [word[i] for i in (2, 4, 5, 6)]
    assert result == decode(bits, 'hamming')
    return {'input': bits, 'checks': checks, 'syndrome': syndrome, 'corrected': word,
            'data_positions': [3, 5, 6, 7], 'result': result}


def main():
    report = json.loads((ROOT/'reports/research-bits-v3.json').read_text())
    backend = LabBackend('bits', report['seed'])
    evidence = {o['evidence_id']: (i+1, o) for i, o in enumerate(report['evidence'])}
    calls = []
    for round_ in report['rounds']:
        for item in round_['emissions'][0]:
            record = dict(item, round=round_['round'], number=len(calls)+1)
            if item['status'] == 'ok':
                eid, observation = evidence[item['evidence_id']]
                assert backend.invoke(item['tool'], {'bits': item['input']}) == observation
                assert observation['result'] == item['result']
                record['alias'] = f'E{eid}'
            else:
                try:
                    backend.invoke(item['tool'], {'bits': item['input']})
                except ValueError:
                    pass
                else:
                    raise AssertionError('Recorded tool error did not reproduce')
                record['alias'] = None
            calls.append(record)
    assert len(calls) == 20 and len(evidence) == 16
    corrupt = next(c for c in calls if c['tool'] == 'decode' and c['input'] == [1,1,1,0,0,0,1])
    trace = decoder_trace(corrupt['input'])
    assert trace['result'] == corrupt['result']['bits']
    assert encode([1,0,0,0], 'hamming') == [1,1,1,0,0,0,0]
    data = {'source': 'reports/research-bits-v3.json', 'entity': report['entity'],
            'tools': report['tools'], 'rounds': report['rounds'], 'review': report['review'],
            'calls': calls, 'decoder': trace,
            'internal_source': 'edison/world.py: encode / decode; edison/transforms.py: HammingTransform'}
    serialized = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    template = (ROOT/'scripts/edison_trace_template.html').read_text()
    assert template.count('__TRACE_DATA__') == 1
    output = ROOT/'docs/edison-trace.html'
    output.write_text(template.replace('__TRACE_DATA__', serialized))
    print(f'{output}: 20 calls replayed, 16 measurements and 4 tool errors verified; decoder trace matches source')


if __name__ == '__main__':
    main()
