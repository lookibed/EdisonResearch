"""Check reproducibility/provenance of saved trials; does not grade their discoveries."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edison.structural_trial import FixtureSource, replay_session


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(path, fixture):
    log = json.loads((path / 'session.json').read_text())
    source = FixtureSource(fixture)
    for relative, expected in log['input_sha256'].items():
        if sha((fixture / relative).read_bytes()) != expected:
            raise ValueError('Fixture changed: ' + relative)
    for entry in log['history']:
        if entry['phase'] != source.phase:
            source.new_source()
        actual = source.call(entry['request']['tool'], entry['request']['arguments'])
        if any(actual[key] != entry[key] for key in actual):
            raise ValueError('Tool replay mismatch: ' + entry['call_id'])
    restored = replay_session(log).data
    if restored != json.loads((path / 'graph.json').read_text()):
        raise ValueError('Final graph replay mismatch')
    for phase in log['reports']:
        partial = deepcopy(log)
        phases = ['base'] if phase == 'base' else ['base', 'new_source']
        partial['attempts'] = [a for a in partial['attempts'] if a['phase'] in phases]
        partial['history'] = [h for h in partial['history'] if h['phase'] in phases]
        if replay_session(partial).data != json.loads((path / (phase + '-graph.json')).read_text()):
            raise ValueError('Frozen checkpoint replay mismatch: ' + phase)
    forbidden = ('world.json', 'expectations.json', 'manual-example.json', 'EXPECTED.md',
                 'w:core', 'oracle_choice')
    for attempt in log['attempts']:
        text = json.dumps(attempt['request'], ensure_ascii=False)
        if any(marker in text for marker in forbidden):
            raise ValueError('Private evaluation marker in actor input')
        if attempt['phase'] == 'base' and 'D04' in {
                h['response'].get('record_id') for h in attempt['request']['history']}:
            raise ValueError('Later-source information in base phase')
    summary = {
        'fixture_id': log['fixture_id'], 'memory_mode': log['memory_mode'],
        'protocol': log['protocol'], 'status': log['status'],
        'system_sha256': sha(log['actor_system'].encode()),
        'input_sha256': log['input_sha256'],
        'tool_calls': len(log['history']), 'model_calls': len(log['attempts']),
        'accepted_turns': sum(a['status'] == 'accepted' for a in log['attempts']),
        'rejected_turns': sum(a['status'] == 'rejected' for a in log['attempts']),
        'transport_errors': sum(a['status'] == 'transport_error' for a in log['attempts']),
        'records_obtained': list(dict.fromkeys(h['response']['record_id'] for h in log['history']
                                             if 'record_id' in h['response'])),
        'tool_errors': [{'call_id': h['call_id'], 'status': h['response']['status']}
                        for h in log['history'] if h['response']['status'] != 'ok'],
        'entities': len(restored['entities']), 'assertions': len(restored['assertions']),
        'questions': len(restored['questions']), 'usage': log.get('usage', {}),
        'checkpoints': log['reports'],
        'checks': {'fixture_hashes': True, 'all_tool_results_replayed': True,
                   'final_graph_replayed': True, 'frozen_checkpoints_replayed': True,
                   'no_private_evaluation_markers_in_actual_packets': True},
        'limits': 'Protocol and replay checks only; not independent semantic evaluation.',
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.fixture)
    (args.directory / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
