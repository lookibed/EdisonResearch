"""Replay actual accepted graph transactions; no model calls or evaluation files."""
import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edison.structural_graph import ResearchGraph
from edison.structural_trial import evidence_index

COLORS = {'observation': '#68cfff', 'source_report': '#bdacff',
          'hypothesis': '#ffd080', 'unresolved': '#a3b2c5'}


def quoted(value):
    return json.dumps(str(value), ensure_ascii=False)


def render(graph, delta, *, full):
    entities = deepcopy(graph['entities'])
    claims = list(graph['assertions'].values())
    if not full:
        entities = {k: e for k, e in entities.items() if not k.startswith(('doc:', 'catalog:'))}
        claims = [c for c in claims if not any(p['entity_id'].startswith(('doc:', 'catalog:'))
                                             for p in c['participants'])
                  and len({p['entity_id'] for p in c['participants']}) >= 2]
        participants = {p['entity_id'] for c in claims for p in c['participants']}
        # The compact projection draws structure; unary readings remain in full mode/JSON.
        first_resource = next(iter(entities), None)
        entities = {k: e for k, e in entities.items() if k in participants or k == first_resource}
    for claim in claims:
        for participant in claim['participants']:
            key = participant['entity_id']
            entities.setdefault(key, {'id': key, 'description': 'Known resource reference; no separate entity description.'})
    if not entities:
        return {'svg': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 120"><text x="20" y="65" fill="#aaa">No entities in this projection</text></svg>', 'metadata': {}}
    fresh = {x['id'] for x in delta.get('entities', []) + delta.get('assertions', [])}
    metadata = {}
    dot = ['digraph G {', 'graph [rankdir=LR,bgcolor="#0b1220",pad=".3",nodesep=".3",ranksep=".55",splines=polyline];',
           'node [shape=box,style="rounded,filled",fillcolor="#162638",color="#42607b",fontcolor="#eff5fb",fontname="DejaVu Sans",fontsize=14,margin=".15,.1"];',
           'edge [fontname="DejaVu Sans",fontsize=10,arrowsize=.6,penwidth=1.4];']
    ids = {key: f'n{i}' for i, key in enumerate(entities)}
    for key, item in entities.items():
        gid = ids[key]
        label = key + '\n' + '\n'.join(textwrap.wrap(item['description'][:72], 25))
        metadata[gid] = {'kind': 'entity', 'record': item}
        color = '#75e7ad' if key in fresh else '#42607b'
        dot.append(f'{gid} [id={quoted(gid)},label={quoted(label)},color={quoted(color)},penwidth=2];')
    for i, claim in enumerate(claims):
        gid = f'a{i}'
        metadata[gid] = {'kind': 'assertion', 'record': claim}
        color = COLORS[claim['origin']]
        ps = claim['participants']
        if not full and len(ps) == 2 and ps[0]['entity_id'] != ps[1]['entity_id']:
            label = claim['relation'] + '\n' + ps[0]['role'] + ' → ' + ps[1]['role']
            style = 'dashed' if claim['origin'] in ('hypothesis', 'unresolved') else 'solid'
            dot.append(f'{ids[ps[0]["entity_id"]]} -> {ids[ps[1]["entity_id"]]} '
                       f'[id={quoted(gid)},label={quoted(label)},color={quoted(color)},'
                       f'fontcolor={quoted(color)},style={quoted(style)}];')
            continue
        hub = 'h' + str(i)
        label = claim['relation']
        if full:
            label += '\n' + '\n'.join(textwrap.wrap(claim['statement'][:125], 30))
        style = '"filled,dashed"' if claim['origin'] in ('hypothesis', 'unresolved') else 'filled'
        border = '#75e7ad' if claim['id'] in fresh else color
        dot.append(f'{hub} [id={quoted(gid)},shape=ellipse,style={style},label={quoted(label)},color={quoted(border)},fontcolor={quoted(color)},fontsize=11];')
        for j, participant in enumerate(claim['participants']):
            edge_id = gid + 'e' + str(j)
            metadata[edge_id] = {'kind': 'assertion', 'record': claim}
            dot.append(f'{ids[participant["entity_id"]]} -> {hub} [id={quoted(edge_id)},label={quoted(participant["role"])},color={quoted(color)},fontcolor={quoted(color)}];')
    dot.append('}')
    raw = subprocess.run(['dot', '-Tsvg'], input='\n'.join(dot), text=True,
                         capture_output=True, check=True).stdout
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg = ET.fromstring(raw)
    svg.attrib.pop('width', None)
    svg.attrib.pop('height', None)
    svg.set('role', 'img')
    svg.set('aria-label', 'Накопленный граф на выбранном шаге реального запуска')
    return {'svg': ET.tostring(svg, encoding='unicode'), 'metadata': metadata}


def load(label, path):
    log = json.loads((path / 'session.json').read_text())
    graph = ResearchGraph(log['public_input'])
    frames = []
    empty = {'entities': [], 'assertions': []}

    def frame(title, attempt=None, observation=None):
        delta = attempt.get('response', {}).get('graph_delta', empty) if attempt else empty
        accepted = attempt and attempt['status'] == 'accepted'
        frames.append({'title': title, 'phase': attempt['phase'] if attempt else 'base',
            'status': attempt['status'] if attempt else 'initial',
            'revision': graph.data['revision'], 'entity_count': len(graph.data['entities']),
            'claim_count': len(graph.data['assertions']),
            'question_count': len(graph.data['questions']),
            'question_statuses': graph.question_statuses(),
            'graph': render(graph.data, delta if accepted else empty, full=False),
            'full_graph': render(graph.data, delta if accepted else empty, full=True),
            'question': attempt.get('response', {}).get('question', '') if attempt else '',
            'last_received': attempt['request']['history'][-1] if attempt and attempt['request']['history'] else None,
            'delta': delta, 'next_observation': observation,
            'diagnostic': attempt.get('diagnostic') if attempt else None})

    frame('Начальные ссылки из START')
    for attempt in log['attempts']:
        if attempt['status'] == 'requested':
            continue
        request = attempt['request']
        for state_ref, item in request['memory']['states'].items():
            graph.register_state(state_ref, item['evidence_refs'][0])
        observation = None
        if attempt['status'] == 'accepted':
            graph.apply(attempt['response']['graph_delta'],
                        evidence=evidence_index(log['public_input'], request['history'], request['events']),
                        known_refs=set(request['known_resource_refs']), turn=attempt['turn'])
            observation = next((h for h in log['history']
                                if h['graph_revision'] == graph.data['revision']), None)
            if observation:
                graph.register_state(observation['state_after'], observation['call_id'])
        title = f'{attempt["phase"]} · ход {attempt["turn"]} · {attempt["status"]}'
        frame(title, attempt, observation)
    final = json.loads((path / 'graph.json').read_text())
    if graph.data != final:
        raise ValueError('Graph replay differs from the saved final graph: ' + str(path))
    return {'label': label, 'fixture_id': log['fixture_id'], 'memory_mode': log['memory_mode'],
            'status': log['status'], 'usage': log.get('usage', {}), 'frames': frames,
            'evidence': evidence_index(log['public_input'], log['history'], log['events'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', nargs=2, metavar=('LABEL', 'DIRECTORY'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/stalker-evolution.html')
    args = parser.parse_args()
    data = {'runs': [load(label, Path(path)) for label, path in args.run]}
    template = (ROOT / 'scripts/structural_evolution_template.html').read_text()
    raw = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    args.output.write_text(template.replace('__DATA__', encoded))
    print(args.output, len(args.output.read_bytes()), 'bytes')


if __name__ == '__main__':
    main()
