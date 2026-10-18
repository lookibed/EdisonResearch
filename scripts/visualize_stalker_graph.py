"""Offline graph comparison: handwritten expectations versus recorded actor reports."""
import json
from pathlib import Path
import subprocess
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/structural-cable-first-run'
FIXTURE = ROOT / 'fixtures/structural_stalker/cable_v1'
COLORS = {'observation': '#68cfff', 'hypothesis': '#ffcd7c', 'unresolved': '#a4b3c5',
          'source_report': '#b9a3ff'}
TITLES = {
    'part_of': 'имеет часть', 'made_of': 'материал области',
    'composition_candidate': 'гипотеза состава', 'material_candidate': 'согласуется с',
    'material_alternatives': 'неразличимые варианты', 'immersed_in': 'погружён в',
    'contained_in': 'находится в', 'process_participants': 'участники процесса',
    'interface_between': 'граница областей', 'observed_delays': 'задержки по состояниям',
    'document_reports': 'сообщение источника', 'surrounds': 'окружение образца',
    'temperature_intervention': 'нагрев образца',
    'manufacturer_reports': 'изготовитель сообщает',
    'has_identity': 'идентичность / маркировка', 'has_label_and_length': 'маркировка и длина',
    'source_report': 'сообщение документа', 'document_claim': 'сообщение документа',
    'reference_equivalence': 'одинаковые эталонные отклики', 'matches_reference': 'совпадение откликов',
    'differs_from': 'отклики различаются', 'hypothesis_material_identity': 'гипотеза материала',
}
LABELS = {
    'sample': 'S17\nОбразец', 'center': 'Центральная\nобласть', 'annulus': 'Окружающая\nобласть',
    'outer': 'Внешняя\nобласть', 'center-material': 'Материал\nцентральной области',
    'annulus-material': 'Материал\nокружающей области', 'outer-material': 'Материал\nвнешней области',
    'vessel': 'V04\nЯчейка', 'medium': 'U09\nОкружающая смесь',
    'transmission': 'Процесс передачи\nимпульсов', 'input-events': 'Входные\nимпульсы',
    'output-events': 'Выходные\nимпульсы', 'interface': 'Граница\nдвух областей', 'model': 'M7\nМодель изделия',
    'S17': 'S17\nОбразец', 'M7': 'M7\nМаркировка / модель', 'V04': 'V04\nЯчейка', 'U09': 'U09\nСодержимое ячейки',
    'glass': 'G\nЭталон стекла', 'material:glass': 'G\nЭталон стекла',
    'GeO2': 'X\nЭталон добавки', 'material:additive': 'X\nЭталон добавки',
    'P-A': 'P-A\nЭталон полимера', 'P-B': 'P-B\nЭталон полимера',
    'material:P-A': 'material:P-A\nЭталон полимера', 'material:P-B': 'material:P-B\nЭталон полимера',
    'water': 'W\nЭталон воды', 'salt': 'N\nЭталон NaCl',
    'material:water': 'W\nЭталон воды', 'material:salt': 'N\nЭталон NaCl',
}
HIDDEN_RELATIONS = {'catalog_links', 'reference_library', 'instrument_limits',
                    'instrument_assumption', 'instrument_limitation', 'unresolved_request'}
ROLES = {'sample': 'образец', 'sample_zone': 'область образца', 'cell': 'ячейка',
         'contents': 'содержимое', 'cell_content': 'содержимое', 'reference': 'эталон',
         'material': 'материал', 'document': 'документ', 'input': 'вход', 'output': 'выход',
         'specimen': 'образец', 'process': 'процесс', 'interface': 'граница',
         'side_1': 'сторона 1', 'side_2': 'сторона 2', 'subject': 'объект сообщения',
         'reported_material': 'названный материал'}


def q(value):
    return json.dumps(str(value), ensure_ascii=False)


def origin(claim):
    value = claim.get('origin', 'hypothesis')
    if value.startswith('interpretation_of_observation') or value == 'observation_summary':
        return 'observation'
    return value if value in COLORS else 'hypothesis'


def build(key, entities, claims, full=False):
    metadata = {}
    entity_map = {e['id']: e for e in entities}
    chosen = []
    for claim in claims:
        ps = claim['participants']
        if full:
            chosen.append(claim)
        elif claim['relation'] not in HIDDEN_RELATIONS:
            ps = [p for p in ps if not p['entity_id'].startswith(('doc:', 'catalog:', 'state-'))]
            if len({p['entity_id'] for p in ps}) >= 2:
                chosen.append(dict(claim, participants=ps))
    participants = {p['entity_id'] for c in chosen for p in c['participants']}
    for name in participants - entity_map.keys():
        entity_map[name] = {'id': name, 'description': 'Публичная ссылка из участников; отдельного описания сущности в отчёте нет.', 'evidence_refs': [], 'external_reference': True}
    if not full:
        entity_map = {k: v for k, v in entity_map.items()
                      if not k.startswith(('doc:', 'catalog:', 'state-'))}
    ids = {name: f'n{i}' for i, name in enumerate(entity_map)}
    dot = ['digraph G {', 'graph [rankdir=TB, bgcolor="#0b1220", fontname="DejaVu Sans", pad="0.4", nodesep="0.32", ranksep="0.5", splines=polyline, outputorder=edgesfirst];',
           'node [shape=box, style="rounded,filled", fillcolor="#162638", color="#35506b", fontcolor="#eff5fb", fontname="DejaVu Sans", fontsize=17, margin="0.17,0.13", penwidth=1.5];',
           'edge [color="#68cfff", fontcolor="#b7cbdf", fontname="DejaVu Sans", fontsize=12, penwidth=1.7, arrowsize=0.65];']
    for name, entity in entity_map.items():
        title = LABELS.get(name) or '\n'.join(textwrap.wrap(name, 23))
        gid = key + '-' + ids[name]
        metadata[gid] = {'kind': 'entity', 'entity': entity}
        color = '#67dfb0' if name in ('sample', 'S17') else '#35506b'
        dot.append(f'{ids[name]} [id={q(gid)}, label={q(title)}, color={q(color)}, tooltip={q(entity.get("description", ""))}];')
    counter = 0

    def connect(a, b, label, claim, role=None):
        nonlocal counter
        gid = key + '-edge-' + str(counter)
        counter += 1
        metadata[gid] = {'kind': 'assertion', 'assertion': claim, 'display_role': role}
        color = COLORS[origin(claim)]
        style = 'dashed' if origin(claim) in ('hypothesis', 'unresolved') else 'solid'
        dot.append(f'{a} -> {b} [id={q(gid)}, label={q(label)}, color={q(color)}, fontcolor={q(color)}, style={q(style)}, tooltip={q(claim["statement"])}];')

    for ci, claim in enumerate(chosen):
        ps = claim['participants']
        rel = claim['relation']
        label = '\n'.join(textwrap.wrap(TITLES.get(rel, rel), 22))
        if rel == 'part_of':
            roles = {p['role']: p['entity_id'] for p in ps}
            connect(ids[roles['whole']], ids[roles['part']], label, claim,
                    'Отображено whole → part; исходное утверждение part_of сохранено в карточке.')
        elif len(ps) == 2 and len({p['entity_id'] for p in ps}) == 2:
            connect(ids[ps[0]['entity_id']], ids[ps[1]['entity_id']], label, claim)
        else:
            hub = 'claim' + str(ci)
            gid = key + '-claim-' + str(ci)
            metadata[gid] = {'kind': 'assertion', 'assertion': claim}
            color = COLORS[origin(claim)]
            dot.append(f'{hub} [id={q(gid)}, shape=ellipse, style=filled, fillcolor="#14202e", color={q(color)}, fontcolor={q(color)}, fontsize=13, label={q(label)}, tooltip={q(claim["statement"])}];')
            for p in ps:
                connect(ids[p['entity_id']], hub, ROLES.get(p['role'], p['role']), claim, p['role'])
    dot.append('}')
    raw = subprocess.run(['dot', '-Tsvg'], input='\n'.join(dot), text=True,
                         capture_output=True, check=True).stdout
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg = ET.fromstring(raw)
    svg.attrib.pop('width', None)
    svg.attrib.pop('height', None)
    svg.set('data-key', key)
    svg.set('role', 'img')
    svg.set('aria-label', 'Ожидаемый граф' if key.startswith('expected') else 'Граф фактического отчёта')
    # Large transparent hit areas make thin edges clickable after zooming.
    ns = '{http://www.w3.org/2000/svg}'
    for group in svg.iter(ns + 'g'):
        if group.get('class') == 'edge':
            path = group.find(ns + 'path')
            if path is not None:
                hit = ET.Element(ns + 'path', dict(path.attrib))
                hit.set('stroke', 'transparent')
                hit.set('stroke-width', '18')
                hit.set('fill', 'none')
                hit.set('class', 'hit')
                group.append(hit)
    return {'svg': ET.tostring(svg, encoding='unicode'), 'metadata': metadata,
            'entities_shown': len(entity_map), 'claims_shown': len(chosen),
            'full': full}


def main():
    manual_file = json.loads((FIXTURE / 'private/manual-example.json').read_text())
    manual = manual_file['base']
    table = json.loads((FIXTURE / 'private/responses.json').read_text())
    records = {r['record_id']: r['response'] for r in table['records']}
    refs = records['D02']['payload']['references']
    expected_entities = [{'id': e['id'], 'description': e['meaning'], 'evidence_refs': e['basis']}
                         for e in manual['entities']]
    expected_entities.extend({'id': r['ref'], 'description': r['name'], 'evidence_refs': ['D02']}
                             for r in refs)
    expected_claims = []
    for c in manual['assertions']:
        ps = [{'role': role, 'entity_id': eid} for role, eid in c['participants'].items()]
        for eid in c.get('reference_model', {}):
            ps.append({'role': 'совместимый компонент', 'entity_id': eid})
        if c.get('compatible_reference'):
            ps.append({'role': 'совместимый эталон', 'entity_id': c['compatible_reference']})
        for eid in c.get('alternatives', []):
            ps.append({'role': 'возможный тип', 'entity_id': eid})
        expected_claims.append({'id': c['id'], 'relation': c['relation'], 'participants': ps,
                                'statement': c.get('scope', TITLES.get(c['relation'], c['relation'])),
                                'scope': c.get('scope', 'Ручной пример, не единственное допустимое представление.'),
                                'origin': c['origin'], 'evidence_refs': c['evidence_refs'],
                                'original_manual_assertion': c})
    graphs = {}
    for phase in ('base', 'new_source'):
        claims = list(expected_claims)
        if phase == 'new_source':
            update = manual_file['new_source_delta']['assertion']
            claims.append({'id': 'new-source-message', 'relation': update['relation'],
                           'participants': [{'role': 'subject', 'entity_id': update['participants']['subject']},
                                            {'role': 'reported_material', 'entity_id': update['reported_material']}],
                           'statement': 'Изготовитель сообщает P-B для покрытия S17; измерение O06 по-прежнему не различает A/B.',
                           'scope': manual_file['new_source_delta']['still_true'],
                           'origin': update['origin'], 'evidence_refs': update['evidence_refs'],
                           'original_manual_assertion': update})
        for full in (False, True):
            key = 'expected-' + phase + '-' + ('full' if full else 'structure')
            graphs[key] = build(key, expected_entities, claims, full)
    reports = {}
    for phase in ('base', 'new_source'):
        report = json.loads((RUN / (phase + '-report.json')).read_text())
        reports[phase] = report
        for full in (False, True):
            key = phase + '-' + ('full' if full else 'structure')
            graphs[key] = build(key, report['entities'], report['assertions'], full)
    session = json.loads((RUN / 'session.json').read_text())
    delivered = {h['response']['record_id'] for h in session['history'] if 'record_id' in h['response']}
    data = {'graphs': graphs, 'reports': reports, 'records': records,
            'received_record_ids': sorted(delivered), 'public_input': session['public_input'],
            'review': json.loads((RUN / 'manual-review.json').read_text())}
    template = (ROOT / 'scripts/stalker_graph_template.html').read_text()
    assert template.count('__GRAPH_DATA__') == 1
    output = ROOT / 'docs/stalker-graph.html'
    output.write_text(template.replace('__GRAPH_DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')))
    print(output)


if __name__ == '__main__':
    main()
