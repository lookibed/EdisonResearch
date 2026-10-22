"""Author the v2 table and two transfer controls, without running a researcher.

Never modifies cable_v1. Evaluation files are not read by the live runner.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'fixtures/structural_stalker'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    original = ROOT / 'cable_v1'
    start = json.loads((original / 'public/start.json').read_text())
    table = json.loads((original / 'private/responses.json').read_text())
    start.update(fixture_id='structural-stalker-cable-v2', status='executable_static_fixture_v2')
    start['undefined_requests'] = ('fixture_not_specified означает пробел таблицы, а не физическую '
                                   'невозможность. Каждый вызов, включая ошибку, получает call_id.')
    start['access_contract'] = {
        'read': 'Только известные catalog:*, doc:* и material:*; чтение доступно во всех состояниях. '
                'Другие ссылки обозначают объекты/порты, а не читаемые документы.',
        'observe_states': ['state-0', 'state-2'],
        'observe_note': 'Все методы observe доступны только при 20 C. state-2 — новый эпизод '
                        'после возврата к 20 C; ответы для него заданы отдельно.',
        'pulse_trace_states': ['state-0', 'state-1', 'state-2'],
        'temperature_transitions': [
            {'from': 'state-0', 'temperature_C': 40, 'to': 'state-1'},
            {'from': 'state-1', 'temperature_C': 20, 'to': 'state-2'}],
        'terminal_state': 'state-2',
        'unknown_target': 'Сначала получить ссылку из доступного источника.',
        'errors': 'unknown_resource / capability_not_available / state_not_available — ограничения '
                  'контракта. Они не являются измерениями вещества.',
    }
    start['evidence_protocol'] = ('START — начальный публичный ввод; C0001 и т.д. — все попытки '
        'действий; record_id — исходная запись источника. Повтор получает новый call_id, '
        'но не новое независимое измерение. Смена состояния не создаёт новый образец.')
    table.update(fixture_id=start['fixture_id'], status='executable_static_fixture_v2')
    # Cold measurements after the return are explicit authored records, not cached observations.
    for record in list(table['records']):
        if record['request']['tool'] == 'observe':
            returned = deepcopy(record)
            returned['record_id'] += '-R'
            returned['when']['states'] = ['state-2']
            returned['response']['record_id'] = returned['record_id']
            returned['response']['payload']['context']['state_ref'] = 'state-2'
            table['records'].append(returned)
    references = next(r['response']['payload']['references'] for r in table['records']
                      if r['record_id'] == 'D02')
    for i, reference in enumerate(references):
        rid = f'D-REF-{i + 1}'
        table['records'].append({
            'record_id': rid,
            'request': {'tool': 'read', 'arguments': {'resource_ref': reference['ref']}},
            'when': {'phases': ['base', 'new_source'],
                     'states': ['state-0', 'state-1', 'state-2'], 'known_refs': [reference['ref']]},
            'response': {'record_id': rid, 'record_kind': 'document_report',
                'source_ref': 'resource:' + reference['ref'], 'source_revision': '1', 'status': 'ok',
                'payload': {'reference': reference, 'reference_conditions': {'temperature_C': 20},
                    'scope': 'Та же запись справочника D02; не новый химический анализ S17.',
                    'additional_data': None}}})
    variants = {
        'cable_v2': None,
        'homogeneous_v2': {
            'profile': [100] * 9 + [20, 20], 'inner': references[0]['response'],
            'middle': references[0]['response'], 'outer': references[0]['response'],
            'medium': references[4]['response'],
            'expectation': 'В двух сечениях нет разрешённой внутренней радиальной границы. '
                'Нельзя объявлять три наблюдённых слоя. Все три точечных отклика совместимы с G; '
                'окружение совместимо с W. Это не доказательство полной однородности. '
                'Типовой паспорт и запись партии не отменяют наблюдений; расхождение сохранить.'},
        'shifted_v2': {
            'profile': [100, 100, 100, 100, 100, 160, 160, 160, 160, 20, 20],
            'inner': references[0]['response'], 'middle': references[0]['response'],
            'outer': references[0]['response'], 'medium': references[4]['response'],
            'expectation': 'В обоих сечениях одна разрешённая внутренняя граница в (20,62.4) um, '
                'а не две на прежних координатах; внешняя в (124.9,125.1). Три точки материала '
                'совместимы с G, несмотря на различие контраста: геометрические части и материал '
                'не тождественны. Не приписывать внутреннюю границу 4 um или полимер по привычке.'},
    }
    for directory, change in variants.items():
        public, private = deepcopy(start), deepcopy(table)
        public['fixture_id'] = private['fixture_id'] = {
            'cable_v2': 'structural-stalker-cable-v2',
            'homogeneous_v2': 'structural-fixture-R42',
            'shifted_v2': 'structural-fixture-K81'}[directory]
        if change:
            for record in private['records']:
                rid = record['record_id'].removesuffix('-R')
                payload = record['response']['payload']
                if rid in ('O02', 'O03'):
                    payload['contrast_units'] = change['profile']
                if rid in ('O04', 'O05', 'O06', 'O07'):
                    payload['values'] = change[{'O04': 'inner', 'O05': 'middle',
                                               'O06': 'outer', 'O07': 'medium'}[rid]]
        target = ROOT / directory
        save(target / 'public/start.json', public)
        save(target / 'private/responses.json', private)
        if change:
            save(target / 'private/expectations.json', {
                'fixture_id': public['fixture_id'], 'authorship': 'Authored before the v2 live trials.',
                'expected_meaning': change['expectation'],
                'rules': ['Only two sampled sections; incomplete material library.',
                          'Document reports retain their scope and can disagree with observations.',
                          'No forced ontology, node count or chemical ground-truth guess.']})
        else:
            # Byte-for-byte frozen pre-existing semantic criteria; fixture_id remains historical.
            (target / 'private/expectations.json').write_bytes(
                (original / 'private/expectations.json').read_bytes())
        save(target / 'manifest.json', {
            'fixture_id': public['fixture_id'], 'parent_fixture': 'cable_v1',
            'files_sha256': {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(target.rglob('*.json')) if p.name != 'manifest.json'},
            'evaluation_access': 'Runner may read public/start.json and private/responses.json only.'})
    print('Authored cable_v2, homogeneous_v2, shifted_v2; cable_v1 unchanged.')


if __name__ == '__main__':
    main()
