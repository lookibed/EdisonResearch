"""Append-only research memory. Validates provenance, never certifies physical truth.

No domain ontology, chemical solver, component activation or combination generator.
IDs are immutable; corrections are new assertions, not edits to old evidence.
"""
from copy import deepcopy
import re

from .fixture_trial import object_schema, STRING, STRINGS, ORIGIN


def array(item, maximum):
    return {'type': 'array', 'items': item, 'maxItems': maximum}


ENTITY = object_schema({'id': STRING, 'description': STRING, 'identity_basis': STRING,
                        'evidence_refs': STRINGS})
RELATION = object_schema({'id': STRING, 'description': STRING})
ASSERTION = object_schema({
    'id': STRING, 'statement': STRING, 'relation': STRING,
    'participants': array(object_schema({'role': STRING, 'entity_id': STRING}), 12),
    'evidence_refs': STRINGS, 'origin': ORIGIN, 'scope': STRING,
    'state_refs': STRINGS, 'supersedes': STRINGS,
})
QUESTION = object_schema({'id': STRING, 'text': STRING, 'targets': STRINGS,
                          'parent_ids': STRINGS, 'evidence_refs': STRINGS})
QUESTION_UPDATE = object_schema({
    'id': STRING, 'question_id': STRING,
    'status': {'type': 'string', 'enum': ['open', 'answered', 'blocked']},
    'reason': STRING, 'evidence_refs': STRINGS,
})
DELTA_SCHEMA = object_schema({
    'entities': array(ENTITY, 4), 'relation_types': array(RELATION, 8),
    'assertions': array(ASSERTION, 5), 'questions': array(QUESTION, 3),
    'question_updates': array(QUESTION_UPDATE, 4),
})


def empty_delta():
    return {key: [] for key in DELTA_SCHEMA['required']}


def validate_shape(value, schema, path='response'):
    kind = schema['type']
    expected = {'object': dict, 'array': list, 'string': str}[kind]
    if type(value) is not expected:
        raise ValueError(f'{path}: expected {kind}')
    if kind == 'object':
        if set(value) != set(schema['properties']):
            raise ValueError(f'{path}: unexpected or missing fields')
        for key, child in schema['properties'].items():
            validate_shape(value[key], child, path + '.' + key)
    elif kind == 'array':
        if len(value) > schema.get('maxItems', 128):
            raise ValueError(f'{path}: too many items')
        for child in value:
            validate_shape(child, schema['items'], path + '[]')
    elif len(value) > 6000 or ('enum' in schema and value not in schema['enum']):
        raise ValueError(f'{path}: invalid string')


class ResearchGraph:
    def __init__(self, start):
        self.data = {'protocol': 'structural-graph-v1', 'revision': 0,
                     'entities': {}, 'relation_types': {}, 'assertions': {},
                     'questions': {}, 'question_updates': {}, 'states': {}, 'events': []}
        self.data['states'][start['initial_state']['state_ref']] = {'evidence_refs': ['START']}
        for resource in start['known_resources']:
            self.data['entities'][resource['ref']] = {
                'id': resource['ref'], 'description': resource['description'],
                'identity_basis': 'Public resource reference; not a structural discovery.',
                'evidence_refs': ['START']}
        self.data['questions']['QROOT'] = {
            'id': 'QROOT', 'text': start['task'], 'targets': [start['known_resources'][0]['ref']],
            'parent_ids': [], 'evidence_refs': ['START']}

    def register_state(self, state_ref, evidence_ref):
        self.data['states'].setdefault(state_ref, {'evidence_refs': [evidence_ref]})

    def question_statuses(self):
        statuses = {qid: 'open' for qid in self.data['questions']}
        for update in self.data['question_updates'].values():
            statuses[update['question_id']] = update['status']
        return statuses

    def fifo_question(self):
        statuses = self.question_statuses()
        # The task-level umbrella is a fallback, not a permanent monopoly on FIFO slots.
        return next((qid for qid in self.data['questions']
                     if qid != 'QROOT' and statuses[qid] == 'open'),
                    'QROOT' if statuses['QROOT'] == 'open' else None)

    def view(self, mode='graph'):
        if mode == 'graph':
            return deepcopy({k: v for k, v in self.data.items() if k != 'events'})
        if mode != 'registry':
            raise ValueError('Unknown memory view')
        # Matched ablation: stable identity and question scheduling remain available.
        # Only accumulated assertion contents/evidence are withheld from feedback.
        return {'protocol': self.data['protocol'], 'revision': self.data['revision'],
                'entities': deepcopy(self.data['entities']),
                'relation_types': deepcopy(self.data['relation_types']),
                'assertion_ids': list(self.data['assertions']),
                'questions': deepcopy(self.data['questions']),
                'question_statuses': self.question_statuses(),
                'question_update_ids': list(self.data['question_updates']),
                'states': deepcopy(self.data['states'])}

    def apply(self, delta, *, evidence, known_refs, turn):
        """Atomic transaction; bad deltas do not erase or partially modify memory."""
        validate_shape(delta, DELTA_SCHEMA, 'delta')
        staged = deepcopy(self.data)
        existing_ids = {k for section in DELTA_SCHEMA['required'] for k in staged[section]}
        additions = {}
        for section in DELTA_SCHEMA['required']:
            additions[section] = {}
            for item in delta[section]:
                key = item['id']
                if not re.fullmatch(r'[\w:./@+-]{1,96}', key):
                    raise ValueError('Invalid stable ID')
                if key in additions[section]:
                    raise ValueError('Duplicate ID in delta')
                if key in existing_ids and key not in staged[section]:
                    raise ValueError('ID belongs to another record type')
                if key in staged[section] and staged[section][key] != item:
                    raise ValueError(f'{section} ID {key} is immutable; use a new ID for a correction')
                additions[section][key] = deepcopy(item)
            staged[section].update(additions[section])
        flat = [key for section in DELTA_SCHEMA['required'] for key in staged[section]]
        if len(flat) != len(set(flat)):
            raise ValueError('IDs must be distinct across record types')
        entities = set(staged['entities']) | set(known_refs)

        def refs(items, allowed, message):
            if any(ref not in allowed for ref in items):
                raise ValueError(message)

        for section, records in additions.items():
            for item in records.values():
                refs(item.get('evidence_refs', []), evidence, 'Unknown evidence reference')
                if section == 'assertions':
                    if item['relation'] not in staged['relation_types']:
                        raise ValueError('Relation needs a definition: ' + item['relation'])
                    if not item['participants']:
                        raise ValueError('Assertion needs a participant')
                    refs([p['entity_id'] for p in item['participants']], entities, 'Unknown participant')
                    refs(item['state_refs'], staged['states'], 'State has not been observed')
                    # Supersession only points back in time; no in-place rewrite or cycles.
                    refs(item['supersedes'], self.data['assertions'], 'Unknown previous assertion')
                    if item['id'] in item['supersedes']:
                        raise ValueError('Assertion cannot supersede itself')
                    if item['origin'] == 'observation':
                        kinds = {evidence[r]['record_kind'] for r in item['evidence_refs']}
                        if not kinds & {'observation', 'intervention_record', 'public_input'}:
                            raise ValueError('Document or error alone is not a physical observation; '
                                             'use source_report for document contents or metadata')
                elif section == 'questions':
                    refs(item['targets'], entities, 'Unknown question target')
                    refs(item['parent_ids'], staged['questions'], 'Unknown parent question')
                elif section == 'question_updates':
                    if item['question_id'] not in staged['questions']:
                        raise ValueError('Unknown question to update')
        visiting, visited = set(), set()

        def visit(qid):
            if qid in visiting:
                raise ValueError('Question ancestry cycle')
            if qid in visited:
                return
            visiting.add(qid)
            for parent in staged['questions'][qid]['parent_ids']:
                visit(parent)
            visiting.remove(qid)
            visited.add(qid)

        for qid in staged['questions']:
            visit(qid)
        # Empty updates are meaningful: the next action may test an old uncertainty.
        staged['revision'] += 1
        staged['events'].append({'revision': staged['revision'], 'turn': turn,
                                 'delta': deepcopy(delta)})
        self.data = staged

    def report(self):
        """Compatibility projection; every historical assertion remains available."""
        statuses = self.question_statuses()
        return {'entities': [{k: v for k, v in e.items() if k != 'identity_basis'}
                             for e in self.data['entities'].values()],
                'assertions': list(deepcopy(self.data['assertions']).values()),
                'open_questions': [q['text'] for qid, q in self.data['questions'].items()
                                   if statuses[qid] != 'answered'],
                'summary': 'Incremental model-proposed research graph; provenance checked, truth unverified.'}
