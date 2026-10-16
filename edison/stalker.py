"""Public discovery actor: source adapters expose scan_entities and activate_entity."""
from dataclasses import asdict


class Stalker:
    def __init__(self, source):
        self.source = source
        self.cursor = 0
        self.finished = False
        self.entities = {}
        self.events = []
        self.scan_calls = 0

    def poll(self):
        if self.finished:
            return []
        page = self.source.scan_entities(self.cursor, 1)
        self.scan_calls += 1
        found = []
        for entry in page['entries']:
            key = (entry['source'], entry['revision'], entry['id'])
            if key in self.entities and self.entities[key]['status'] != 'activation_failed':
                continue
            record = dict(entry)
            if entry['interface'] not in ('bit-link-v1', 'bit-transform-v1'):
                record['status'] = 'unsupported_interface'
            else:
                try:
                    self.source.activate_entity(entry['id'])
                    record['status'] = 'active'
                    found.append(record)
                except (RuntimeError, ValueError):
                    record['status'] = 'activation_failed'
            self.entities[key] = record
            self.events.append({'action': 'discovered', 'entity': dict(record),
                                'catalog': asdict(self.source.primitives)})
        self.finished = page['next_cursor'] is None
        if not self.finished:
            self.cursor = page['next_cursor']
        return found

    def refresh(self):
        """Re-scan an appendable source; known entities are not re-imported."""
        self.cursor = 0
        self.finished = False
