"""Adapt the general research loop to entities found by the existing Stalker."""
from .investigator import ResearchScientist
from .scientist import KnowledgeBase


def component_tools(entity):
    bits = {'type': 'array', 'items': {'type': 'integer', 'enum': [0, 1]},
            'minItems': 1, 'maxItems': 1024}
    properties = {'input': bits}
    if entity['interface'] == 'bit-transform-v1':
        properties['flip_positions'] = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}}
        description = ('Invoke the unknown transform: encode input, optionally flip encoded-bit positions, '
                       'then decode. Returns result.encoded, result.received and result.decoded bit lists. '
                       'Positions refer to the encoded vector; invalid positions are rejected.')
    elif entity['interface'] == 'bit-link-v1':
        properties.update(direction={'enum': ['incoming', 'outgoing']},
                          replicate={'type': 'integer', 'minimum': 0, 'maximum': 1024})
        description = ('Transmit input on source→node (incoming) or node→receiver (outgoing). '
                       'Returns result.received bits. Changing replicate gives a new noise stream; '
                       'identical requests reuse their previous observation.')
    else:
        raise ValueError('Unsupported component interface')
    return [{'name': 'probe', 'description': description, 'arguments': {
        'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}}]


class LLMComponentScientist:
    def __init__(self, probe_entity, knowledge=None, rounds=2, call_budget=24, researcher=None):
        self.probe_entity = probe_entity
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()
        self.researcher = researcher if researcher is not None else ResearchScientist(
            rounds=rounds, call_budget=call_budget)

    def study(self, entity):
        context = (entity['id'], entity['source'], entity['revision'])
        tools = component_tools(entity)
        loaded = [o for o in self.knowledge.evidence.values()
                  if (o['entity'], o['source'], o['revision']) == context]
        def invoke(tool, arguments):
            if tool != 'probe':
                raise ValueError('Unknown tool')
            return self.probe_entity(entity['id'], arguments)
        report = self.researcher.investigate(entity, tools, invoke, initial_evidence=loaded)
        for observation in report['evidence']:
            self.knowledge.evidence[observation['evidence_id']] = observation
            self.knowledge.add_fact(entity, 'observed_response',
                                    {'request': observation['experiment'], 'result': observation['result']},
                                    'Only the exact recorded request; no general law is asserted.',
                                    [observation['evidence_id']])
        self.knowledge.interpretations.extend(dict(claim, entity=entity['id'], source=entity['source'],
                                                  revision=entity['revision'])
                                             for claim in report['interpretations'])
        self.knowledge.research_runs.append(report)
        result = {'entity': entity['id'], 'source': entity['source'], 'revision': entity['revision'],
                  'status': 'research_budget_completed', 'observations': len(report['evidence'])}
        self.knowledge.studies.append(result)
        return result
