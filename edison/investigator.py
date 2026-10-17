"""LLM scientist: chooses hypotheses and Python experiments, then revises from evidence."""
import hashlib
import json
from .experiments import ExperimentExecutor
from .openrouter import request_json

SYSTEM = (
    'You are an experimental scientist studying unknown tools. Form competing explanations, '
    'predict outcomes and design experiments that distinguish them. You write Python, using '
    'call(tool_name, arguments_dict) to obtain a trusted observation and emit(json_value) to '
    'report computed summaries. call returns an object with status, result, and evidence_id. '
    'The executor has Python standard libraries but no network, secrets or project files. '
    'Do not open files or create processes. Prefer small discriminating experiments. '
    'Use earlier observations to revise explanations, including counterexamples. '
    'Treat errors and resource limits as observations of execution, not laws of the object. '
    'A tool result is evidence; your interpretation can be wrong. Cite provided E1, E2, ... '
    'aliases, state scope and uncertainties. Never invent an evidence alias. '
    'When there are no observations, interpretations must be empty. stdout is reserved; use emit, not print. '
    'The experiment field MUST contain only executable Python source, never a plan, prose, '
    'Markdown, a Program: prefix or discussion after the code. Tool responses are Python '
    'dicts: use response["status"], response["result"] and response["evidence_id"], '
    'never attribute access such as response.status. Put explanations in question and hypotheses. '
    'Output exactly one JSON object with the requested keys. No code fences.'
)


class ResearchModel:
    def respond(self, request, review=False):
        keys = {'interpretations', 'remaining_unknowns', 'suggested_next_question'} if review else {
            'question', 'hypotheses', 'experiment', 'interpretations'}
        return request_json(request, SYSTEM, keys)


def interpretation_schema():
    return {'type': 'array', 'maxItems': 8, 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['claim', 'evidence_ids', 'scope', 'uncertainties'],
        'properties': {'claim': {'type': 'string'}, 'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                       'scope': {'type': 'string'}, 'uncertainties': {'type': 'array', 'items': {'type': 'string'}}}}}


def response_schema(review=False):
    properties = {'interpretations': interpretation_schema()}
    if review:
        properties.update(remaining_unknowns={'type': 'array', 'items': {'type': 'string'}},
                          suggested_next_question={'type': 'string'})
    else:
        properties.update(question={'type': 'string'}, experiment={'type': 'string'},
                          hypotheses={'type': 'array', 'minItems': 2, 'maxItems': 5, 'items': {
                              'type': 'object', 'additionalProperties': False,
                              'required': ['id', 'statement', 'prediction', 'falsification'],
                              'properties': {k: {'type': 'string'} for k in
                                             ('id', 'statement', 'prediction', 'falsification')}}})
    return {'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}


def validate_response(response, aliases, review=False):
    expected = set(response_schema(review)['required'])
    if not isinstance(response, dict) or set(response) - {'usage'} != expected:
        raise ValueError('Unexpected scientist response fields')
    interpretations = response['interpretations']
    if not isinstance(interpretations, list) or len(interpretations) > 8:
        raise ValueError('Invalid interpretations')
    resolved = []
    for item in interpretations:
        if (not isinstance(item, dict) or set(item) != {'claim', 'evidence_ids', 'scope', 'uncertainties'}
                or not isinstance(item['claim'], str) or not item['claim'].strip()
                or not isinstance(item['scope'], str) or not item['scope'].strip()
                or type(item['evidence_ids']) is not list or not item['evidence_ids']
                or any(type(i) is not str or i not in aliases for i in item['evidence_ids'])
                or type(item['uncertainties']) is not list
                or any(type(s) is not str for s in item['uncertainties'])):
            raise ValueError('Interpretation must cite existing observations and a scope')
        resolved.append(dict(item, evidence_ids=[aliases[i] for i in item['evidence_ids']],
                             status='llm_interpretation_unverified'))
    if review:
        if (type(response['remaining_unknowns']) is not list
                or any(type(s) is not str for s in response['remaining_unknowns'])
                or type(response['suggested_next_question']) is not str):
            raise ValueError('Invalid review')
    else:
        hypotheses = response['hypotheses']
        if (type(response['question']) is not str or not response['question'].strip()
                or type(response['experiment']) is not str or not 1 <= len(response['experiment']) <= 16000
                or type(hypotheses) is not list or not 2 <= len(hypotheses) <= 5):
            raise ValueError('Expected a question, competing hypotheses and executable procedure')
        ids = []
        for h in hypotheses:
            if (not isinstance(h, dict) or set(h) != {'id', 'statement', 'prediction', 'falsification'}
                    or any(type(v) is not str or not v.strip() for v in h.values())):
                raise ValueError('Invalid hypothesis')
            ids.append(h['id'])
        if len(set(ids)) != len(ids):
            raise ValueError('Hypothesis IDs must differ')
    return resolved


class ResearchScientist:
    def __init__(self, model=None, executor=None, rounds=2, call_budget=24):
        if rounds <= 0 or call_budget <= 0:
            raise ValueError('Positive round and tool budgets required')
        self.model = model if model is not None else ResearchModel()
        self.executor = executor if executor is not None else ExperimentExecutor(max_calls=call_budget)
        self.rounds, self.call_budget = rounds, call_budget

    def investigate(self, entity, tools, invoke, initial_evidence=()):
        self.executor.check()
        ledger = {}
        iterations = []
        interpretations = []
        attempts = 0
        seen_requests = {}
        duplicate_requests = 0
        failures = []
        for observation in initial_evidence:
            if ((observation['entity'], observation['source'], observation['revision'])
                    != (entity['id'], entity['source'], entity['revision'])):
                raise ValueError('Loaded evidence scope mismatch')
            ledger[observation['evidence_id']] = observation
            experiment = observation['experiment']
            if set(experiment) == {'tool', 'arguments'}:
                tool, arguments = experiment['tool'], experiment['arguments']
            elif len(tools) == 1:
                tool, arguments = tools[0]['name'], experiment
            else:
                continue
            key = json.dumps([entity['source'], entity['revision'], tool, arguments], sort_keys=True)
            seen_requests[key] = observation
        def observe(tool, arguments):
            nonlocal attempts, duplicate_requests
            key = json.dumps([entity['source'], entity['revision'], tool, arguments], sort_keys=True)
            if key in seen_requests:
                duplicate_requests += 1
                return seen_requests[key]
            if attempts >= self.call_budget:
                return {'status': 'budget_exhausted'}
            attempts += 1
            try:
                observation = invoke(tool, arguments)
            except (ValueError, RuntimeError, TypeError, KeyError):
                failures.append({'tool': tool, 'arguments': arguments, 'status': 'tool_error'})
                return {'status': 'tool_error'}
            if observation.get('status') == 'ok':
                if ((observation['entity'], observation['source'], observation['revision'])
                        != (entity['id'], entity['source'], entity['revision'])):
                    raise ValueError('Evidence scope mismatch')
                ledger[observation['evidence_id']] = observation
                seen_requests[key] = observation
            else:
                failures.append({'tool': tool, 'arguments': arguments, 'status': observation.get('status')})
            return observation
        def request(review):
            aliases = {f'E{i}': key for i, key in enumerate(ledger, 1)}
            observations = [{'id': name, 'tool_request': ledger[key]['experiment'],
                             'result': ledger[key]['result']} for name, key in aliases.items()]
            value = {'protocol': 'research-scientist-v1',
                     'goal': 'Investigate the unknown entity; choose the questions, interventions and analysis yourself.',
                     'entity': entity, 'tools': tools,
                     'remaining_unique_tool_calls': self.call_budget - attempts,
                     'observations': observations,
                     'tool_errors': list(failures),
                     'previous_rounds': [{k: v for k, v in step.items() if k in
                                          ('question', 'hypotheses', 'program', 'execution_status',
                                           'emissions', 'diagnostic')} for step in iterations],
                     'phase': 'review_and_seek_counterexamples' if review else 'design_and_revise',
                     'response_schema': response_schema(review)}
            return value, aliases
        for index in range(self.rounds):
            prompt, aliases = request(False)
            step = {'round': index + 1,
                    'request_sha256': hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
                    'request': prompt}
            try:
                response = self.model.respond(prompt)
                claims = validate_response(response, aliases)
                interpretations.extend(claims)
                outcome = self.executor.run(response['experiment'], tools, observe)
                step.update(question=response['question'], hypotheses=response['hypotheses'],
                            program=response['experiment'], execution_status=outcome['status'],
                            emissions=outcome['emissions'], diagnostic=outcome.get('diagnostic'),
                            usage=response.get('usage', {}), interpretations=claims)
            except (ValueError, RuntimeError, TimeoutError) as exc:
                # Isolation failures must stop instead of silently executing on the host.
                from .experiments import SandboxUnavailable
                if isinstance(exc, SandboxUnavailable):
                    raise
                step.update(execution_status='research_error', error=type(exc).__name__)
            iterations.append(step)
        prompt, aliases = request(True)
        review = {'status': 'failed'}
        try:
            response = self.model.respond(prompt, review=True)
            claims = validate_response(response, aliases, review=True)
            interpretations.extend(claims)
            review = {'status': 'ok', 'interpretations': claims,
                      'remaining_unknowns': response['remaining_unknowns'],
                      'suggested_next_question': response['suggested_next_question'],
                      'usage': response.get('usage', {}), 'request': prompt}
        except (ValueError, RuntimeError, TimeoutError) as exc:
            review['error'] = type(exc).__name__
        return {'protocol': 'research-scientist-v1', 'entity': entity, 'tools': tools,
                'rounds': iterations, 'review': review, 'evidence': list(ledger.values()),
                'interpretations': interpretations, 'unique_tool_attempts': attempts,
                'duplicate_requests': duplicate_requests,
                'tool_errors': failures,
                'status': 'observed' if ledger else 'no_observations',
                'claims_automatically_verified': False}
