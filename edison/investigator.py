"""LLM scientist: chooses hypotheses and Python experiments, then revises from evidence."""
import hashlib
import json
import ast
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

DESIGN = '''\nPhase: experiment design. Choose a small experiment that separates plausible
explanations; do not assume familiar names reveal the hidden mechanism. Keep
alternatives open, and seek counterexamples to your current favourite. A finite
sample supports only its tested scope. No catalogue of properties or target law
is supplied. You choose inputs and analysis within the public tool contracts.
Use the remaining call budget; do not repeat recorded requests unless using the
cached result in analysis. Tool errors contain no result or evidence_id: check
response["status"] == "ok" before reading those keys. Each program starts fresh;
previous variables do not persist. emit accepts one JSON-serializable value.
interpretations can cite only observations supplied before this program runs;
new measurements will be available for interpretation in the next turn.
Keep the answer compact: normally 2-3 hypotheses and short predictions.
Every call() consumes one program call, including duplicate and invalid requests.
Use execution_limits from this request; reserve budget for a follow-up round.
Do not enumerate a large Cartesian product when a few contrasting probes suffice.
If format_feedback is present, correct the rejected response. It is a formatting
or syntax failure, not experimental evidence. Never claim the rejected code ran.
'''

REVIEW = '''\nPhase: evidence review. Do not write or propose executable code in this response.
Separate observed regularities from explanations. Cite only supplied observation
aliases, state tested scope, counterexamples and remaining uncertainty. No
observations means no interpretations. Identify a next question that could
disprove the leading explanation. Do not claim a universal law from finite data.
Return at most four concise interpretations, cite only the most relevant evidence
aliases for each claim, and do not restate the full measurement table.
'''


class ResearchModel:
    def respond(self, request, review=False):
        keys = {'interpretations', 'remaining_unknowns', 'suggested_next_question'} if review else {
            'question', 'hypotheses', 'experiment', 'interpretations'}
        return request_json(request, SYSTEM + (REVIEW if review else DESIGN), keys,
                            schema=request['response_schema'], temperature=0.2, max_tokens=8192)


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
        properties.update(question={'type': 'string'}, experiment={'type': 'string',
                          'description': 'Executable Python source only. No prose, Markdown or code fences. '
                                         'Use call(name, dict) and emit(value); responses are dictionaries.'},
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
        try:
            ast.parse(response['experiment'], filename='<experiment>', mode='exec')
        except (SyntaxError, ValueError, RecursionError):
            raise ValueError('experiment must be syntactically valid Python source only') from None
    return resolved


class ResearchScientist:
    def __init__(self, model=None, executor=None, rounds=2, call_budget=24):
        if rounds <= 0 or call_budget <= 0:
            raise ValueError('Positive round and tool budgets required')
        self.model = model if model is not None else ResearchModel()
        self.executor = executor if executor is not None else ExperimentExecutor(max_calls=call_budget)
        self.rounds, self.call_budget = rounds, call_budget

    def _respond(self, prompt, aliases, trace, review=False):
        """One bounded format repair; no retries of paid transport failures."""
        for attempt in range(2):
            record = {'attempt': attempt + 1}
            trace.append(record)
            try:
                response = self.model.respond(prompt, review=review)
            except (ValueError, RuntimeError, TimeoutError) as exc:
                diagnostic = str(exc)
                safe = diagnostic in ('OpenRouter response token limit', 'OpenRouter connection failed',
                                      'Invalid OpenRouter response', 'OpenRouter response is too large',
                                      'OPENROUTER_API_KEY is not configured')
                safe = safe or (diagnostic.startswith('OpenRouter HTTP ') and diagnostic[16:].isdigit())
                record.update(status='transport_error', usage=getattr(exc, 'usage', {}),
                              diagnostic=diagnostic if safe else type(exc).__name__)
                raise
            record.update(usage=response.get('usage', {}), response=response)
            try:
                claims = validate_response(response, aliases, review=review)
            except ValueError as exc:
                record['status'] = 'rejected_format'
                record['diagnostic'] = str(exc)
                if attempt:
                    raise
                prompt = dict(prompt, format_feedback={
                    'error': str(exc), 'rejected_response': response,
                    'instruction': 'Return the complete corrected object. No code has run.'})
            else:
                record['status'] = 'accepted'
                return response, claims

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
            value = {'protocol': 'research-scientist-v3',
                     'goal': 'Investigate the unknown entity; choose the questions, interventions and analysis yourself.',
                     'entity': entity, 'tools': tools,
                     'remaining_unique_tool_calls': self.call_budget - attempts,
                     'execution_limits': {
                         'max_program_tool_calls': getattr(self.executor, 'max_calls', self.call_budget),
                         'remaining_unique_tool_attempts': self.call_budget - attempts,
                         'suggested_unique_calls_this_round': max(0, (self.call_budget - attempts) //
                             max(1, self.rounds - len(iterations))),
                         'note': 'Program call limit includes cached and invalid calls; '
                                 'unique attempt limit is shared across rounds.'},
                     'observations': observations,
                     'tool_errors': list(failures),
                     'previous_rounds': [{k: v for k, v in step.items() if k in
                                          ('question', 'hypotheses', 'program', 'execution_status',
                                           'emissions', 'diagnostic', 'error')} for step in iterations],
                     'phase': 'review_and_seek_counterexamples' if review else 'design_and_revise',
                     'response_schema': response_schema(review)}
            return value, aliases
        for index in range(self.rounds):
            prompt, aliases = request(False)
            step = {'round': index + 1,
                    'request_sha256': hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
                    'request': prompt, 'model_attempts': []}
            try:
                response, claims = self._respond(prompt, aliases, step['model_attempts'])
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
        review = {'status': 'failed', 'model_attempts': []}
        try:
            response, claims = self._respond(prompt, aliases, review['model_attempts'], review=True)
            interpretations.extend(claims)
            review = {'status': 'ok', 'interpretations': claims,
                      'remaining_unknowns': response['remaining_unknowns'],
                      'suggested_next_question': response['suggested_next_question'],
                      'usage': response.get('usage', {}), 'request': prompt,
                      'model_attempts': review['model_attempts']}
        except (ValueError, RuntimeError, TimeoutError) as exc:
            review['error'] = type(exc).__name__
        return {'protocol': 'research-scientist-v3', 'entity': entity, 'tools': tools,
                'rounds': iterations, 'review': review, 'evidence': list(ledger.values()),
                'interpretations': interpretations, 'unique_tool_attempts': attempts,
                'duplicate_requests': duplicate_requests,
                'tool_errors': failures,
                'status': 'observed' if ledger else 'no_observations',
                'claims_automatically_verified': False}
