"""Process-separated research: python -m edison.pipeline --provider none."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from .evaluator import EvaluatorProcess
from .ideas import Branches
from .proposer import CommandProposer, ProposalFailure, request_for
from .scientist import KnowledgeBase, Scientist
from .stalker import Stalker


def run(profile: str, seed: int, budget: int, proposer=None, model_calls: int = 4,
        probe_messages: int = 512, holdout_messages: int = 8192,
        discovery: bool = False, discovery_interval: int = 2, scientist_budget: int = 512,
        knowledge_report: dict | None = None, scientist_mode: str = 'fixed',
        scientist_rounds: int = 2, scientist_tool_budget: int = 24) -> dict:
    if model_calls < 0:
        raise ValueError("Model call limit must be nonnegative")
    if discovery_interval <= 0:
        raise ValueError('Discovery interval must be positive')
    if scientist_mode not in ('fixed', 'llm') or min(scientist_rounds, scientist_tool_budget) <= 0:
        raise ValueError('Invalid scientist mode or research budgets')
    with EvaluatorProcess(profile, seed, budget, probe_messages, holdout_messages,
                          discovery=discovery, scientist_budget=scientist_budget) as evaluator:
        branches = Branches(evaluator.primitives)
        initial_catalog = asdict(evaluator.primitives)
        stalker = Stalker(evaluator) if discovery else None
        knowledge = KnowledgeBase.from_report(knowledge_report) if knowledge_report is not None else None
        if discovery and scientist_mode == 'llm':
            from .llm_scientist import LLMComponentScientist
            scientist = LLMComponentScientist(evaluator.probe_entity, knowledge,
                                               scientist_rounds, scientist_tool_budget)
        else:
            scientist = Scientist(evaluator.probe_entity, knowledge) if discovery else None
        priority = deque()
        expanded = set()
        events = []
        tested = []
        for step in range(budget):
            if stalker is not None and step % discovery_interval == 0:
                if stalker.finished:
                    stalker.refresh()
                found = stalker.poll()
                for event in stalker.events:
                    event.setdefault('before_trial', step + 1)
                branches.update_primitives(evaluator.primitives)
                for entity in found:
                    scientist.study(entity)
            while priority and priority[0] in expanded:
                priority.popleft()
            # At least every other turn belongs to the structural FIFO queue.
            preferred = priority.popleft() if step % 2 and priority else None
            idea = branches.next(preferred)
            expanded.add(idea.id)
            # Call the proposer before probing; neither outcomes nor memory go into its request.
            if proposer is not None and len(events) < model_calls and step % 2 == 0:
                request = request_for(idea.construction, evaluator.primitives)
                event = {"parent_id": idea.id, "request": request,
                         "request_sha256": hashlib.sha256(
                             json.dumps(request, sort_keys=True).encode()).hexdigest()}
                started = time.monotonic()
                try:
                    batch = proposer.propose(idea.construction, evaluator.primitives)
                    added = []
                    for proposal in batch.proposals:
                        child, is_new = branches.suggest(idea.id, proposal)
                        if is_new:
                            priority.append(child.id)
                        added.append({"idea_id": child.id, "new": is_new,
                                      "construction": asdict(proposal.construction)})
                    event.update(status="ok", proposals=added, usage=batch.usage)
                except (RuntimeError, ValueError, TimeoutError) as exc:
                    code = (exc.code if isinstance(exc, ProposalFailure) else "proposal_timeout"
                            if isinstance(exc, TimeoutError) else "invalid_proposal_response"
                            if isinstance(exc, ValueError) else "proposal_unavailable")
                    event.update(status="error", error=code)
                event["elapsed_seconds"] = round(time.monotonic() - started, 6)
                events.append(event)
            evaluator.probe(idea.construction)
            tested.append(idea.id)
        report = evaluator.finish()
        assert len(report["observations"]) == len(tested)
        for idea_id, observation in zip(tested, report["observations"]):
            observation["idea_id"] = idea_id
            if scientist is not None:
                # Retrospective links; no property values are sent to the generator.
                c = branches.graph.nodes[idea_id].construction
                contexts = {(e['id'], e['source'], e['revision']) for e in stalker.entities.values()
                            if e['status'] == 'active'}
                observation['component_fact_refs'] = scientist.knowledge.references_for(c, contexts)
        succeeded = sum(e["status"] == "ok" for e in events)
        generator_status = ("disabled" if proposer is None else "not_called" if not events
                            else "ok" if succeeded == len(events) else "failed" if not succeeded
                            else "partial")
        model_ids = {p["idea_id"] for e in events if e["status"] == "ok"
                     for p in e["proposals"] if p["new"]}
        report.update({
            "protocol": "process-v1", "budget": budget,
            "probe_messages": probe_messages, "holdout_messages": holdout_messages,
            "public": evaluator.public, "generator_events": events,
            "generator_status": generator_status,
            "model_calls_attempted": len(events), "model_calls_succeeded": succeeded,
            "model_origin_tested": len(model_ids.intersection(tested)),
            "graph": {"nodes": [asdict(n) for n in branches.graph.nodes],
                      "edges": [asdict(e) for e in branches.graph.edges],
                      "pending": list(branches.pending), "priority_pending": list(priority),
                      "duplicate_proposals": branches.graph.duplicate_proposals},
        })
        if stalker is not None:
            report['protocol'] = 'discovery-v1'
            report['discovery'] = {
                'initial_catalog': initial_catalog, 'final_catalog': asdict(evaluator.primitives),
                'interval': discovery_interval, 'scan_complete': stalker.finished,
                'scan_calls': stalker.scan_calls,
                'events': stalker.events, 'entities': list(stalker.entities.values()),
                'knowledge': scientist.knowledge.report(),
                'knowledge_sent_to_generator': False,
                'knowledge_loaded': knowledge_report is not None,
                'scientist_mode': scientist_mode,
            }
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("repetition", "relay", "coding", "inversion", "combined"),
                        default="combined")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--provider", choices=("none", "openrouter"), default="none")
    parser.add_argument("--model-calls", type=int, default=4)
    parser.add_argument('--discover', action='store_true', help='Start empty; discover and study entities')
    parser.add_argument('--discovery-interval', type=int, default=2)
    parser.add_argument('--scientist-budget', type=int, default=512)
    parser.add_argument('--knowledge-in', help='Reuse a component knowledge JSON file')
    parser.add_argument('--knowledge-out', help='Save component knowledge separately')
    parser.add_argument('--scientist-mode', choices=('fixed', 'llm'), default='fixed')
    parser.add_argument('--scientist-rounds', type=int, default=2)
    parser.add_argument('--scientist-tool-budget', type=int, default=24)
    args = parser.parse_args()
    if args.budget <= 0 or args.model_calls < 0 or args.discovery_interval <= 0 or args.scientist_budget < 0:
        parser.error("Budget must be positive; model calls must be nonnegative")
    if (args.knowledge_in or args.knowledge_out) and not args.discover:
        parser.error('Knowledge files require --discover')
    if args.scientist_mode == 'llm' and not args.discover:
        parser.error('LLM component scientist requires --discover')
    if args.scientist_mode == 'llm' and not os.environ.get('OPENROUTER_API_KEY'):
        parser.error('Configure OPENROUTER_API_KEY securely for the scientist')
    proposer = None
    if args.provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("Configure OPENROUTER_API_KEY securely before selecting OpenRouter")
        # Isolated Python startup ignores user-site modules and PYTHONPATH.
        command = [sys.executable, "-I", str(Path(__file__).with_name("openrouter.py").resolve())]
        proposer = CommandProposer(command)
    report = run(args.profile, args.seed, args.budget, proposer, args.model_calls,
                 discovery=args.discover, discovery_interval=args.discovery_interval,
                 scientist_budget=args.scientist_budget,
                 knowledge_report=json.loads(Path(args.knowledge_in).read_text()) if args.knowledge_in else None,
                 scientist_mode=args.scientist_mode, scientist_rounds=args.scientist_rounds,
                 scientist_tool_budget=args.scientist_tool_budget)
    if args.knowledge_out:
        Path(args.knowledge_out).write_text(json.dumps(report['discovery']['knowledge'], indent=2) + '\n')
    report["provider"] = args.provider
    if args.provider == "openrouter":
        from .openrouter import DEFAULT_MODEL
        report["requested_model"] = os.environ.get("EDISON_MODEL") or DEFAULT_MODEL
    print(json.dumps(report))


if __name__ == "__main__":
    main()
