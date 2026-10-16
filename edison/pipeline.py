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


def run(profile: str, seed: int, budget: int, proposer=None, model_calls: int = 4,
        probe_messages: int = 512, holdout_messages: int = 8192) -> dict:
    if model_calls < 0:
        raise ValueError("Model call limit must be nonnegative")
    with EvaluatorProcess(profile, seed, budget, probe_messages, holdout_messages) as evaluator:
        branches = Branches(evaluator.primitives)
        priority = deque()
        expanded = set()
        events = []
        tested = []
        for step in range(budget):
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
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("repetition", "relay", "coding", "inversion", "combined"),
                        default="combined")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--provider", choices=("none", "openrouter"), default="none")
    parser.add_argument("--model-calls", type=int, default=4)
    args = parser.parse_args()
    if args.budget <= 0 or args.model_calls < 0:
        parser.error("Budget must be positive; model calls must be nonnegative")
    proposer = None
    if args.provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("Configure OPENROUTER_API_KEY securely before selecting OpenRouter")
        # Isolated Python startup ignores user-site modules and PYTHONPATH.
        command = [sys.executable, "-I", str(Path(__file__).with_name("openrouter.py").resolve())]
        proposer = CommandProposer(command)
    report = run(args.profile, args.seed, args.budget, proposer, args.model_calls)
    report["provider"] = args.provider
    if args.provider == "openrouter":
        from .openrouter import DEFAULT_MODEL
        report["requested_model"] = os.environ.get("EDISON_MODEL") or DEFAULT_MODEL
    print(json.dumps(report))


if __name__ == "__main__":
    main()
