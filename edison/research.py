"""Composition experiments: python -m edison.research --budget 96 --seeds 3."""
import argparse
from dataclasses import asdict
import json
from .composition import CompositionWorld, ExperimentMemory
from .ideas import Branches


def run(profile: str, seed: int, method: str, budget: int,
        probe_messages: int = 512, holdout_messages: int = 8192) -> dict:
    if holdout_messages <= 0:
        raise ValueError("Holdout size must be positive")
    world = CompositionWorld(profile, seed)
    branches = Branches(world.primitives, method, seed)
    memory = ExperimentMemory(world, budget, probe_messages)
    tested = []
    for _ in range(budget):
        idea = branches.next()
        memory.probe(idea.construction)
        tested.append(idea.id)
    try:
        selected = memory.best()
    except ValueError:
        selected = None
    held = memory.finalize(selected, holdout_messages) if selected is not None else None
    measurement = held.measurement if held else None
    return {
        "protocol": "composition-v1", "profile": profile, "seed": seed,
        "method": method, "budget": budget, "probe_messages": probe_messages,
        "holdout_messages": holdout_messages,
        "max_transmitted_bits": world.max_transmitted_bits,
        "max_cost": memory.max_cost, "target_ber": memory.target_ber,
        "attempts": len(memory.observations),
        "executed": sum(o.measurement is not None for _, o in memory.observations),
        "resource_limited": sum(o.reason == "resource_limit" for _, o in memory.observations),
        "duplicate_requests": memory.duplicate_requests,
        "probe_transmitted_bits": sum(o.required_transmitted_bits
                                      for _, o in memory.observations if o.measurement is not None),
        "observations": [{"idea_id": i, "construction": asdict(c), "outcome": asdict(o)}
                         for i, (c, o) in zip(tested, memory.observations)],
        "graph": {"nodes": [asdict(n) for n in branches.graph.nodes],
                  "edges": [asdict(e) for e in branches.graph.edges],
                  "pending": list(branches.pending),
                  "duplicate_proposals": branches.graph.duplicate_proposals},
        "selected": asdict(selected) if selected is not None else None,
        "holdout": asdict(held) if held else None,
        "success": bool(measurement and measurement.cost <= memory.max_cost
                        and measurement.ber <= memory.target_ber),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--probe-messages", type=int, default=512)
    parser.add_argument("--holdout-messages", type=int, default=8192)
    args = parser.parse_args()
    if min(args.seeds, args.budget, args.probe_messages, args.holdout_messages) <= 0:
        parser.error("Counts and budget must be positive")
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for profile in CompositionWorld.PROFILES:
            for method in ("fair", "random"):
                print(json.dumps(run(profile, seed, method, args.budget,
                                     args.probe_messages, args.holdout_messages)), flush=True)


if __name__ == "__main__":
    main()
