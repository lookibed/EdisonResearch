"""Run with python -m edison.benchmark; outputs JSON lines."""
import argparse
from dataclasses import asdict
import json
from .protocol import Session
from .search import search
from .world import World


def run(profile: str, seed: int, method: str, budget: int,
        probe_messages: int = 512, holdout_messages: int = 8192) -> dict:
    session = Session(World(profile, seed), budget, probe_messages)
    selected = search(session, method, seed)
    probe = next(o.measurement for o in session.observations if o.candidate == selected)
    held = session.finalize(selected, holdout_messages)
    catalog = session.catalog
    return {
        "profile": profile, "seed": seed, "method": method, "budget": budget,
        "experiments": len(session.observations),
        "duplicate_requests": session.duplicate_requests,
        "observations": [asdict(o) for o in session.observations],
        "probe_transmitted_bits": sum(
            o.measurement.bits * o.measurement.cost for o in session.observations),
        "selected": asdict(selected), "probe": asdict(probe),
        "holdout": asdict(held), "holdout_ber": held.ber,
        "target_ber": catalog.target_ber, "max_cost": catalog.max_cost,
        "success": held.ber <= catalog.target_ber and held.cost <= catalog.max_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--probe-messages", type=int, default=512)
    parser.add_argument("--holdout-messages", type=int, default=8192)
    args = parser.parse_args()
    if min(args.seeds, args.budget, args.probe_messages, args.holdout_messages) <= 0:
        parser.error("Counts and budget must be positive")
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for profile in World.PROFILES:
            for method in ("random", "breadth"):
                print(json.dumps(run(profile, seed, method, args.budget,
                                     args.probe_messages, args.holdout_messages)), flush=True)


if __name__ == "__main__":
    main()
