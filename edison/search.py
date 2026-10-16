"""Reference searches, not Edison. No private world parameters or holdout access."""
from .protocol import Session, best_observed, candidates
from .world import Candidate, rng_for


def search(session: Session, method: str, seed: int) -> Candidate:
    options = candidates(session.catalog)
    if method == "random":
        rng_for("search", seed).shuffle(options)
    elif method == "breadth":
        # Structural order only; observations never reject branches.
        options.sort(key=lambda c: (c.repetitions, c.flip, c.route, c.codec))
    else:
        raise ValueError(f"Unknown method: {method}")
    for candidate in options[:session.budget]:
        session.probe(candidate)
    return best_observed(session)
