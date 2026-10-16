"""Public proposals, deduplication, observations, and one final evaluation."""
from dataclasses import dataclass
from itertools import product
from .world import Candidate, Catalog, Measurement, World


def candidates(catalog: Catalog) -> list[Candidate]:
    return [Candidate(*values) for values in product(
        catalog.routes, catalog.codecs, catalog.repetitions, (False, True))]


@dataclass(frozen=True)
class Observation:
    candidate: Candidate
    measurement: Measurement


class Session:
    def __init__(self, world: World, budget: int, probe_messages: int = 512):
        if budget <= 0 or probe_messages <= 0:
            raise ValueError("Budget and probe size must be positive")
        self.__world = world
        self.catalog = world.catalog
        self.budget = budget
        self.probe_messages = probe_messages
        self.__observations: dict[Candidate, Observation] = {}
        self.__closed = False
        self.duplicate_requests = 0

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self.__observations.values())

    def probe(self, candidate: Candidate) -> Observation:
        if self.__closed:
            raise RuntimeError("Session closed after final evaluation")
        self.__world.validate(candidate)
        if candidate in self.__observations:
            self.duplicate_requests += 1
            return self.__observations[candidate]
        if len(self.__observations) >= self.budget:
            raise RuntimeError("Experiment budget exhausted")
        result = Observation(candidate, self.__world.measure(
            candidate, split="probe", messages=self.probe_messages))
        self.__observations[candidate] = result
        return result

    def finalize(self, candidate: Candidate, messages: int = 8192) -> Measurement:
        if self.__closed:
            raise RuntimeError("Only one holdout evaluation allowed")
        if candidate not in self.__observations:
            raise ValueError("Final candidate must have been tested")
        if messages <= 0:
            raise ValueError("Holdout size must be positive")
        result = self.__world.measure(candidate, split="holdout", messages=messages)
        self.__closed = True
        return result


def best_observed(session: Session) -> Candidate:
    def rank(observation: Observation) -> tuple[bool, float, float]:
        m = observation.measurement
        return (m.cost > session.catalog.max_cost, m.ber, m.cost)
    if not session.observations:
        raise ValueError("No experiments performed")
    return min(session.observations, key=rank).candidate
