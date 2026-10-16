"""Private evaluator for paths and nested transforms built from primitives."""
from dataclasses import dataclass
from .ideas import Construction, Primitives
from .world import Measurement, World, decode, encode, rng_for


@dataclass(frozen=True)
class Outcome:
    measurement: Measurement | None
    reason: str | None
    required_transmitted_bits: int


class CompositionWorld:
    PROFILES = World.PROFILES + ("combined",)

    def __init__(self, profile: str, seed: int, max_transmitted_bits: int = 2_000_000):
        if profile not in self.PROFILES or max_transmitted_bits <= 0:
            raise ValueError("Invalid profile or execution limit")
        self.profile, self.seed = profile, seed
        self.max_transmitted_bits = max_transmitted_bits
        rng = rng_for("composition", seed, profile)
        tokens = [f"p{rng.getrandbits(64):016x}" for _ in range(4)]
        a, b, identity, hamming = tokens
        self.primitives = Primitives(tuple(sorted((a, b))), tuple(sorted((identity, hamming))))
        self._transforms = {identity: "identity", hamming: "hamming"}
        self._links = {(left, right): (rng.uniform(.22, .30), False)
                       for left in ("source", a, b) for right in (a, b, "sink")}
        direct = ("source", "sink")
        self._links[direct] = (.28, False)
        self.target_ber = .025
        if profile == "repetition":
            self._links[direct] = (rng.uniform(.065, .075), False)
            self.max_cost = 3.2
        elif profile == "relay":
            self._links["source", a] = (.004, False)
            self._links[a, "sink"] = (rng.uniform(.003, .006), False)
            self.max_cost = 2.2
        elif profile == "coding":
            self._links[direct] = (rng.uniform(.035, .045), False)
            self.max_cost = 2.0
        elif profile == "inversion":
            self._links[direct] = (rng.uniform(.003, .008), True)
            self.max_cost = 1.2
        else:
            self._links["source", a] = (.02, False)
            self._links[a, "sink"] = (.02, True)
            self.max_cost = 4.0

    def validate(self, c: Construction) -> None:
        if not isinstance(c, Construction):
            raise ValueError("Expected Construction")
        if (type(c.path) is not tuple or type(c.transforms) is not tuple
                or any(n not in self.primitives.nodes for n in c.path)
                or any(t not in self.primitives.transforms for t in c.transforms)
                or type(c.repetitions) is not int or c.repetitions < 1
                or c.repetitions % 2 != 1 or type(c.flip) is not bool):
            raise ValueError("Invalid construction")

    def measure(self, c: Construction, *, split: str, messages: int) -> Outcome:
        self.validate(c)
        if split not in ("probe", "holdout") or messages <= 0:
            raise ValueError("Invalid split or message count")
        length = 4
        layers = []
        for token in c.transforms:
            mode = self._transforms[token]
            layers.append((mode, length))
            if mode == "hamming":
                length = ((length + 3) // 4) * 7
        path = ("source",) + c.path + ("sink",)
        links = [self._links[left, right] for left, right in zip(path, path[1:])]
        required = messages * length * c.repetitions * len(links)
        if required > self.max_transmitted_bits:
            return Outcome(None, "resource_limit", required)
        source = rng_for("composition", self.seed, self.profile, split, "messages")
        noise = rng_for("composition", self.seed, self.profile, split, "noise", c)
        errors = 0
        for _ in range(messages):
            original = [source.randrange(2) for _ in range(4)]
            packet = list(original)
            for mode, _ in layers:
                if mode == "hamming":
                    packet += [0] * (-len(packet) % 4)
                    packet = [bit for i in range(0, len(packet), 4)
                              for bit in encode(packet[i:i + 4], mode)]
            received = []
            for bit in packet:
                votes = 0
                for _ in range(c.repetitions):
                    transmitted = bit
                    for probability, inverted in links:
                        transmitted ^= int(inverted) ^ int(noise.random() < probability)
                    votes += transmitted ^ int(c.flip)
                received.append(int(votes > c.repetitions // 2))
            for mode, original_length in reversed(layers):
                if mode == "hamming":
                    received = [bit for i in range(0, len(received), 7)
                                for bit in decode(received[i:i + 7], mode)][:original_length]
            errors += sum(a != b for a, b in zip(original, received))
        return Outcome(Measurement(errors, messages * 4,
                                   len(links) * c.repetitions * length / 4), None, required)


class ExperimentMemory:
    """Owns measurements; never passed to the generator or branch queue."""

    def __init__(self, world: CompositionWorld, budget: int, probe_messages: int = 512):
        if budget <= 0 or probe_messages <= 0:
            raise ValueError("Budget and message count must be positive")
        self.__world = world
        self.budget, self.probe_messages = budget, probe_messages
        self.max_cost, self.target_ber = world.max_cost, world.target_ber
        self.__observations: dict[Construction, Outcome] = {}
        self.__closed = False
        self.duplicate_requests = 0

    @property
    def observations(self) -> tuple[tuple[Construction, Outcome], ...]:
        return tuple(self.__observations.items())

    def probe(self, c: Construction) -> Outcome:
        if self.__closed:
            raise RuntimeError("Session closed")
        self.__world.validate(c)
        if c in self.__observations:
            self.duplicate_requests += 1
            return self.__observations[c]
        if len(self.__observations) >= self.budget:
            raise RuntimeError("Experiment budget exhausted")
        result = self.__world.measure(c, split="probe", messages=self.probe_messages)
        self.__observations[c] = result
        return result

    def best(self) -> Construction:
        evaluated = [(c, o.measurement) for c, o in self.__observations.items()
                     if o.measurement is not None]
        if not evaluated:
            raise ValueError("No evaluated candidates")
        return min(evaluated, key=lambda pair: (
            pair[1].cost > self.max_cost, pair[1].ber, pair[1].cost))[0]

    def finalize(self, c: Construction, messages: int = 8192) -> Outcome:
        if self.__closed:
            raise RuntimeError("Only one holdout allowed")
        if c not in self.__observations:
            raise ValueError("Candidate was not tested")
        result = self.__world.measure(c, split="holdout", messages=messages)
        self.__closed = True
        return result
