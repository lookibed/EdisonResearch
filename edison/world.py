"""Private evaluator: search implementations use protocol.py only.

This API boundary is not a sandbox against malicious Python agents.
"""
from dataclasses import dataclass
import hashlib
import random


def rng_for(*parts: object) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(repr(parts).encode()).digest(), "big"))


@dataclass(frozen=True)
class Candidate:
    route: str
    codec: str
    repetitions: int = 1
    flip: bool = False


@dataclass(frozen=True)
class Catalog:
    routes: tuple[str, ...]
    codecs: tuple[str, ...]
    repetitions: tuple[int, ...]
    max_cost: float
    target_ber: float = 0.025


@dataclass(frozen=True)
class Measurement:
    bit_errors: int
    bits: int
    cost: float

    @property
    def ber(self) -> float:
        return self.bit_errors / self.bits


def encode(bits: list[int], mode: str) -> list[int]:
    if mode == "identity":
        return list(bits)
    a, b, c, d = bits
    return [a ^ b ^ d, a ^ c ^ d, a, b ^ c ^ d, b, c, d]


def decode(bits: list[int], mode: str) -> list[int]:
    if mode == "identity":
        return list(bits)
    word = list(bits)
    syndrome = 0
    for mask in (1, 2, 4):
        parity = 0
        for position, bit in enumerate(word, 1):
            if position & mask:
                parity ^= bit
        if parity:
            syndrome |= mask
    if syndrome:
        word[syndrome - 1] ^= 1
    return [word[i] for i in (2, 4, 5, 6)]


class World:
    PROFILES = ("repetition", "relay", "coding", "inversion")

    def __init__(self, profile: str, seed: int):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        self.profile, self.seed = profile, seed
        rng = rng_for("world", seed, profile)
        tokens = [f"x{rng.getrandbits(64):016x}" for _ in range(5)]
        routes, codec_ids = tokens[:3], tokens[3:]
        rng.shuffle(routes)
        rng.shuffle(codec_ids)
        self._codecs = dict(zip(codec_ids, ("identity", "hamming")))
        # Each link has independent bit-flip noise and optional inversion.
        if profile == "repetition":
            links = [((rng.uniform(.065, .075), False),),
                     ((.22, False), (.22, False)), ((.3, False), (.3, False))]
            max_cost = 3.2
        elif profile == "relay":
            links = [((.28, False),),
                     ((rng.uniform(.003, .006), False), (.004, False)),
                     ((.22, False), (.22, False))]
            max_cost = 2.2
        elif profile == "coding":
            links = [((rng.uniform(.035, .045), False),),
                     ((.2, False), (.2, False)), ((.3, False), (.3, False))]
            max_cost = 2.0
        else:
            links = [((rng.uniform(.003, .008), True),),
                     ((.22, False), (.22, False)), ((.3, False), (.3, False))]
            max_cost = 1.2
        self._routes = dict(zip(routes, links))
        # Public order reveals no private mapping.
        self.catalog = Catalog(tuple(sorted(routes)), tuple(sorted(codec_ids)),
                               (1, 3, 5), max_cost)

    def validate(self, candidate: Candidate) -> None:
        if not isinstance(candidate, Candidate):
            raise ValueError("Expected Candidate")
        if (candidate.route not in self._routes or candidate.codec not in self._codecs
                or type(candidate.repetitions) is not int
                or candidate.repetitions not in self.catalog.repetitions
                or type(candidate.flip) is not bool):
            raise ValueError("Candidate outside public action space")

    def measure(self, candidate: Candidate, *, split: str, messages: int) -> Measurement:
        self.validate(candidate)
        if split not in ("probe", "holdout") or messages <= 0:
            raise ValueError("Invalid split or message count")
        mode, links = self._codecs[candidate.codec], self._routes[candidate.route]
        source = rng_for(self.seed, self.profile, split, "messages")
        noise = rng_for(self.seed, self.profile, split, "noise", candidate)
        errors = 0
        for _ in range(messages):
            original = [source.randrange(2) for _ in range(4)]
            received = []
            for bit in encode(original, mode):
                votes = 0
                for _ in range(candidate.repetitions):
                    transmitted = bit
                    for probability, inverted in links:
                        transmitted ^= int(inverted) ^ int(noise.random() < probability)
                    votes += transmitted ^ int(candidate.flip)
                received.append(int(votes > candidate.repetitions // 2))
            errors += sum(a != b for a, b in zip(original, decode(received, mode)))
        cost = len(links) * candidate.repetitions * (1 if mode == "identity" else 7 / 4)
        return Measurement(errors, 4 * messages, cost)
