import unittest
from edison.benchmark import run
from edison.protocol import Session, candidates
from edison.world import Candidate, World, decode, encode


class BenchmarkTests(unittest.TestCase):
    def test_hamming_corrects_every_single_bit_error(self):
        for value in range(16):
            bits = [(value >> i) & 1 for i in range(4)]
            encoded = encode(bits, "hamming")
            self.assertEqual(decode(encoded, "hamming"), bits)
            for index in range(7):
                damaged = list(encoded)
                damaged[index] ^= 1
                self.assertEqual(decode(damaged, "hamming"), bits)

    def test_budget_dedup_and_finalization(self):
        session = Session(World("relay", 9), 1, 16)
        first, second = candidates(session.catalog)[:2]
        self.assertEqual(session.probe(first), session.probe(first))
        self.assertEqual(len(session.observations), 1)
        self.assertEqual(session.duplicate_requests, 1)
        with self.assertRaises(RuntimeError):
            session.probe(second)
        with self.assertRaises(ValueError):
            session.finalize(second, 16)
        session.finalize(first, 16)
        with self.assertRaises(RuntimeError):
            session.probe(first)
        with self.assertRaises(RuntimeError):
            session.finalize(first, 16)

    def test_invalid_proposal_does_not_consume_budget(self):
        session = Session(World("coding", 4), 1)
        with self.assertRaises(ValueError):
            session.probe(Candidate("unknown", "unknown"))
        self.assertEqual(session.observations, ())

    def test_catalog_has_no_profile_or_properties(self):
        catalog = World("inversion", 1).catalog
        self.assertEqual(set(vars(catalog)),
                         {"routes", "codecs", "repetitions", "max_cost", "target_ber"})
        self.assertEqual(len(set(candidates(catalog))), 36)
        self.assertNotEqual(catalog.routes, World("inversion", 2).catalog.routes)

    def test_reproducible_and_distinct_splits(self):
        world = World("coding", 17)
        candidate = candidates(world.catalog)[0]
        probe = world.measure(candidate, split="probe", messages=128)
        self.assertEqual(probe, world.measure(candidate, split="probe", messages=128))
        self.assertNotEqual(probe.bit_errors, world.measure(
            candidate, split="holdout", messages=128).bit_errors)

    def test_each_profile_is_solvable_by_full_reference_search(self):
        for profile in World.PROFILES:
            with self.subTest(profile=profile):
                result = run(profile, 0, "breadth", 36, 512, 4096)
                self.assertEqual(result["experiments"], 36)
                self.assertTrue(result["success"], result)

    def test_target_mechanisms_improve_over_unmodified_transmission(self):
        # Evaluator tests may inspect private truth; search implementations may not.
        for profile in World.PROFILES:
            with self.subTest(profile=profile):
                world = World(profile, 0)
                identity = next(k for k, v in world._codecs.items() if v == "identity")
                hamming = next(k for k, v in world._codecs.items() if v == "hamming")
                direct = next(k for k, v in world._routes.items() if len(v) == 1)
                raw = Candidate(direct, identity)
                if profile == "repetition":
                    improved = Candidate(direct, identity, 3)
                elif profile == "coding":
                    improved = Candidate(direct, hamming)
                elif profile == "relay":
                    relay = min(world._routes, key=lambda k: sum(p for p, _ in world._routes[k]))
                    improved = Candidate(relay, identity)
                else:
                    improved = Candidate(direct, identity, flip=True)
                before = world.measure(raw, split="holdout", messages=4096)
                after = world.measure(improved, split="holdout", messages=4096)
                self.assertGreater(before.ber, world.catalog.target_ber)
                self.assertLessEqual(after.ber, world.catalog.target_ber)
                self.assertLessEqual(after.cost, world.catalog.max_cost)

    def test_low_error_over_budget_is_failure(self):
        result = run("inversion", 2, "random", 24)
        self.assertLess(result["holdout_ber"], result["target_ber"])
        self.assertGreater(result["holdout"]["cost"], result["max_cost"])
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
