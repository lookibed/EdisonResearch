from dataclasses import replace
from itertools import product
import unittest

from edison.composition import CompositionWorld, ExperimentMemory
from edison.ideas import Branches, Construction, Generator, Primitives
from edison.research import run


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.primitives = Primitives(("a", "b"), ("c", "d"))

    def test_fifo_covers_every_construction_through_depth_three(self):
        expected = set()
        for path_size in range(4):
            for transform_size in range(4):
                for repeat_steps in range(4):
                    for flip in (False, True):
                        if path_size + transform_size + repeat_steps + int(flip) > 3:
                            continue
                        for path in product(self.primitives.nodes, repeat=path_size):
                            for transforms in product(self.primitives.transforms, repeat=transform_size):
                                expected.add(Construction(path, transforms, 1 + 2 * repeat_steps, flip))
        branches = Branches(self.primitives)
        visited = [branches.next() for _ in expected]
        self.assertEqual({n.construction for n in visited}, expected)
        self.assertEqual(len({n.id for n in visited}), len(visited))
        self.assertEqual([n.discovery_depth for n in visited],
                         sorted(n.discovery_depth for n in visited))
        self.assertTrue(branches.pending)

    def test_duplicate_keeps_both_origins_and_back_edges(self):
        branches = Branches(self.primitives)
        for _ in range(29):
            branches.next()
        target = Construction(("a",), ("c",))
        node = next(n for n in branches.graph.nodes if n.construction == target)
        parents = {edge.parent for edge in branches.graph.edges if edge.child == node.id}
        self.assertGreaterEqual(len(parents), 2)
        self.assertEqual(sum(n.construction == target for n in branches.graph.nodes), 1)
        self.assertTrue(any(e.child == 0 and e.parent != 0 for e in branches.graph.edges))
        self.assertGreater(branches.graph.duplicate_proposals, 0)

    def test_no_fixed_limits_on_composition_depth_or_repetitions(self):
        parent = Construction(("a",) * 20, ("c",) * 20, 101)
        children = Generator().expand(parent, self.primitives)
        self.assertIn(replace(parent, repetitions=103), [p.construction for p in children])
        self.assertIn(replace(parent, path=parent.path + ("b",)),
                      [p.construction for p in children])

    def test_measurement_changes_do_not_change_generation_or_pending_branches(self):
        def trace(world):
            branches = Branches(world.primitives)
            memory = ExperimentMemory(world, 30, 32)
            order = []
            for _ in range(30):
                idea = branches.next()
                order.append(idea.construction)
                memory.probe(idea.construction)
            return order, list(branches.pending), memory.observations

        normal = CompositionWorld("inversion", 42)
        broken = CompositionWorld("inversion", 42)
        broken._links = {key: (.5, False) for key in broken._links}
        first, second = trace(normal), trace(broken)
        self.assertEqual(first[:2], second[:2])
        self.assertNotEqual(first[2], second[2])


class CompositionTests(unittest.TestCase):
    def test_nested_transforms_roundtrip_with_padding(self):
        world = CompositionWorld("coding", 0)
        world._links = {key: (0., False) for key in world._links}
        hamming = next(k for k, v in world._transforms.items() if v == "hamming")
        construction = Construction(transforms=(hamming, hamming, hamming))
        result = world.measure(construction, split="holdout", messages=128)
        self.assertEqual(result.measurement.bit_errors, 0)
        # 4 -> 7 -> 14 -> 28 bits; intermediate padding must be removed on decode.
        self.assertEqual(result.measurement.cost, 7)

    def test_composed_solution_requires_relay_transform_and_inversion(self):
        world = CompositionWorld("combined", 0)
        relay = next(k for k in world.primitives.nodes if world._links["source", k][0] == .02)
        hamming = next(k for k, v in world._transforms.items() if v == "hamming")
        solution = Construction((relay,), (hamming,), flip=True)
        result = world.measure(solution, split="holdout", messages=4096).measurement
        self.assertLessEqual(result.ber, world.target_ber)
        self.assertLessEqual(result.cost, world.max_cost)
        for ablation in (replace(solution, path=()), replace(solution, transforms=()),
                         replace(solution, flip=False)):
            with self.subTest(ablation=ablation):
                self.assertGreater(world.measure(ablation, split="holdout", messages=4096)
                                   .measurement.ber, world.target_ber)

    def test_resource_limit_is_recorded_without_discarding_idea(self):
        world = CompositionWorld("coding", 0, max_transmitted_bits=100)
        memory = ExperimentMemory(world, 2, 16)
        construction = Construction(repetitions=101)
        outcome = memory.probe(construction)
        self.assertIsNone(outcome.measurement)
        self.assertEqual(outcome.reason, "resource_limit")
        self.assertEqual(len(memory.observations), 1)
        self.assertEqual(memory.probe(construction), outcome)
        self.assertEqual(memory.duplicate_requests, 1)
        self.assertEqual(len(memory.observations), 1)
        # An unexecutable idea does not stop the next ordinary experiment.
        self.assertIsNotNone(memory.probe(Construction()).measurement)

    def test_budget_validation_and_holdout_closure(self):
        memory = ExperimentMemory(CompositionWorld("relay", 0), 1, 32)
        with self.assertRaises(ValueError):
            memory.probe(Construction(path=("unknown",)))
        self.assertEqual(len(memory.observations), 0)
        root = Construction()
        memory.probe(root)
        with self.assertRaises(RuntimeError):
            memory.probe(replace(root, flip=True))
        with self.assertRaises(ValueError):
            memory.finalize(replace(root, flip=True))
        held = memory.finalize(root, 32)
        self.assertNotEqual(held, memory.observations[0][1])
        with self.assertRaises(RuntimeError):
            memory.finalize(root, 32)
        with self.assertRaises(RuntimeError):
            memory.probe(root)

    def test_fifo_rediscovers_each_profile(self):
        for profile in CompositionWorld.PROFILES:
            with self.subTest(profile=profile):
                result = run(profile, 0, "fair", 96, 512, 4096)
                self.assertTrue(result["success"], result["selected"])
                self.assertEqual(result["attempts"], 96)
                self.assertEqual(result["duplicate_requests"], 0)
                self.assertEqual(len(result["observations"]), 96)
                self.assertTrue(result["graph"]["pending"])

    def test_complete_reproduction_including_graph_and_experiments(self):
        self.assertEqual(run("coding", 5, "fair", 8, 32, 128),
                         run("coding", 5, "fair", 8, 32, 128))


if __name__ == "__main__":
    unittest.main()
