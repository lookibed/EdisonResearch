import io
import json
import os
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from edison.evaluator import EvaluatorProcess
from edison.ideas import Construction, Primitives, Proposal
from edison.openrouter import ENDPOINT, generate
from edison.pipeline import run
from edison.proposer import Batch, CommandProposer, ProposalFailure, parse_batch, request_for
from edison.research import run as in_process_run


class FakeProposer:
    def __init__(self):
        self.parents = []

    def propose(self, parent, primitives):
        self.parents.append(parent)
        # Multi-edit proposal that cannot be a depth-one structural child.
        return Batch((Proposal(Construction(path=(primitives.nodes[0],) * 3,
                                            repetitions=9, flip=True), "model_proposal"),),
                     {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


class FailedProposer:
    def propose(self, parent, primitives):
        raise TimeoutError("private provider diagnostic must not enter report")


class EvaluatorTests(unittest.TestCase):
    def test_process_matches_existing_evaluator(self):
        reference = in_process_run("coding", 3, "fair", 12, 64, 256)
        actual = run("coding", 3, 12, probe_messages=64, holdout_messages=256)
        self.assertEqual(actual["selected"], reference["selected"])
        self.assertEqual(actual["holdout"], reference["holdout"])
        self.assertEqual(actual["observations"], reference["observations"])
        self.assertEqual(actual["generator_status"], "disabled")

    def test_private_operations_are_rejected_and_finish_closes(self):
        with EvaluatorProcess("relay", 1, 2, 32, 128) as evaluator:
            self.assertNotEqual(evaluator._process.pid, os.getpid())
            self.assertEqual(set(evaluator.public), {"primitives", "max_cost", "target_ber"})
            with self.assertRaises(RuntimeError):
                evaluator._call({"op": "get_world"})
            with self.assertRaises(RuntimeError):
                evaluator.probe(Construction(path=("unknown",)))
            root = Construction()
            first = evaluator.probe(root)
            self.assertEqual(first, evaluator.probe(root))
            report = evaluator.finish()
            self.assertEqual(len(report["observations"]), 1)
            self.assertEqual(report["duplicate_requests"], 1)
            with self.assertRaises(RuntimeError):
                evaluator.finish()
            with self.assertRaises(RuntimeError):
                evaluator.probe(root)
        self.assertFalse(evaluator._process.is_alive())


class ProposerTests(unittest.TestCase):
    def setUp(self):
        self.primitives = Primitives(("n1", "n2"), ("t1", "t2"))

    def test_request_has_only_public_inputs_and_no_solution_examples(self):
        request = request_for(Construction(), self.primitives)
        self.assertEqual(set(request), {"protocol", "task", "parent", "primitives",
                                        "max_proposals", "response_schema"})
        self.assertNotIn("examples", request)
        for key in ("seed", "profile", "observations", "holdout", "ratings", "usage"):
            self.assertNotIn(key, request)

    def test_batch_validation_rejects_unknown_tokens_even_repetitions_and_extra_fields(self):
        valid = {"path": [], "transforms": [], "repetitions": 1, "flip": False}
        for invalid in (dict(valid, repetitions=2), dict(valid, repetitions=True),
                        dict(valid, path=["unknown"]), dict(valid, flip="false"),
                        dict(valid, execute="code")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_batch({"constructions": [invalid]}, self.primitives)
        with self.assertRaises(ValueError):
            parse_batch({"constructions": [valid] * 9}, self.primitives)
        self.assertEqual(parse_batch({"constructions": [valid]}, self.primitives)
                         .proposals[0].construction, Construction())

    def test_external_command_receives_filtered_environment_and_public_request(self):
        code = (
            "import json,os,sys; r=json.load(sys.stdin); "
            "assert 'EDISON_PRIVATE_TEST_SENTINEL' not in os.environ; "
            "assert 'GH_TOKEN' not in os.environ; "
            "assert set(r)=={'protocol','task','parent','primitives','max_proposals','response_schema'}; "
            "c=r['parent']; c['repetitions']=3; print(json.dumps({'constructions':[c]}))"
        )
        with patch.dict(os.environ, {"EDISON_PRIVATE_TEST_SENTINEL": "test-only"}):
            batch = CommandProposer([sys.executable, "-I", "-c", code]).propose(
                Construction(), self.primitives)
        self.assertEqual(batch.proposals[0].construction.repetitions, 3)

    def test_timeout_and_malformed_response(self):
        with self.assertRaises(TimeoutError):
            CommandProposer([sys.executable, "-c", "import time; time.sleep(1)"], .05).propose(
                Construction(), self.primitives)
        with self.assertRaises(ValueError):
            CommandProposer([sys.executable, "-c", "print('not json')"]).propose(
                Construction(), self.primitives)

    def test_adapter_status_is_preserved_without_arbitrary_diagnostics(self):
        for diagnostic, expected in (("OpenRouter HTTP 402", "openrouter_http_402"),
                                     ("private response", "proposal_command_failed")):
            code = f"import sys; print({diagnostic!r}, file=sys.stderr); sys.exit(1)"
            with self.subTest(diagnostic=diagnostic), self.assertRaises(ProposalFailure) as error:
                CommandProposer([sys.executable, "-I", "-c", code]).propose(
                    Construction(), self.primitives)
            self.assertEqual(error.exception.code, expected)

    def test_openrouter_http_payload_and_usage_without_live_network(self):
        content = {"constructions": [{"path": [], "transforms": [], "repetitions": 3, "flip": False}]}
        response = {"choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only", "EDISON_MODEL": "test/model"}):
            with patch("edison.openrouter.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as http:
                result = generate(request_for(Construction(), self.primitives))
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, ENDPOINT)
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test/model")
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(result["usage"]["total_tokens"], 10)
        self.assertEqual(parse_batch(result, self.primitives).proposals[0].construction.repetitions, 3)

    def test_openrouter_errors_do_not_expose_response_body(self):
        error = HTTPError(ENDPOINT, 401, "private diagnostic", {}, io.BytesIO(b"private response"))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only"}):
            with patch("edison.openrouter.urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "^OpenRouter HTTP 401$"):
                    generate(request_for(Construction(), self.primitives))

    def test_reasoning_exhausting_output_budget_has_explicit_error(self):
        response = {"choices": [{"finish_reason": "length", "message": {"content": None}}]}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only"}):
            with patch("edison.openrouter.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaisesRegex(RuntimeError, "^OpenRouter response token limit$"):
                    generate(request_for(Construction(), self.primitives))


class PipelineTests(unittest.TestCase):
    def test_real_command_adapter_drives_process_separated_pipeline(self):
        code = ("import json,sys; r=json.load(sys.stdin); c=r['parent']; "
                "c['repetitions']+=4; print(json.dumps({'constructions':[c]}))")
        proposer = CommandProposer([sys.executable, "-I", "-c", code])
        report = run("coding", 1, 12, proposer, 2, 32, 128)
        self.assertEqual(report["generator_status"], "ok")
        self.assertEqual(report["model_calls_succeeded"], 2)
        self.assertGreater(report["model_origin_tested"], 0)
        self.assertEqual(len(report["observations"]), 12)

    def test_external_suggestions_are_tested_and_do_not_remove_sibling_branches(self):
        proposer = FakeProposer()
        report = run("relay", 4, 20, proposer, 4, 32, 128)
        self.assertEqual(report["generator_status"], "ok")
        self.assertEqual(report["model_calls_succeeded"], 4)
        self.assertEqual(report["model_origin_tested"], 1)
        tested = {o["idea_id"] for o in report["observations"]}
        self.assertTrue(set(range(7)).issubset(tested))
        self.assertTrue(report["graph"]["pending"])
        self.assertEqual(report["duplicate_requests"], 0)
        self.assertEqual(len(report["observations"]), 20)
        for event in report["generator_events"]:
            self.assertNotIn("observations", event["request"])

    def test_provider_failure_keeps_structural_search_and_is_not_model_success(self):
        baseline = run("coding", 0, 12, probe_messages=32, holdout_messages=128)
        fallback = run("coding", 0, 12, FailedProposer(), 2, 32, 128)
        self.assertEqual(fallback["observations"], baseline["observations"])
        self.assertEqual(fallback["graph"], baseline["graph"])
        self.assertEqual(fallback["generator_status"], "failed")
        self.assertEqual(fallback["model_calls_succeeded"], 0)
        self.assertNotIn("private provider diagnostic", json.dumps(fallback))


if __name__ == "__main__":
    unittest.main()
