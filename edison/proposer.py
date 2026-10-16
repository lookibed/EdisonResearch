"""External data-only proposal generator. Never receives experiment outcomes."""
from dataclasses import asdict, dataclass
import json
import os
import re
import subprocess
import tempfile
from .ideas import Construction, Primitives, Proposal
from .wire import construction_from_json, construction_to_json


def request_for(parent: Construction, primitives: Primitives) -> dict:
    return {
        "protocol": "edison-proposals-v1",
        "task": ("Propose alternative constructions for reliable bit transmission at low cost. "
                 "Primitive properties are unknown. Do not assume a named mechanism is present. "
                 "A path lists successive intermediate nodes; transforms are applied in order "
                 "and reversed by the receiver; repetitions use majority voting; flip inverts "
                 "received bits before decoding. You may compose multiple edits. "
                 "No experimental results are available to you."),
        "parent": construction_to_json(parent), "primitives": asdict(primitives),
        "max_proposals": 8,
        "response_schema": {
            "type": "object", "additionalProperties": False, "required": ["constructions"],
            "properties": {"constructions": {
                "type": "array", "maxItems": 8, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["path", "transforms", "repetitions", "flip"],
                    "properties": {
                        "path": {"type": "array", "items": {"enum": list(primitives.nodes)}},
                        "transforms": {"type": "array", "items": {"enum": list(primitives.transforms)}},
                        "repetitions": {"type": "integer", "minimum": 1,
                                        "description": "Must be odd"},
                        "flip": {"type": "boolean"},
                    },
                },
            }},
        },
    }


@dataclass(frozen=True)
class Batch:
    proposals: tuple[Proposal, ...]
    usage: dict[str, int]


class ProposalFailure(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def parse_batch(data: object, primitives: Primitives) -> Batch:
    if (not isinstance(data, dict) or set(data) - {"constructions", "usage"}
            or type(data.get("constructions")) is not list or len(data["constructions"]) > 8):
        raise ValueError("Invalid proposal batch")
    proposals = tuple(Proposal(construction_from_json(c, primitives), "model_proposal")
                      for c in data["constructions"])
    usage = data.get("usage", {})
    if (not isinstance(usage, dict)
            or any(k not in ("prompt_tokens", "completion_tokens", "total_tokens")
                   or type(v) is not int or v < 0 for k, v in usage.items())):
        raise ValueError("Invalid usage metadata")
    return Batch(proposals, usage)


class CommandProposer:
    """A trusted adapter command, not an arbitrary-code sandbox.

    Fresh process, temporary cwd, filtered environment and no shell interpolation.
    stdout/stderr go to files; only bounded JSON stdout is loaded into memory.
    Provider diagnostics are not copied into logs (they could contain credentials).
    """

    def __init__(self, command: list[str], timeout: float = 30):
        if not command or timeout <= 0:
            raise ValueError("Command and positive timeout required")
        self.command, self.timeout = command, timeout

    def propose(self, parent: Construction, primitives: Primitives) -> Batch:
        allowed = {"PATH", "LANG", "LC_ALL", "OPENROUTER_API_KEY", "EDISON_MODEL",
                   "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                   "http_proxy", "https_proxy", "all_proxy", "no_proxy",
                   "SSL_CERT_FILE", "SSL_CERT_DIR"}
        environment = {k: v for k, v in os.environ.items() if k in allowed}
        with tempfile.TemporaryDirectory(prefix="edison-proposal-") as cwd:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                try:
                    result = subprocess.run(
                        self.command, input=json.dumps(request_for(parent, primitives)).encode(),
                        stdout=stdout, stderr=stderr, cwd=cwd, env=environment,
                        timeout=self.timeout, check=False)
                except subprocess.TimeoutExpired:
                    raise TimeoutError("Proposal command timed out") from None
                except OSError:
                    raise RuntimeError("Proposal command could not start") from None
                if result.returncode:
                    stderr.seek(0)
                    diagnostic = stderr.read(256).decode(errors="replace").strip()
                    # Only fixed, credential-free diagnostic categories cross the boundary.
                    match = re.fullmatch(r"OpenRouter HTTP ([45][0-9]{2})", diagnostic)
                    code = ("openrouter_http_" + match.group(1) if match else {
                        "OPENROUTER_API_KEY is not configured": "missing_api_key",
                        "OpenRouter connection failed": "provider_connection_failed",
                        "Invalid OpenRouter response": "invalid_provider_response",
                        "OpenRouter response token limit": "provider_token_limit",
                    }.get(diagnostic, "proposal_command_failed"))
                    raise ProposalFailure(code)
                stdout.seek(0)
                raw = stdout.read(1_048_577)
                if len(raw) > 1_048_576:
                    raise ValueError("Proposal response is too large")
        try:
            return parse_batch(json.loads(raw), primitives)
        except (ValueError, TypeError, UnicodeDecodeError):
            raise ValueError("Invalid proposal response") from None
