"""Trusted OpenRouter command adapter; JSON request on stdin, JSON result on stdout.

Uses only the standard library; may be run directly with python -I /path/openrouter.py.
"""
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4.1-mini"


class OpenRouterFailure(RuntimeError):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage or {}


def generate(request: dict) -> dict:
    return request_json(request, (
        "You generate research proposals. Return only a JSON object matching the "
        "response_schema in the user request. Do not include examples or commentary. "
        "You have no experiment history. Do not claim your proposals were tested."), {'constructions'})


def request_json(request: dict, system: str, expected_keys: set[str], *, schema=None,
                 temperature=0.8, max_tokens=4096) -> dict:
    """Shared transport; different agents keep independent prompts and histories."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    payload = {
        "model": os.environ.get("EDISON_MODEL") or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(request)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "reasoning": {"effort": "low"},
        "temperature": temperature,
    }
    if schema is not None:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'scientist_response', 'strict': True, 'schema': schema}}
        # Route only to providers that support the requested parameters.
        # An unsupported model must fail explicitly, never silently downgrade.
        payload['provider'] = {'require_parameters': True}
    http_request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(http_request, timeout=25) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise RuntimeError("OpenRouter response is too large")
        data = json.loads(raw)
        choice = data["choices"][0]
        usage = {k: v for k, v in data.get('usage', {}).items() if k in
                 ('prompt_tokens', 'completion_tokens', 'total_tokens') and type(v) is int}
        if choice.get("finish_reason") == "length":
            raise OpenRouterFailure("OpenRouter response token limit", usage)
        content = json.loads(choice["message"]["content"])
        if not isinstance(content, dict) or set(content) != expected_keys:
            raise ValueError("Unexpected generated object")
        content["usage"] = usage
        return content
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("OpenRouter connection failed") from None
    except (KeyError, IndexError, ValueError, TypeError):
        raise RuntimeError("Invalid OpenRouter response") from None


def main():
    try:
        raw = sys.stdin.buffer.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Request is too large")
        print(json.dumps(generate(json.loads(raw))))
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
