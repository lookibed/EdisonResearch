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


def generate(request: dict) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    payload = {
        "model": os.environ.get("EDISON_MODEL") or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You generate research proposals. Return only a JSON object matching the "
                "response_schema in the user request. Do not include examples or commentary. "
                "You have no experiment history. Do not claim your proposals were tested.")},
            {"role": "user", "content": json.dumps(request)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
        "temperature": 0.8,
    }
    http_request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(http_request, timeout=25) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise RuntimeError("OpenRouter response is too large")
        data = json.loads(raw)
        content = json.loads(data["choices"][0]["message"]["content"])
        if not isinstance(content, dict) or set(content) != {"constructions"}:
            raise ValueError("Unexpected generated object")
        usage = data.get("usage", {})
        content["usage"] = {k: usage[k] for k in
                            ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
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
