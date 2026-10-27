"""OpenAI Responses API adapter; standard library only, no automatic retries.

Run with: python3 -m quantscale.openai_adapter
Required environment: OPENAI_API_KEY, QUANTSCALE_MODEL (pin a model snapshot).
Docs: https://developers.openai.com/api/docs/guides/token-counting
"""
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post_json(path, payload):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY for the OpenAI adapter")
    request = Request("https://api.openai.com/v1/" + path,
                      data=json.dumps(payload).encode(),
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                      method="POST")
    try:
        with urlopen(request, timeout=150) as response:
            return json.load(response)
    except HTTPError as exc:
        # Do not include request headers or credentials in errors/logs.
        raise RuntimeError("OpenAI HTTP %d; no automatic retry (usage may be uncertain)" % exc.code) from None


def invoke(request, model, post=post_json):
    instructions = (
        "You are a QuantScale quantitative researcher. Return exactly one JSON object, without markdown. "
        "Use only supplied information. No model training. All researcher proposals must use the strategy_schema. "
        "Researcher output: {\"hypothesis\":\"reason and proposed revision\",\"strategy\":{"
        "\"family\":\"momentum\",\"lookback\":60,\"skip\":1,\"fraction\":0.25,\"volatility_scale\":false}}. "
        "Manager output: {\"directions\":{\"0\":\"momentum\",\"1\":\"reversal\"}}, with one entry per worker. "
        "Manager directions must be a family from strategy_schema.family. "
        "Independent agents only see their own results; preserve the given research protocol."
    )
    base = {"model": model, "instructions": instructions, "input": json.dumps(request, sort_keys=True)}
    count = post("responses/input_tokens", base)["input_tokens"]
    if type(count) is not int or count < 0:
        raise ValueError("Invalid provider input-token count")
    available = request["max_total_tokens"] - count
    if available < 64:
        return {"budget_exhausted": True, "usage": {"total_tokens": 0}, "response": {}}
    raw = post("responses", {**base, "max_output_tokens": available, "store": False})
    usage = raw["usage"]
    texts = [content["text"] for item in raw.get("output", []) if item.get("type") == "message"
             for content in item.get("content", []) if content.get("type") == "output_text"]
    try:
        if raw.get("status") != "completed":
            raise ValueError("Incomplete model response: " + str(raw.get("incomplete_details")))
        response = json.loads("".join(texts))
        if not isinstance(response, dict):
            raise ValueError("Response must be a JSON object")
    except ValueError as exc:
        # Preserve usage even for truncated/refused/malformed model responses.
        response = {"adapter_error": str(exc)}
    return {"response": response, "usage": usage,
            "provenance": {"model": raw.get("model", model), "response_id": raw.get("id")}}


def main():
    try:
        model = os.environ.get("QUANTSCALE_MODEL")
        if not model:
            raise ValueError("Set QUANTSCALE_MODEL to the exact model used for this experiment")
        result = invoke(json.load(sys.stdin), model)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print("OpenAI adapter: " + str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
