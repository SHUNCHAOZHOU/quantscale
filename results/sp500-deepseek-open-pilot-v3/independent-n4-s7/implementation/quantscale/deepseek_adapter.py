"""DeepSeek Flash backend with actual usage accounting and local .env loading.

https://api-docs.deepseek.com/api/create-chat-completion/
The preflight input reservation is conservative, not a provider token count.
"""
import json
import os
from pathlib import Path
import shlex
import socket
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .research import BudgetExhausted
from .prompts import system_instructions
from .manager_protocol import strict_json_object


def load_env(path=".env"):
    """Load literal KEY=VALUE entries; never execute shell or expand variables."""
    path = Path(path)
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            raise ValueError("Invalid .env entry on line %d" % number)
        values = shlex.split(value, comments=True, posix=True)
        if len(values) > 1:
            raise ValueError("Quote whitespace in .env entry on line %d" % number)
        os.environ.setdefault(key, values[0] if values else "")


def completion_url(base_url):
    parts = urlsplit(base_url.strip())
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ValueError("DEEPSEEK_BASE_URL must be an HTTPS URL without credentials, query or fragment")
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward Authorization to a redirect target.
        return None


class DeepSeekBackend:
    name = "deepseek-flash"
    token_accounting = "provider actual total_tokens; conservative UTF-8 input reservation, not an exact token count"

    def __init__(self, env_file=".env", post=None):
        load_env(env_file)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set DEEPSEEK_API_KEY in .env or the process environment")
        self.url = completion_url(os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.roles = {}
        for role, default_effort, default_output in (("researcher", "low", "32768"), ("manager", "none", "8192")):
            prefix = "DEEPSEEK_" + role.upper() + "_"
            effort = os.environ.get(prefix + "REASONING_EFFORT", os.environ.get("DEEPSEEK_REASONING_EFFORT", default_effort))
            output = int(os.environ.get(prefix + "MAX_OUTPUT_TOKENS", os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", default_output)))
            if effort not in ("none", "low", "high", "max"):
                raise ValueError(prefix + "REASONING_EFFORT must be none, low, high or max")
            if not 64 <= output <= 393216:
                raise ValueError(prefix + "MAX_OUTPUT_TOKENS must be 64..393216")
            self.roles[role] = {"reasoning_effort": effort, "max_output_tokens": output}
        self.effort = self.roles["researcher"]["reasoning_effort"]
        self.max_output_tokens = self.roles["researcher"]["max_output_tokens"]
        self.post = post or self._post
        self.configuration = {"provider": "deepseek", "model": self.name,
                              "reasoning_effort": self.effort, "max_output_tokens": self.max_output_tokens,
                              "role_settings": self.roles,
                              "input_budget_preflight": "UTF-8 JSON bytes + 1024 overhead reservation; estimated"}

    def _post(self, payload):
        request = Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(),
                          headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
                          method="POST")
        try:
            with build_opener(NoRedirect()).open(request, timeout=150) as response:
                return json.load(response)
        except HTTPError as exc:
            # No provider response bodies/URLs: gateways can echo credentials.
            raise RuntimeError("DeepSeek HTTP %d; no automatic retry; no frozen submission produced" % exc.code) from None
        except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError):
            raise RuntimeError("DeepSeek network/TLS failure; no automatic retry; usage may be uncertain") from None

    def invoke(self, request):
        role = request.get("role", "researcher")
        if role not in self.roles:
            raise ValueError("Unknown DeepSeek role")
        settings = self.roles[role]
        effort = settings["reasoning_effort"]
        instructions = system_instructions(request.get("strategy_mode", "python"))
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False, sort_keys=True)}]
        # No exact provider counting endpoint is assumed. Keep a deliberately large
        # reserve; actual response usage, including reasoning, remains authoritative.
        reserved = len(json.dumps(messages, ensure_ascii=False).encode()) + 1024
        available = request["max_total_tokens"] - reserved
        if available < 64:
            raise BudgetExhausted("Remaining allowance is smaller than the conservative DeepSeek input reservation")
        payload = {"model": self.name, "messages": messages, "stream": False,
                   "thinking": {"type": "disabled" if effort == "none" else "enabled"},
                   "max_tokens": min(settings["max_output_tokens"], available),
                   "response_format": {"type": "json_object"}}
        if effort != "none":
            payload["reasoning_effort"] = effort
        raw = self.post(payload)
        usage = raw.get("usage", {})
        total = usage.get("total_tokens")
        if (type(total) is not int or total <= 0
                or any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens"))
                or total != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise RuntimeError("DeepSeek returned missing/inconsistent usage; run aborted")
        response = {}
        finish_reason = None
        content = None
        try:
            choice = raw["choices"][0]
            finish_reason = choice["finish_reason"]
            content = choice["message"]["content"]
            if finish_reason != "stop":
                raise ValueError("Model response did not finish: " + str(finish_reason))
            response = strict_json_object(content)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            response = {"adapter_error": str(exc), "finish_reason": finish_reason,
                        "adapter_error_kind": "incomplete" if finish_reason != "stop" else "json_parse"}
            if isinstance(content, str):
                response["raw_content"] = content
        response["_provenance"] = {**self.configuration, "served_model": raw.get("model"),
                                   "role": role, **settings, "applied_max_output_tokens": payload["max_tokens"],
                                   "response_id": raw.get("id"), "system_fingerprint": raw.get("system_fingerprint"),
                                   "usage": usage, "input_reservation": reserved}
        # Research loop charges usage before parsing strategy/errors and aborts on
        # any actual overrun. The reservation is explicitly not a hard guarantee.
        return response, total
