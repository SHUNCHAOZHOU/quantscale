import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantscale.deepseek_adapter import DeepSeekBackend, completion_url, load_env
from quantscale.research import BudgetExhausted


class DeepSeekTests(unittest.TestCase):
    def backend(self, post, **env):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-key",
                                     "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1/",
                                     "DEEPSEEK_REASONING_EFFORT": "high",
                                     "DEEPSEEK_MAX_OUTPUT_TOKENS": "4096", **env}, clear=True):
            return DeepSeekBackend(env_file="/nonexistent/test.env", post=post)

    def test_env_literal_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"KEEP": "original"}, clear=True):
            path = Path(temp) / ".env"
            path.write_text('export TOKEN="abc#123" # comment\nKEEP=replacement\nLITERAL=\'$(echo never-run)\'\n')
            load_env(path)
            self.assertEqual(os.environ["TOKEN"], "abc#123")
            self.assertEqual(os.environ["KEEP"], "original")
            self.assertEqual(os.environ["LITERAL"], "$(echo never-run)")

    def test_url_normalization(self):
        self.assertEqual(completion_url("https://api.deepseek.com/v1/"), "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(completion_url("https://api.deepseek.com/chat/completions"), "https://api.deepseek.com/chat/completions")
        for value in ("http://api.deepseek.com", "https://user:secret@example.com", "https://example.com?key=secret"):
            with self.assertRaises(ValueError):
                completion_url(value)

    def test_flash_payload_and_actual_usage(self):
        calls = []
        def post(payload):
            calls.append(payload)
            return {"model": "deepseek-flash", "id": "fixture", "system_fingerprint": "revision",
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"hypothesis":"x","strategy":{}}'}}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 700, "total_tokens": 1200,
                              "completion_tokens_details": {"reasoning_tokens": 600}}}
        backend = self.backend(post)
        response, usage = backend.invoke({"max_total_tokens": 16000})
        self.assertEqual(calls[0]["model"], "deepseek-flash")
        self.assertEqual(calls[0]["thinking"], {"type": "enabled"})
        self.assertEqual(usage, 1200)
        self.assertEqual(response["_provenance"]["usage"]["completion_tokens_details"]["reasoning_tokens"], 600)
        self.assertNotIn("fixture-key", json.dumps(response))
        self.assertNotIn("fixture-key", json.dumps(backend.configuration))
        self.assertLessEqual(calls[0]["max_tokens"] + response["_provenance"]["input_reservation"], 16000)

    def test_preflight_no_network_call(self):
        def post(payload):
            self.fail("Must not send a request when allowance is too small")
        with self.assertRaises(BudgetExhausted):
            self.backend(post).invoke({"max_total_tokens": 10})

    def test_incomplete_response_keeps_usage(self):
        backend = self.backend(lambda p: {"choices": [{"finish_reason": "length", "message": {"content": "{"}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 90, "total_tokens": 100}})
        response, usage = backend.invoke({"max_total_tokens": 16000})
        self.assertEqual(usage, 100)
        self.assertIn("adapter_error", response)
        self.assertEqual(response["adapter_error_kind"], "incomplete")
        self.assertEqual(response["raw_content"], "{")

    def test_role_defaults_and_explicit_overrides(self):
        calls = []
        def post(payload):
            calls.append(payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture"}, clear=True):
            backend = DeepSeekBackend(env_file="/nonexistent", post=post)
        researcher, _ = backend.invoke({"role": "researcher", "max_total_tokens": 65536})
        manager, _ = backend.invoke({"role": "manager", "max_total_tokens": 65536})
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(calls[0]["max_tokens"], 32768)
        self.assertEqual(calls[1]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", calls[1])
        self.assertEqual(calls[1]["max_tokens"], 8192)
        self.assertEqual(manager["_provenance"]["reasoning_effort"], "none")
        self.assertEqual(researcher["_provenance"]["applied_max_output_tokens"], 32768)
        backend = self.backend(post, DEEPSEEK_MANAGER_REASONING_EFFORT="low",
                               DEEPSEEK_MANAGER_MAX_OUTPUT_TOKENS="2048")
        backend.invoke({"role": "manager", "max_total_tokens": 16000})
        self.assertEqual(calls[-1]["reasoning_effort"], "low")
        self.assertEqual(calls[-1]["max_tokens"], 2048)

    def test_duplicate_keys_and_malformed_json_preserve_repair_evidence(self):
        backend = self.backend(lambda p: {"choices": [{"finish_reason": "stop", "message": {"content": '{"0":"a","0":"b"}'}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 90, "total_tokens": 100}})
        response, usage = backend.invoke({"role": "manager", "max_total_tokens": 16000})
        self.assertEqual(response["adapter_error_kind"], "json_parse")
        self.assertIn("Duplicate JSON key", response["adapter_error"])
        self.assertEqual(usage, 100)

    def test_usage_required(self):
        backend = self.backend(lambda p: {"usage": {"total_tokens": 1}})
        with self.assertRaisesRegex(RuntimeError, "usage"):
            backend.invoke({"max_total_tokens": 16000})

    def test_non_thinking_mode(self):
        calls = []
        def post(payload):
            calls.append(payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        self.backend(post, DEEPSEEK_REASONING_EFFORT="none").invoke({"max_total_tokens": 16000})
        self.assertEqual(calls[0]["thinking"], {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
