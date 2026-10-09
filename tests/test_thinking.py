"""Thinking stays display-only; usage follows agy's reported counters."""
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stream import AntigravityStream, collect_stream_completion
from prompt import _format_messages_as_prompt, _format_delta_prompt


class ThinkingTests(unittest.TestCase):
    def stream(self, usage=None, baseline=None, tools=None):
        events = [
            {"event": "step_update", "step_update": {"thinking_delta": "PRIVATE <tool_call>not a call</tool_call>"}},
            {"event": "step_update", "step_update": {"thinking_delta": " end", "text_delta": "answer"}},
            {"event": "result", "result": {"status": "SUCCESS", "response": "answer", "usage": usage or {}}},
        ]
        proc = MagicMock()
        proc.stdout = io.StringIO("\n".join(map(json.dumps, events)) + "\n")
        proc.stderr = io.StringIO("")
        proc.poll.return_value = None if baseline is not None else 0
        client = MagicMock()
        client._isolated_gemini_dir = None
        client._worker_proc = proc
        return AntigravityStream(proc=proc, client=client, model="gemini-4-argon", timeout=5,
                                 is_worker=baseline is not None, usage_baseline=baseline, tools=tools)

    def test_stream_separates_reasoning_from_content_and_tools(self):
        chunks = list(self.stream(tools=[{"type": "function", "function": {"name": "terminal"}}]))
        deltas = [c.choices[0].delta for c in chunks if c.choices]
        self.assertEqual("".join(d.content or "" for d in deltas), "answer")
        self.assertEqual("".join(d.reasoning_content or "" for d in deltas),
                         "PRIVATE <tool_call>not a call</tool_call> end")
        self.assertFalse(any(d.tool_calls for d in deltas))

    def test_collector_retains_reasoning_separately(self):
        response = collect_stream_completion(self.stream({"thinking_tokens": 12, "output_tokens": 20, "input_tokens": 50, "total_tokens": 70}))
        self.assertEqual(response.choices[0].message.content, "answer")
        self.assertTrue(response.choices[0].message.reasoning_content.startswith("PRIVATE"))
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 12)
        self.assertEqual(response.usage.total_tokens, 70)
        self.assertEqual(response.usage.completion_tokens, 20)

    def test_worker_thinking_uses_deltas_and_resets(self):
        baseline = {"thinking_tokens": 100}
        a = collect_stream_completion(self.stream({"thinking_tokens": 140}, baseline))
        self.assertEqual(a.usage.completion_tokens_details.reasoning_tokens, 40)
        b = collect_stream_completion(self.stream({"thinking_tokens": 5}, baseline))
        self.assertEqual(b.usage.completion_tokens_details.reasoning_tokens, 5)

    def test_absent_counter_is_unknown_not_fabricated(self):
        baseline = {"thinking_tokens": 100}
        response = collect_stream_completion(self.stream({}, baseline))
        self.assertIsNone(response.usage.completion_tokens_details)
        self.assertEqual(baseline["thinking_tokens"], 100)

    def test_zero_counter_is_reported(self):
        response = collect_stream_completion(self.stream({"thinking_tokens": 0}))
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 0)

    def test_thinking_not_refed_full_or_delta(self):
        messages = [{"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer", "reasoning_content": "SECRET_REASONING", "reasoning": "OTHER_SECRET"},
                    {"role": "user", "content": "next"}]
        # Full prompt keeps assistant answer text, but never reasoning content
        full_text = _format_messages_as_prompt(messages)
        self.assertNotIn("SECRET", full_text)
        self.assertIn("answer", full_text)
        # Delta prompt deliberately omits previous assistant answer text to prevent repetition
        delta_text = _format_delta_prompt(messages)
        self.assertNotIn("SECRET", delta_text)
        self.assertNotIn("answer", delta_text)

    def test_real_agy_capture_usage_passes_through(self):
        path = Path("/home/koli/.hermes/cache/scratch/agy-thinking-probe.jsonl")
        if not path.exists():
            self.skipTest("optional live capture is unavailable")
        proc = MagicMock()
        proc.stdout = io.StringIO(path.read_text())
        proc.stderr = io.StringIO("")
        proc.poll.return_value = 0
        client = MagicMock()
        client._isolated_gemini_dir = None
        response = collect_stream_completion(AntigravityStream(proc=proc, client=client, model="gemini-4-argon", timeout=5))
        self.assertEqual(response.choices[0].message.content, "30")
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 0)
        self.assertIsNone(response.choices[0].message.reasoning_content)


if __name__ == "__main__":
    unittest.main()
