"""agy-native tool steps become Hermes tool calls; unmappable ones stay neutralized."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import native_tools  # noqa: E402
from client import AntigravityClient  # noqa: E402

TERMINAL_TOOL = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]


def _step(name, params, idx=2):
    return {"step_index": idx, "state": "ACTIVE", "step_type": "tool", "tool_name": name,
            "tool_info": {"name": name, "parameters": params}}


class TranslateTests(unittest.TestCase):
    def test_run_command_maps_to_terminal(self):
        call = native_tools.translate(_step("run_command", {"CommandLine": "echo hi", "Cwd": "/w"}), {"terminal"}, 0)
        self.assertEqual(call.function.name, "terminal")
        self.assertEqual(json.loads(call.function.arguments), {"command": "echo hi", "workdir": "/w"})
        self.assertEqual(call.id, "agy_2_run_command")

    def test_unavailable_hermes_tool_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("run_command", {"CommandLine": "x"}), {"read_file"}, 0))

    def test_unknown_native_tool_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("browser_click_element", {"x": 1}), {"terminal"}, 0))

    def test_missing_parameters_is_not_mapped(self):
        self.assertIsNone(native_tools.translate({"step_type": "tool", "tool_name": "run_command"}, {"terminal"}, 0))

    def test_view_file_and_search(self):
        c = native_tools.translate(_step("view_file", {"AbsolutePath": "/a.py", "StartLine": 5}), {"read_file"}, 0)
        self.assertEqual(json.loads(c.function.arguments), {"path": "/a.py", "offset": 5})
        c = native_tools.translate(_step("grep_search", {"Query": "foo", "SearchPath": "/r"}), {"search_files"}, 0)
        self.assertEqual(json.loads(c.function.arguments), {"pattern": "foo", "target": "content", "path": "/r"})

    def test_write_file_maps_to_write_file(self):
        call = native_tools.translate(_step("write_file", {"Path": "/tmp/foo.txt", "Content": "hello world"}), {"write_file"}, 0)
        self.assertEqual(call.function.name, "write_file")
        self.assertEqual(json.loads(call.function.arguments), {"path": "/tmp/foo.txt", "content": "hello world"})

    def test_read_file_maps_to_read_file(self):
        call = native_tools.translate(_step("read_file", {"Path": "/tmp/foo.txt", "StartLine": 10, "EndLine": 25}), {"read_file"}, 0)
        self.assertEqual(call.function.name, "read_file")
        self.assertEqual(json.loads(call.function.arguments), {"path": "/tmp/foo.txt", "offset": 10, "limit": 16})

    def test_edit_file_maps_to_patch(self):
        call = native_tools.translate(_step("edit_file", {"Path": "/tmp/foo.py", "Target": "old", "Replacement": "new"}), {"patch"}, 0)
        self.assertEqual(call.function.name, "patch")
        self.assertEqual(json.loads(call.function.arguments), {"path": "/tmp/foo.py", "old_string": "old", "new_string": "new"})

    def test_replace_and_multi_replace_file_content_map_to_patch(self):
        call = native_tools.translate(
            _step("replace_file_content", {
                "TargetFile": "/tmp/foo.py",
                "TargetContent": "old",
                "ReplacementContent": "new",
                "AllowMultiple": True,
            }),
            {"patch"},
            0,
        )
        self.assertEqual(call.function.name, "patch")
        self.assertEqual(
            json.loads(call.function.arguments),
            {"path": "/tmp/foo.py", "old_string": "old", "new_string": "new", "replace_all": True},
        )
        chunk_call = native_tools.translate(
            _step("multi_replace_file_content", {
                "TargetFile": "/tmp/foo.py",
                "ReplacementChunks": [{"TargetContent": "a", "ReplacementContent": "b", "AllowMultiple": False}],
            }),
            {"patch"},
            0,
        )
        self.assertEqual(
            json.loads(chunk_call.function.arguments),
            {"path": "/tmp/foo.py", "old_string": "a", "new_string": "b", "replace_all": False},
        )
        # Multi-chunk cannot be represented atomically by a single Hermes patch call.
        self.assertIsNone(native_tools.translate(
            _step("multi_replace_file_content", {
                "TargetFile": "/tmp/foo.py",
                "ReplacementChunks": [
                    {"TargetContent": "a", "ReplacementContent": "b"},
                    {"TargetContent": "c", "ReplacementContent": "d"},
                ],
            }),
            {"patch"},
            0,
        ))

    def test_list_dir_and_search_web_domain(self):
        c = native_tools.translate(_step("list_dir", {"DirectoryPath": "/src"}), {"search_files"}, 0)
        self.assertEqual(json.loads(c.function.arguments), {"pattern": "*", "target": "files", "path": "/src"})
        w = native_tools.translate(_step("search_web", {"query": "asyncio", "domain": "docs.python.org"}), {"web_search"}, 0)
        self.assertEqual(json.loads(w.function.arguments), {"query": "site:docs.python.org asyncio"})

    def test_ask_question_maps_to_clarify(self):
        c = native_tools.translate(
            _step("ask_question", {
                "questions": [{
                    "question": "Which branch?",
                    "options": ["(Recommended) main", "develop"],
                    "is_multi_select": False,
                }],
            }),
            {"clarify"},
            0,
        )
        self.assertEqual(c.function.name, "clarify")
        self.assertEqual(
            json.loads(c.function.arguments),
            {"questions": [{"question": "Which branch?", "choices": ["main", "develop"], "multi_select": False}]},
        )

    def test_subagent_and_process_and_schedule_mappings(self):
        sub = native_tools.translate(
            _step("invoke_subagent", {"Prompt": "Audit auth", "Role": "Security Reviewer"}),
            {"delegate_task"},
            0,
        )
        self.assertEqual(sub.function.name, "delegate_task")
        self.assertEqual(
            json.loads(sub.function.arguments),
            {"tasks": [{"goal": "Audit auth", "context": "Role: Security Reviewer"}]},
        )
        lst = native_tools.translate(_step("manage_subagents", {"Action": "list"}), {"delegate_task"}, 0)
        self.assertEqual(json.loads(lst.function.arguments), {"action": "list"})
        msg = native_tools.translate(
            _step("send_message", {"Recipient": "sub-1", "Message": "Focus on token" }),
            {"delegate_task"},
            0,
        )
        self.assertEqual(json.loads(msg.function.arguments), {"action": "steer", "subagent_id": "sub-1", "message": "Focus on token"})
        pm = native_tools.translate(_step("manage_task", {"Action": "status", "TaskId": "proc_1"}), {"process_manage"}, 0)
        self.assertEqual(json.loads(pm.function.arguments), {"action": "poll", "session_id": "proc_1"})
        cron = native_tools.translate(
            _step("schedule", {"Prompt": "Check health", "CronExpression": "0 * * * *"}),
            {"cronjob_manage"},
            0,
        )
        self.assertEqual(json.loads(cron.function.arguments), {"action": "create", "prompt": "Check health", "schedule": "0 * * * *"})


class StreamBridgeTests(unittest.TestCase):
    def _run(self, events, tools):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = AntigravityClient(cwd=tmp.name)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.side_effect = [json.dumps(e) + "\n" for e in events] + [""] * 50
        proc.stdin = MagicMock()
        # Hermetic: never probe the host keyring (on Linux that probe itself goes through Popen).
        with patch("client.is_authenticated", return_value=True), \
                patch("subprocess.Popen", return_value=proc), patch.object(client, "_terminate_process") as term:
            chunks = list(client.chat.completions.create(model="claude-sonnet-4-6", tools=tools, stream=True,
                                                         messages=[{"role": "user", "content": "run it"}]))
        return chunks, term, proc

    def test_native_run_command_is_reissued_as_hermes_terminal_call(self):
        events = [{"event": "init", "conversation_id": "c"},
                  {"event": "step_update", "step_update": _step("run_command", {"CommandLine": "echo SOUL_42"})}]
        chunks, term, proc = self._run(events, TERMINAL_TOOL)
        term.assert_called_once_with(proc)  # agy never gets to execute it
        calls = [tc for ch in chunks for tc in (ch.choices[0].delta.tool_calls or [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "terminal")
        self.assertEqual(json.loads(calls[0].function.arguments)["command"], "echo SOUL_42")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_unmappable_native_tool_still_neutralized_without_call(self):
        events = [{"event": "step_update", "step_update": _step("browser_click_element", {"x": 1})}]
        chunks, term, proc = self._run(events, TERMINAL_TOOL)
        term.assert_called_once_with(proc)
        calls = [tc for ch in chunks for tc in (ch.choices[0].delta.tool_calls or [])]
        self.assertEqual(calls, [])


def _transcript(root, conversation_id, tool_calls):
    logs = Path(root) / "brain" / conversation_id / ".system_generated" / "logs"
    logs.mkdir(parents=True)
    lines = [json.dumps({"step_index": 0, "type": "USER_INPUT", "content": "x"}),
             json.dumps({"step_index": 1, "type": "PLANNER_RESPONSE", "tool_calls": tool_calls})]
    (logs / "transcript_full.jsonl").write_text("\n".join(lines) + "\n")


class TruncatedPreviewTests(unittest.TestCase):
    """agy stream events cut string params to 512 chars + an ellipsis (live-verified 2026-10-12)."""

    FULL = "python3 -c \"" + "print('x')\n" * 120 + "print('END')\""

    def preview(self):
        return self.FULL[:512] + native_tools.ELLIPSIS

    def test_preview_is_detected_only_at_the_agy_limit(self):
        self.assertEqual(native_tools.truncated_keys({"CommandLine": self.preview(), "Cwd": "/tmp"}), ["CommandLine"])
        self.assertEqual(native_tools.truncated_keys({"CommandLine": "echo hi" + native_tools.ELLIPSIS}), [])

    def test_full_command_is_restored_from_the_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            _transcript(d, "conv1", [{"name": "run_command", "args": {"CommandLine": self.FULL, "Cwd": "/tmp"}}])
            full = native_tools.restore_truncated({"CommandLine": self.preview(), "Cwd": "/tmp"},
                                                  "run_command", "conv1", [Path(d)], wait=0)
        self.assertEqual(full["CommandLine"], self.FULL)

    def test_nested_truncated_chunk_is_restored_from_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            full_chunks = [{"TargetContent": "old", "ReplacementContent": self.FULL}]
            preview_chunks = [{"TargetContent": "old", "ReplacementContent": self.preview()}]
            _transcript(d, "conv_nested", [{"name": "multi_replace_file_content", "args": {"TargetFile": "/a.py", "ReplacementChunks": full_chunks}}])
            self.assertEqual(native_tools.truncated_keys({"TargetFile": "/a.py", "ReplacementChunks": preview_chunks}), ["ReplacementChunks"])
            restored = native_tools.restore_truncated(
                {"TargetFile": "/a.py", "ReplacementChunks": preview_chunks},
                "multi_replace_file_content",
                "conv_nested",
                [Path(d)],
                wait=0,
            )
        assert restored is not None
        self.assertEqual(restored["ReplacementChunks"][0]["ReplacementContent"], self.FULL)

    def test_mismatching_transcript_or_missing_file_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            _transcript(d, "conv1", [{"name": "run_command", "args": {"CommandLine": "rm -rf /tmp/other " + "x" * 600}}])
            self.assertIsNone(native_tools.restore_truncated({"CommandLine": self.preview()}, "run_command",
                                                             "conv1", [Path(d)], wait=0))
            self.assertIsNone(native_tools.restore_truncated({"CommandLine": self.preview()}, "run_command",
                                                             "nope", [Path(d)], wait=0))

    def test_translate_never_emits_a_cut_command(self):
        step = _step("run_command", {"CommandLine": self.preview(), "Cwd": "/tmp"})
        self.assertIsNone(native_tools.translate(step, {"terminal"}, 0))
        self.assertIsNone(native_tools.translate(step, {"terminal"}, 0, resolve=lambda p, n: None))
        call = native_tools.translate(step, {"terminal"}, 0,
                                      resolve=lambda p, n: dict(p, CommandLine=self.FULL))
        self.assertEqual(json.loads(call.function.arguments)["command"], self.FULL)

    def test_transcript_written_slightly_after_the_event_is_waited_for(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            timer = threading.Timer(0.3, _transcript, args=(d, "conv2",
                                    [{"name": "run_command", "args": {"CommandLine": self.FULL}}]))
            timer.start()
            full = native_tools.restore_truncated({"CommandLine": self.preview()}, "run_command",
                                                  "conv2", [Path(d)], wait=2)
            timer.join()
        self.assertEqual(full["CommandLine"], self.FULL)


if __name__ == "__main__":
    unittest.main()
