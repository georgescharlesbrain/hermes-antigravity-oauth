"""Translate agy-native tool steps into Hermes tool calls.

Some Antigravity models (Claude in particular) ignore the ``<tool_call>`` text protocol and call agy's
built-in tools such as ``run_command``. The plugin never lets agy execute those on the host. Before
this module, the step was killed and the turn came back empty. Now the intent is re-issued as the
equivalent Hermes tool call, so Hermes' own tool, approval and sandbox layer runs it. Unmappable
native tools are still neutralized as before.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

Mapper = Callable[[dict[str, Any]], dict[str, Any] | None]


def _pick(params: dict[str, Any], *names: str) -> Any:
    lowered = {str(k).lower(): v for k, v in params.items()}
    for n in names:
        v = lowered.get(n.lower())
        if v not in (None, ""):
            return v
    return None


def _terminal(p: dict[str, Any]) -> dict[str, Any] | None:
    cmd = _pick(p, "CommandLine", "command", "cmd")
    if not cmd:
        return None
    out: dict[str, Any] = {"command": str(cmd)}
    if cwd := _pick(p, "Cwd", "cwd", "WorkingDirectory"):
        out["workdir"] = str(cwd)
    return out


def _read_file(p: dict[str, Any]) -> dict[str, Any] | None:
    path = _pick(p, "Path", "AbsolutePath", "path", "file_path", "FilePath")
    if not path:
        return None
    out: dict[str, Any] = {"path": str(path)}
    if (start := _pick(p, "StartLine", "offset", "start_line")) is not None:
        try:
            out["offset"] = max(1, int(start))
        except (TypeError, ValueError):
            pass
    if (limit := _pick(p, "limit", "Limit")) is not None:
        try:
            out["limit"] = int(limit)
        except (TypeError, ValueError):
            pass
    elif (end := _pick(p, "EndLine", "end_line", "End")) is not None:
        try:
            start_val = out.get("offset", 1)
            end_val = int(end)
            if end_val >= start_val:
                out["limit"] = end_val - start_val + 1
        except (TypeError, ValueError):
            pass
    return out


def _write_file(p: dict[str, Any]) -> dict[str, Any] | None:
    path = _pick(p, "Path", "TargetFile", "path", "AbsolutePath", "file_path", "FilePath")
    content = _pick(p, "Content", "CodeContent", "content")
    return {"path": str(path), "content": str(content)} if path and content is not None else None


def _patch(p: dict[str, Any]) -> dict[str, Any] | None:
    path = _pick(p, "Path", "TargetFile", "path", "file_path", "FilePath", "AbsolutePath")
    chunks = _pick(p, "ReplacementChunks", "replacement_chunks")
    if isinstance(chunks, list):
        if len(chunks) != 1 or not isinstance(chunks[0], dict):
            return None
        p = {**p, **chunks[0]}
    old_string = _pick(p, "Target", "TargetContent", "old_string", "OldString", "target")
    new_string = _pick(p, "Replacement", "ReplacementContent", "new_string", "NewString", "replacement")
    if not path or old_string is None or new_string is None:
        return None
    out: dict[str, Any] = {"path": str(path), "old_string": str(old_string), "new_string": str(new_string)}
    if (allow_multi := _pick(p, "AllowMultiple", "replace_all")) is not None:
        out["replace_all"] = bool(allow_multi)
    return out


def _search_content(p: dict[str, Any]) -> dict[str, Any] | None:
    pattern = _pick(p, "Query", "pattern", "SearchPattern")
    if not pattern:
        return None
    out: dict[str, Any] = {"pattern": str(pattern), "target": "content"}
    if path := _pick(p, "SearchPath", "path", "Directory"):
        out["path"] = str(path)
    return out


def _search_files(p: dict[str, Any]) -> dict[str, Any] | None:
    pattern = _pick(p, "Pattern", "pattern", "Query")
    if not pattern:
        return None
    out: dict[str, Any] = {"pattern": str(pattern), "target": "files"}
    if path := _pick(p, "SearchDirectory", "path", "DirectoryPath"):
        out["path"] = str(path)
    return out


def _list_dir(p: dict[str, Any]) -> dict[str, Any] | None:
    path = _pick(p, "DirectoryPath", "Path", "path", "Directory", "SearchDirectory")
    if not path:
        return None
    return {"pattern": "*", "target": "files", "path": str(path)}


def _web_search(p: dict[str, Any]) -> dict[str, Any] | None:
    q = _pick(p, "query", "Query")
    if not q:
        return None
    q_str = str(q)
    domain = _pick(p, "domain", "Domain")
    if domain and f"site:{domain}" not in q_str:
        q_str = f"site:{domain} {q_str}"
    return {"query": q_str}


def _web_extract(p: dict[str, Any]) -> dict[str, Any] | None:
    url = _pick(p, "Url", "url")
    return {"urls": [str(url)]} if url else None


def _clarify(p: dict[str, Any]) -> dict[str, Any] | None:
    raw_questions = _pick(p, "questions", "Questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        return None
    out_questions: list[dict[str, Any]] = []
    for item in raw_questions[:5]:
        if not isinstance(item, dict):
            return None
        q_text = _pick(item, "question", "Question")
        if not q_text:
            return None
        entry: dict[str, Any] = {"question": str(q_text)}
        opts = _pick(item, "options", "choices", "Options", "Choices")
        if isinstance(opts, list) and opts:
            cleaned = []
            for opt in opts[:4]:
                s = str(opt).strip()
                if s.startswith("(Recommended)"):
                    s = s[len("(Recommended)"):].strip()
                if s:
                    cleaned.append(s)
            if cleaned:
                entry["choices"] = cleaned
        multi = _pick(item, "is_multi_select", "multi_select", "IsMultiSelect")
        if multi is not None:
            entry["multi_select"] = bool(multi)
        out_questions.append(entry)
    return {"questions": out_questions} if out_questions else None


def _delegate_task(p: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _pick(p, "Prompt", "prompt", "goal", "Goal")
    if not prompt:
        return None
    task: dict[str, Any] = {"goal": str(prompt)}
    role = _pick(p, "Role", "role")
    context = _pick(p, "Context", "context")
    if role and context:
        task["context"] = f"Role: {role}\n{context}"
    elif role:
        task["context"] = f"Role: {role}"
    elif context:
        task["context"] = str(context)
    return {"tasks": [task]}


def _manage_subagents(p: dict[str, Any]) -> dict[str, Any] | None:
    action = str(_pick(p, "Action", "action") or "").strip().lower()
    if action == "list":
        return {"action": "list"}
    if action in ("kill", "stop"):
        ids = _pick(p, "ConversationIds", "conversation_ids", "subagent_id")
        if isinstance(ids, list) and len(ids) == 1 and ids[0]:
            return {"action": "stop", "subagent_id": str(ids[0])}
        if isinstance(ids, str) and ids:
            return {"action": "stop", "subagent_id": ids}
    return None


def _steer_subagent(p: dict[str, Any]) -> dict[str, Any] | None:
    recipient = _pick(p, "Recipient", "recipient", "subagent_id")
    message = _pick(p, "Message", "message")
    if not recipient or not message:
        return None
    return {"action": "steer", "subagent_id": str(recipient), "message": str(message)}


def _process_manage(p: dict[str, Any]) -> dict[str, Any] | None:
    action = str(_pick(p, "Action", "action") or "").strip().lower()
    task_id = _pick(p, "TaskId", "CommandId", "session_id", "task_id", "command_id")
    if action == "list":
        return {"action": "list"}
    if action == "kill" and task_id:
        return {"action": "kill", "session_id": str(task_id)}
    if action in ("status", "") and task_id:
        if _pick(p, "Terminate", "terminate") is True:
            return {"action": "kill", "session_id": str(task_id)}
        out: dict[str, Any] = {"action": "poll", "session_id": str(task_id)}
        if (wait_s := _pick(p, "WaitDurationSeconds", "DurationSeconds", "timeout")) is not None:
            try:
                if float(wait_s) > 0:
                    out["action"] = "wait"
                    out["timeout"] = int(float(wait_s))
            except (TypeError, ValueError):
                pass
        return out
    if (action == "send_input" or _pick(p, "Input", "input") is not None) and task_id:
        inp = _pick(p, "Input", "input", "data")
        if inp is not None:
            return {"action": "send", "session_id": str(task_id), "data": str(inp)}
    return None


def _cronjob_manage(p: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _pick(p, "Prompt", "prompt")
    if not prompt:
        return None
    cron = _pick(p, "CronExpression", "cron_expression", "schedule")
    duration = _pick(p, "DurationSeconds", "duration_seconds")
    out: dict[str, Any] = {"action": "create", "prompt": str(prompt)}
    if cron:
        out["schedule"] = str(cron)
    elif duration is not None:
        try:
            out["schedule"] = f"in {max(1, int(duration))}s"
        except (TypeError, ValueError):
            return None
    else:
        return None
    return out


# agy tool name -> (Hermes tool name, argument mapper)
NATIVE_TO_HERMES: dict[str, tuple[str, Mapper]] = {
    "run_command": ("terminal", _terminal),
    "view_file": ("read_file", _read_file),
    "read_file": ("read_file", _read_file),
    "write_to_file": ("write_file", _write_file),
    "write_file": ("write_file", _write_file),
    "edit_file": ("patch", _patch),
    "replace_file_content": ("patch", _patch),
    "multi_replace_file_content": ("patch", _patch),
    "grep_search": ("search_files", _search_content),
    "find_by_name": ("search_files", _search_files),
    "list_dir": ("search_files", _list_dir),
    "search_web": ("web_search", _web_search),
    "read_url_content": ("web_extract", _web_extract),
    "ask_question": ("clarify", _clarify),
    "invoke_subagent": ("delegate_task", _delegate_task),
    "manage_subagents": ("delegate_task", _manage_subagents),
    "send_message": ("delegate_task", _steer_subagent),
    "manage_task": ("process_manage", _process_manage),
    "command_status": ("process_manage", _process_manage),
    "send_command_input": ("process_manage", _process_manage),
    "wait": ("process_manage", _process_manage),
    "schedule": ("cronjob_manage", _cronjob_manage),
}


def tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names = set()
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


# agy's stream-json step events carry a display preview of tool parameters: string values longer
# than 512 characters are cut to 512 and end with "…". Executing such a preview would run half a
# command (an unterminated quote at best). The full arguments are in the conversation transcript
# that agy writes under <gemini dir>/brain/<conversation_id>/.system_generated/logs/.
PREVIEW_LIMIT = 512
ELLIPSIS = "\u2026"
TRANSCRIPT_WAIT_SECONDS = 2.0


def _is_truncated_value(val: Any) -> bool:
    if isinstance(val, str):
        return val.endswith(ELLIPSIS) and len(val) >= PREVIEW_LIMIT
    if isinstance(val, dict):
        return any(_is_truncated_value(v) for v in val.values())
    if isinstance(val, list):
        return any(_is_truncated_value(v) for v in val)
    return False


def _matches_preview(preview: Any, full: Any) -> bool:
    if isinstance(preview, str):
        if preview.endswith(ELLIPSIS) and len(preview) >= PREVIEW_LIMIT:
            return isinstance(full, str) and full.startswith(preview[:-1])
        return full == preview
    if isinstance(preview, dict):
        if not isinstance(full, dict):
            return False
        return all(k in full and _matches_preview(v, full[k]) for k, v in preview.items())
    if isinstance(preview, list):
        if not isinstance(full, list) or len(full) != len(preview):
            return False
        return all(_matches_preview(p_item, f_item) for p_item, f_item in zip(preview, full))
    return full == preview


def truncated_keys(params: dict[str, Any]) -> list[str]:
    return [k for k, v in params.items() if _is_truncated_value(v)]


def _transcript_files(gemini_dirs: Iterable[Path], conversation_id: str) -> list[Path]:
    rel = Path("brain") / conversation_id / ".system_generated" / "logs" / "transcript_full.jsonl"
    return [Path(d) / rel for d in gemini_dirs if d]


def restore_truncated(params: dict[str, Any], native: str, conversation_id: str,
                      gemini_dirs: Iterable[Path], wait: float = TRANSCRIPT_WAIT_SECONDS) -> dict[str, Any] | None:
    """Full parameters for a step whose preview was cut, or None when they cannot be proven.

    Matches the newest transcript tool call of the same tool whose every truncated value starts
    with the preview (minus the ellipsis). The transcript can lag the stream by a moment."""
    keys = truncated_keys(params)
    if not keys:
        return params
    if not conversation_id:
        return None
    files = _transcript_files(list(gemini_dirs), conversation_id)
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                for call in reversed(entry.get("tool_calls") or []):
                    if not isinstance(call, dict) or call.get("name") != native:
                        continue
                    args = call.get("args")
                    if not isinstance(args, dict):
                        continue
                    if all(_matches_preview(params[k], args.get(k)) for k in keys):
                        merged = dict(params)
                        for k in keys:
                            merged[k] = args[k]
                        return merged
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


def translate(step: dict[str, Any], available: set[str], index: int,
              resolve: Callable[[dict[str, Any], str], dict[str, Any] | None] | None = None) -> SimpleNamespace | None:
    """Return a Hermes tool-call delta for an agy native tool step, or None if it cannot be mapped."""
    native = str(step.get("tool_name") or (step.get("tool_info") or {}).get("name") or "")
    mapping = NATIVE_TO_HERMES.get(native)
    if not mapping:
        return None
    hermes_name, mapper = mapping
    if hermes_name not in available:
        return None
    params = (step.get("tool_info") or {}).get("parameters")
    if not isinstance(params, dict):
        return None
    if truncated_keys(params):
        # Never hand Hermes a cut command: restore the full text or refuse to translate.
        params = resolve(params, native) if resolve else None
        if params is None:
            return None
    args = mapper(params)
    if args is None:
        return None
    ident = f"agy_{step.get('step_index', index)}_{native}"
    return SimpleNamespace(index=index, id=ident, type="function",
                           function=SimpleNamespace(name=hermes_name, arguments=json.dumps(args, ensure_ascii=False)))
