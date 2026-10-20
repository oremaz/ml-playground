"""Persistent events and exports for benchmark playground workspaces."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid


def _serializable(value):
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()
                if str(key).lower() not in {"api_key", "authorization", "password", "access_token", "hf_token"}}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _serializable(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _serializable(vars(value))
    return str(value)


def append_event(workspace, event, *, operation=None, operation_id=None, parent_id=None,
                 run_id=None, data=None, secrets=()):
    if workspace is None:
        return
    record = {"schema_version": 1, "event_id": uuid.uuid4().hex,
              "timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
              "operation": operation, "operation_id": operation_id,
              "parent_id": parent_id, "run_id": run_id, "data": _serializable(data)}
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
    for secret in secrets:
        if secret:
            encoded = encoded.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
    descriptor = os.open(Path(workspace) / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(encoded + "\n")
        stream.flush()
    return record["event_id"]


def read_events(workspace):
    path = Path(workspace) / "events.jsonl"
    if not path.exists():
        return []
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def operation(workspace, name, arguments=None, *, parent_id=None, run_id=None):
    identifier = uuid.uuid4().hex
    started = time.monotonic()
    append_event(workspace, "execution.start", operation=name, operation_id=identifier,
                 parent_id=parent_id, run_id=run_id, data={"arguments": arguments})
    result = {"operation_id": identifier}
    try:
        yield result
    except BaseException as exc:
        append_event(workspace, "execution.error", operation=name, operation_id=identifier,
                     parent_id=parent_id, run_id=run_id,
                     data={"elapsed_seconds": time.monotonic() - started,
                           "error_type": type(exc).__name__, "error": str(exc)})
        raise
    else:
        append_event(workspace, "execution.error" if result.get("status") == "failed" else "execution.end",
                     operation=name, operation_id=identifier,
                     parent_id=parent_id, run_id=run_id,
                     data={"elapsed_seconds": time.monotonic() - started, "result": result})


def export_history(workspace):
    workspace = Path(workspace)
    artifacts = {}
    patterns = ("conversation.json", "context.json", "profiles.json", "tables.json", "plan.json",
                "interpretation.json", "validation_report.json", "last_result.json", "active_agent.json", "revision_*.py",
                "revision_*.json", "run_*.py", "run_*.json", "run_*.log",
                "agents/*/messages.json", "agents/*/state.json", "agents/*/evaluation.json",
                "agents/*/preparation.py", "agents/*/preparation_config.json", "agents/*/preparation.log",
                "agents/*/trial_*/experiment.py", "agents/*/trial_*/config.json",
                "agents/*/trial_*/features.json", "agents/*/trial_*/execution.log")
    for pattern in patterns:
        for path in sorted(workspace.glob(pattern)):
            content = path.read_text()
            artifacts[str(path.relative_to(workspace))] = {"content": json.loads(content) if path.suffix == ".json" else content,
                                    "artifact_modified_at": datetime.fromtimestamp(
                                        path.stat().st_mtime, timezone.utc).isoformat()}
    return {"schema_version": 1, "workspace": workspace.name,
            "note": "Events record observed requests and execution from the time tracing was enabled. "
                    "Earlier artifacts are original saved outputs; missing prompts, provider metadata, "
                    "durations and native tool calls cannot be recovered. Artifact modification times "
                    "are not request timestamps. Application execution events are not native LLM tool calls.",
            "events": read_events(workspace) + [event for folder in sorted((workspace / "agents").glob("*"))
                                                if folder.is_dir() for event in read_events(folder)],
            "artifacts": artifacts}


def history_markdown(history):
    parts = [f"# Playground conversation and execution log: {history['workspace']}", history["note"]]
    conversation = history["artifacts"].get("conversation.json", {}).get("content", [])
    for index, message in enumerate(conversation, 1):
        content = message.get("content", "")
        parts.append(f"## Message {index}: {message.get('role', 'unknown')}")
        parts.append(content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False))
    for name, artifact in history["artifacts"].items():
        if name == "conversation.json":
            continue
        parts.append(f"## Saved artifact: {name}")
        parts.append(f"Artifact modification time: {artifact['artifact_modified_at']}")
        content = artifact["content"]
        content = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)
        fence = "````" if "```" in content else "```"
        parts.append(f"{fence}\n{content}\n{fence}")
    parts.append("## Recorded events")
    parts.append("```json\n" + json.dumps(history["events"], indent=2, ensure_ascii=False) + "\n```")
    return "\n\n".join(parts) + "\n"
