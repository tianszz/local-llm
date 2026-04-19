import json
import re
import subprocess
import uuid
from pathlib import Path

BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command and return stdout/stderr. Use for file listing, searching, running scripts, checking system state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file"}
                },
                "required": ["path"],
            },
        },
    },
]

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text):
    """Return list of {id, name, arguments} dicts extracted from model output. Empty if none."""
    results = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if not name:
            continue
        results.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": name,
            "arguments": args if isinstance(args, str) else json.dumps(args),
        })
    return results


def strip_tool_calls(text):
    """Remove <tool_call>...</tool_call> blocks from text. Returns None if nothing remains."""
    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    return cleaned or None


def execute_tool(name, arguments):
    """Execute a built-in tool. arguments may be a JSON string or dict. Returns output string."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Error: could not parse arguments: {arguments}"
    else:
        args = arguments

    if name == "shell":
        cmd = args.get("command", "")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            out = result.stdout
            if result.stderr:
                out += ("\nSTDERR:\n" if result.stdout else "") + result.stderr
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: timed out after 30s"
        except Exception as e:
            return f"Error: {e}"

    if name == "read_file":
        path = args.get("path", "")
        try:
            return Path(path).expanduser().read_text()
        except Exception as e:
            return f"Error: {e}"

    return f"Error: unknown tool '{name}'"
