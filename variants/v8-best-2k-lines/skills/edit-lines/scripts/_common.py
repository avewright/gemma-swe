"""Shared argument handling for the edit-lines scripts.

The model often wraps arguments in extra quotes or backticks (e.g. '"734"', '`rich/cells.py`'). The scripts run from
a temp directory, so relative paths are resolved against the sandbox workspace: $PWD (the shell's original cwd,
unchanged by the skill runner's os.chdir), else /workspace.
"""
import os


def clean(arg) -> str:
    s = str(arg).strip()
    for _ in range(2):
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
            s = s[1:-1]
    return s


def to_int(arg, name: str) -> int:
    try:
        return int(clean(arg))
    except ValueError:
        raise SystemExit(f"ERROR: {name} must be a line number, got {arg!r}.")


def resolve(path: str) -> str:
    path = clean(path)
    if path.startswith("/workspace/"):
        path = path[len("/workspace/"):]
    if os.path.isabs(path):
        return path
    for root in (os.environ.get("PWD"), "/workspace", os.getcwd()):
        if root and os.path.exists(os.path.join(root, path)):
            return os.path.join(root, path)
    raise SystemExit(f"ERROR: file not found: {path} (paths are relative to the repository root).")
