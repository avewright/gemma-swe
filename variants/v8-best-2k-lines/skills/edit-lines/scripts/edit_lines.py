"""Replace lines START..END (1-indexed, inclusive) of a repository file with NEW_TEXT.

END = START - 1 inserts before START; NEW_TEXT "" deletes the range. Python files that stop compiling are left
unchanged and the SyntaxError is reported. Prints the edited region with line numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import clean, resolve, to_int  # noqa: E402

if len(sys.argv) < 4:
    sys.exit('ERROR: usage: args=["path/to/file.py", "START", "END", "new code"]')
quoted = str(sys.argv[1]).strip()[:1] in "\"'`"
full = resolve(sys.argv[1])
start, end = to_int(sys.argv[2], "START"), to_int(sys.argv[3], "END")
new_text = sys.argv[4] if len(sys.argv) > 4 else ""
if quoted:  # the model wrapped every argument in quotes; unwrap the code too
    new_text = clean(new_text)
if "\n" not in new_text and "\\n" in new_text:  # literal \n instead of real line breaks
    new_text = new_text.replace("\\n", "\n")
with open(full, encoding="utf-8") as fh:
    lines = fh.read().split("\n")
if not (1 <= start <= len(lines) + 1) or not (start - 1 <= end <= len(lines)):
    sys.exit(f"ERROR: range {start}-{end} is outside the file (1-{len(lines)}). Nothing changed.")
new_lines = new_text.split("\n") if new_text != "" else []
updated = lines[: start - 1] + new_lines + lines[end:]
content = "\n".join(updated)
if full.endswith(".py"):
    try:
        compile(content, full, "exec")
    except SyntaxError as e:
        sys.exit(f"ERROR: the edit would break Python syntax (line {e.lineno}: {e.msg}). Nothing changed.")
with open(full, "w", encoding="utf-8") as fh:
    fh.write(content)
lo, hi = max(1, start - 2), min(len(updated), start + len(new_lines) + 1)
print(f"OK: lines {start}-{end} replaced by {len(new_lines)} line(s). Now:")
for i in range(lo, hi + 1):
    print(f"{i:>5}| {updated[i - 1]}")
