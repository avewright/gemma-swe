"""Print lines START..END (1-indexed, inclusive) of a repository file with line numbers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import resolve, to_int  # noqa: E402

if len(sys.argv) < 4:
    sys.exit('ERROR: usage: args=["path/to/file.py", "START", "END"]')
full = resolve(sys.argv[1])
start, end = to_int(sys.argv[2], "START"), to_int(sys.argv[3], "END")
with open(full, encoding="utf-8") as fh:
    lines = fh.read().split("\n")
start, end = max(1, start), min(len(lines), end)
for i in range(start, end + 1):
    print(f"{i:>5}| {lines[i - 1]}")
print(f"({len(lines)} lines in file)")
