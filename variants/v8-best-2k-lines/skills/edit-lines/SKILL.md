---
name: edit-lines
description: The only way to edit files. Call run_skill_script with skill_name "edit-lines" to show numbered lines or replace a line range.
---

# Editing by line number

There is no `edit_file` tool. Always call the tool `run_skill_script`:

- Numbered lines 120-140 of a file:
  `run_skill_script(skill_name="edit-lines", file_path="scripts/show_lines.py", args=["rich/cells.py", "120", "140"])`
- Replace lines 124-127 (inclusive) with new code:
  `run_skill_script(skill_name="edit-lines", file_path="scripts/edit_lines.py", args=["rich/cells.py", "124", "127", "<new code>"])`
  Write the new code with real line breaks and full indentation. Insert before line 124 without removing: use
  "124", "123". Delete lines: pass "" as the new code.

Edits that would break Python syntax are rejected and the file is left unchanged; the result shows the edited region
with line numbers. Line numbers shift after an edit: show the lines again before the next edit.
