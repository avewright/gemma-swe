---
dataset_info:
  features:
    - name: instance_id
      dtype: string
    - name: repo
      dtype: string
    - name: base_commit
      dtype: string
    - name: patch
      dtype: string
    - name: test_patch
      dtype: string
    - name: problem_statement
      dtype: string
    - name: hints_text
      dtype: string
    - name: created_at
      dtype: string
  splits:
    - name: train
      num_examples: 129
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
---

# Local practice tasks

129 public contest tasks from `data/tasks.jsonl`. Stream them from this folder:

```python
from datasets import load_dataset

ds = load_dataset("dataset", split="train", streaming=True)
```

This folder stays local. The contest rules do not allow publishing the competition data.
