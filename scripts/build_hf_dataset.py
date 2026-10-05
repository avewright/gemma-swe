"""Build a local streamable dataset from the contest practice tasks.

Source: data/tasks.jsonl. Output: dataset/, loadable with
load_dataset("dataset", split="train", streaming=True).

Do not upload this folder to the public Hugging Face Hub. The contest
rules forbid sharing the competition data with anyone who has not joined.
"""

import json
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "tasks.jsonl"
OUTPUT = ROOT / "dataset"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    dataset = Dataset.from_list(rows)
    (OUTPUT / "data").mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT / "data" / "train-00000-of-00001.parquet")
    print(f"wrote {len(dataset)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
