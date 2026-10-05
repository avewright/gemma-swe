# gemma-swe

Work on the Kaggle `gemma-4-developer-agent` contest: an agent bundle for
gemma-4-31b-it that fixes real GitHub issues, graded by hidden tests.

| Path | What it holds |
| --- | --- |
| `submission/` | The current candidate bundle (agent, prompts, sampling, eval config) |
| `variants/` | One folder per experiment config (`v8-best-2k`, `sweep-t10`, ...) |
| `starter/` | The organizers' sample submission |
| `scripts/` | Local harness runner, benchmarks, analysis, dataset tools |
| `distill/` | Teacher-trace collection, packing and LoRA training |
| `experiments/` | Written results and diagnostics per pod run (`REPORT.md` first) |
| `archive/` | Old prompts kept for reference |
| `dataset/` | HF dataset card and parquet |
| `data/` | Contest task list and harness README; large files are git-ignored |

Git-ignored, local only: `runs/`, `traces/`, `dist/` (built zips), `data/snapshots`, `data/embeddings`.

Build a submission: copy a `variants/<name>` folder to a temp dir, zip it as `submission.zip`,
then `kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "..."`.
