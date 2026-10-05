#!/usr/bin/env python3
"""Fine-tune a LoRA adapter on resolved teacher traces (rejection sampling, step 3).

Keeps rollouts that passed the hidden tests, renders them with the student's own
chat template (tools included), and trains only on the assistant turns that made
a tool call. Prompts, tool outputs, nudges and stalled turns are masked out.

Check rendering first; this needs only the tokenizer, no GPU:
  python distill/train_lora.py --dry-run
Then train (QLoRA by default; one process, the model is spread over all visible GPUs):
  python distill/train_lora.py --name lora-v1
The adapter lands in runs/lora/<name>/adapter. export_adapter.py puts it in a variant.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

import quality

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GEMMA_ROOT", Path(__file__).resolve().parents[1]))
HOLDOUT = HERE.parent / "scripts" / "dev40.txt"
TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
NON_TEXT = ("vision", "audio", "multi_modal", "mm_", "embed_")


# ---------- data ----------

def select(args) -> list[dict]:
    rows = pq.read_table(args.data).to_pylist()
    holdout = set(HOLDOUT.read_text().split())
    leaked = {r["instance_id"] for r in rows if r["resolved"]} & holdout
    if leaked and not args.allow_holdout:
        raise SystemExit(f"{len(leaked)} dev40 holdout tasks have traces ({sorted(leaked)[:3]}...); "
                         "they would contaminate the bench. Rerun collect.py without them or pass --allow-holdout.")
    pinned = set(json.loads(args.ids.read_text())) if args.ids else None
    reasons: Counter = Counter()
    keep: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        found = quality.issues(r, args.max_outside_refs, require_submit=not args.allow_unsubmitted)
        if args.teachers and r["teacher"] not in args.teachers:
            found.append("other teacher")
        if pinned is not None and r["id"] not in pinned:
            found.append("not in --ids")
        if found:
            reasons[found[0]] += 1
        else:
            keep[r["instance_id"]].append(r)
    picked = []
    for group in keep.values():
        # Best first (verified its fix, then fewest calls: the student's budget is tight),
        # and one per teacher before a second from the same teacher, for variety.
        group.sort(key=quality.rank)
        seen, ordered = set(), []
        for r in group:
            ordered.insert(len(seen) if r["teacher"] not in seen else len(ordered), r)
            seen.add(r["teacher"])
        picked += ordered[: args.max_per_task]
        reasons["over --max-per-task"] += len(ordered[args.max_per_task:])
    print(f"traces: {len(rows)} total, {len(picked)} selected from {len(keep)} tasks; dropped {dict(reasons)}")
    return picked


def first_sentences(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > 0 else cut


def thought_for(m: dict, args) -> str:
    """The short thought the student should produce before this tool call (see quality.clean_thought)."""
    content = (m.get("content") or "").strip()
    reasoning = first_sentences(m.get("reasoning") or "", args.max_thought_chars)
    if args.thought == "none":
        return ""
    thought = (reasoning or content) if args.thought == "reasoning" else (content or reasoning)
    return quality.clean_thought(thought)


def to_chat(row: dict, args, system: str | None) -> list[dict]:
    """OpenAI-style trace -> HF chat messages.

    Tool arguments become dicts, which chat templates expect. The teacher's short thought
    goes in `reasoning`, which Gemma 4 renders as the thinking channel before the tool call,
    where the student produces it at inference (thinking is on in sampling.yaml). Plain
    assistant text would be rendered after the tool response instead.
    """
    out, pending = [], {}
    for m in json.loads(row["messages"]):
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": system or m["content"]})
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "tool":
            tool = {"role": "tool", "tool_call_id": m.get("tool_call_id"), "name": m.get("name"), "content": m["content"]}
            # Put each result right after its own call (after the split above).
            out.insert(pending.pop(m.get("tool_call_id"), len(out)), tool)
            for k in pending:
                pending[k] += 1
        elif role == "assistant" and m.get("tool_calls"):
            # The student makes one call per step, so parallel calls become consecutive steps,
            # the thought on the first. Their results were independent, so the order is valid.
            for n, c in enumerate(m["tool_calls"]):
                msg = {"role": "assistant", "content": "",
                       "tool_calls": [{"id": c["id"], "type": "function",
                                       "function": {"name": c["function"]["name"],
                                                    "arguments": json.loads(c["function"]["arguments"] or "{}")}}]}
                thought = thought_for(m, args) if n == 0 else ""
                if thought:
                    msg["reasoning"] = thought
                out.append(msg)
                pending[c["id"]] = len(out)
        elif role == "assistant":
            # A stalled turn the harness nudged. Kept as context, never trained.
            out.append({"role": "assistant", "content": (m.get("content") or "").strip()})
    # Nothing after the last tool call is trained, so don't spend tokens on it.
    last = max(i for i, m in enumerate(out) if m.get("tool_calls"))
    return out[: last + 1]


def render(tok, messages: list[dict], tools: list[dict], args, generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=generation_prompt,
                                   enable_thinking=args.thinking, preserve_thinking=True)


GEMMA4_TURN, GEMMA4_END = "<|turn>model\n", "<turn|>"
GEMMA4_RESPONSE, GEMMA4_RESPONSE_END = "<|tool_response>", "<tool_response|>"


def gemma4_spans(text: str) -> list[tuple[int, int]]:
    """Trained spans for Gemma 4, whose model turn holds several tool calls and their responses:
    <|turn>model\n [<|channel>thought ...<channel|>] <|tool_call>...<tool_call|><|tool_response>response...<tool_response|> [next call] ...
    Each span runs from the start of a step to its <|tool_response> token, the token the model
    stops on at inference. Response bodies, and steps that end without a tool call, are masked."""
    spans, pos = [], 0
    while (start := text.find(GEMMA4_TURN, pos)) >= 0:
        step = start + len(GEMMA4_TURN)
        end = text.find(GEMMA4_END, step)
        end = len(text) if end < 0 else end
        while (call_end := text.find(GEMMA4_RESPONSE, step, end)) >= 0:
            spans.append((step, call_end + len(GEMMA4_RESPONSE)))
            resp_end = text.find(GEMMA4_RESPONSE_END, call_end, end)
            if resp_end < 0:
                break
            step = resp_end + len(GEMMA4_RESPONSE_END)
        pos = end
    return spans


def prefix_spans(tok, messages: list[dict], tools: list[dict], args, full: str) -> list[tuple[int, int]] | None:
    """Other templates: each span is the text between "history + generation prompt" and
    "history + this turn". Needs renders that extend each other; None if they don't."""
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        before = render(tok, messages[:i], tools, args, generation_prompt=True)
        through = render(tok, messages[: i + 1], tools, args)
        if not (through.startswith(before) and full.startswith(through)):
            return None
        spans.append((len(before), len(through)))
    return spans


def encode(tok, messages: list[dict], tools: list[dict], args) -> tuple[dict | None, str]:
    """Token ids plus labels that are -100 everywhere except the student's own generations."""
    full = render(tok, messages, tools, args)
    spans = gemma4_spans(full) if GEMMA4_RESPONSE in (tok.chat_template or "") else prefix_spans(tok, messages, tools, args, full)
    if spans is None:
        return None, "template renders are not prefixes of each other"
    if not spans:
        return None, "no assistant tool calls"
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > args.max_len:
        return None, "longer than --max-len"
    labels = [tid if any(s <= start < e for s, e in spans) else -100
              for tid, (start, _) in zip(ids, enc["offset_mapping"])]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels, "text": full, "spans": spans}, "ok"


ELIDED = json.dumps({"status": "ok", "note": "older output elided to fit the context window"})
KEEP_RECENT = 5  # like the harness's compaction, which keeps the last 5 events verbatim


def encode_fitting(tok, messages: list[dict], tools: list[dict], args) -> tuple[dict | None, str]:
    """encode(), eliding the oldest tool outputs until the example fits --max-len.

    The contest harness compacts long histories (HARNESS_README 7.2), so the student never
    sees a full 40-call trajectory either. Every thought and tool call is still trained."""
    ex, why = encode(tok, messages, tools, args)
    outputs = [i for i, m in enumerate(messages) if m["role"] == "tool"][:-KEEP_RECENT]
    elided = 0
    while ex is None and why == "longer than --max-len" and outputs:
        for i in outputs[:3]:
            messages[i] = dict(messages[i], content=ELIDED)
        elided += len(outputs[:3])
        outputs = outputs[3:]
        ex, why = encode(tok, messages, tools, args)
    if ex and elided:
        why = "ok, older outputs elided"
    return ex, why


def build(tok, rows: list[dict], args) -> list[dict]:
    system = Path(args.system_prompt).read_text() if args.system_prompt else None
    examples, reasons, lengths = [], Counter(), []
    for row in rows:
        messages = to_chat(row, args, system)
        ex, why = encode_fitting(tok, messages, json.loads(row["tools"]), args)
        if ex:
            ex["messages"] = messages
        reasons[why] += 1
        if ex:
            ex["id"], ex["row"], ex["tools"] = row["id"], row, json.loads(row["tools"])
            examples.append(ex)
            lengths.append(len(ex["input_ids"]))
    trained = sum(sum(l != -100 for l in ex["labels"]) for ex in examples)
    print(f"examples: {len(examples)} kept; {dict(reasons)}")
    if lengths:
        lengths.sort()
        print(f"tokens per example: median {lengths[len(lengths) // 2]}, max {lengths[-1]}; "
              f"trained tokens {trained} of {sum(lengths)} ({trained / sum(lengths):.1%})")
    return examples


def show(tok, ex: dict, path: Path) -> None:
    """Write one rendered example with trained spans wrapped in ⟦ ⟧, to eyeball the format."""
    text, marked, pos = ex["text"], [], 0
    for s, e in ex["spans"]:
        marked += [text[pos:s], "⟦", text[s:e], "⟧"]
        pos = e
    marked.append(text[pos:])
    path.write_text("".join(marked))
    print(f"rendered example {ex['id']} -> {path} (trained spans are inside ⟦ ⟧)")


def export(examples: list[dict], path: Path) -> None:
    """The final training set as chat JSONL (HF messages + tools), for any SFT trainer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ex in examples:
            r = ex["row"]
            fh.write(json.dumps({
                "id": ex["id"], "instance_id": r["instance_id"], "repo": r["repo"], "teacher": r["teacher_model"],
                "tool_calls": r["tool_calls"], "n_tokens": len(ex["input_ids"]),
                "n_trained_tokens": sum(l != -100 for l in ex["labels"]),
                "messages": ex["messages"], "tools": ex["tools"],
            }, ensure_ascii=False) + "\n")
    print(f"exported {len(examples)} examples to {path}")


# ---------- training ----------

def lora_targets(model, suffixes: list[str]) -> list[str]:
    import torch
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in suffixes
             and not any(tag in n.lower() for tag in NON_TEXT)]
    if not names:
        raise SystemExit(f"no Linear modules named {suffixes}; check --targets against model.named_modules()")
    return names


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto", "attn_implementation": args.attn}
    if args.qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        return AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    except ValueError:
        # Multimodal checkpoints (Gemma 3/4) may only load through the image-text class.
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(args.base, **kwargs)


def train(tok, examples: list[dict], args, out: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model = load_model(args)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = lora_targets(model, args.targets.split(","))
    print(f"LoRA on {len(targets)} modules, e.g. {targets[0]}")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    class Examples(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            ex = examples[i]
            return {k: ex[k] for k in ("input_ids", "attention_mask", "labels")}

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    optimizers = (None, None)
    if args.optim == "polar-normuon":
        from normuon import PolarNorMuon
        # The Trainer still builds the cosine schedule with warmup around this optimizer.
        optimizers = (PolarNorMuon([p for p in model.parameters() if p.requires_grad], lr=args.lr), None)
    trainer = Trainer(
        optimizers=optimizers,
        model=model,
        train_dataset=Examples(),
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            # transformers 5 dropped warmup_ratio: 5% of the optimizer steps, at least 1.
            warmup_steps=max(1, round(0.05 * args.epochs * len(examples) / args.grad_accum)),
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        ),
    )
    trainer.train()
    model.save_pretrained(out / "adapter", safe_serialization=True)
    print(f"adapter saved to {out / 'adapter'}. Next: python distill/export_adapter.py --adapter {out / 'adapter'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "traces" / "packed" / "traces.parquet")
    parser.add_argument("--name", default="lora-v1", help="output folder under runs/lora/")
    parser.add_argument("--base", default="google/gemma-4-31b-it",
                        help="bf16 weights of the served model (Kaggle serves the int4 QAT build of it)")
    parser.add_argument("--tokenizer", help="defaults to --base")
    parser.add_argument("--system-prompt", help="replace the traces' system prompt, e.g. submission/prompts/system.md")
    parser.add_argument("--teachers", nargs="*", help="only traces from these teachers")
    parser.add_argument("--max-per-task", type=int, default=2, help="cap per task so easy tasks don't dominate")
    parser.add_argument("--ids", type=Path, help="JSON list of trace ids: use only these (e.g. a frozen snapshot)")
    parser.add_argument("--max-outside-refs", type=int, default=2,
                        help="drop traces with more commands than this that look outside /workspace (wasted calls)")
    parser.add_argument("--allow-unsubmitted", action="store_true", help="keep resolved traces that never called submit_patch")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--thought", choices=["auto", "reasoning", "none"], default="auto",
                        help="thought before each tool call: auto = the teacher's visible text, else its reasoning "
                             "clipped to --max-thought-chars; reasoning = clipped reasoning first; none = no thought")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false",
                        help="render with thinking off (only if the submission turns thinking off)")
    parser.add_argument("--max-thought-chars", type=int, default=400)
    parser.add_argument("--max-len", type=int, default=16384,
                        help="longer traces get their oldest tool outputs elided to fit. The contest context is "
                             "32768, but Gemma's 262k vocabulary makes logits for long sequences costly in memory")
    parser.add_argument("--rank", type=int, default=16, help="the harness allows up to 128")
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", default=TARGETS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--optim", choices=["adamw", "polar-normuon"], default="adamw",
                        help="polar-normuon: NorMuon with Polar Express orthogonalization (distill/normuon.py)")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="train on bf16 weights (needs ~70 GB+)")
    parser.add_argument("--attn", default="sdpa", help="sdpa or flash_attention_2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="select, render and mask only; no model")
    parser.add_argument("--export", type=Path, help="also write the final examples as chat JSONL")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.base)
    if not tok.chat_template:
        raise SystemExit("tokenizer has no chat template; pass --tokenizer pointing at one that does")
    examples = build(tok, select(args), args)
    if not examples:
        raise SystemExit("no usable examples")
    out = ROOT / "runs" / "lora" / args.name
    out.mkdir(parents=True, exist_ok=True)
    show(tok, examples[0], out / "example.txt")
    (out / "train_meta.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "examples": [ex["id"] for ex in examples]}, indent=1))
    if args.export:
        export(examples, args.export)
    if args.dry_run:
        return
    train(tok, examples, args, out)


if __name__ == "__main__":
    main()
