#!/usr/bin/env python3
"""
Teacher-forced next-token evaluation for Gemma, base vs CPT adapter.

Reproduces the metric set used in the project's evaluation.md:
  top-1 / top-5 / top-10 accuracy, average probability of the correct
  token, average NLL, perplexity, average rank of the correct token.

TWO STEPS. Build the evaluation set ONCE, then score every model against
that same file. If you rebuild between models the sampled positions change
and the comparison is meaningless.

  # 1. build once (CPU, no model weights needed beyond the tokenizer)
  python next_token_eval_gemma.py build \
      --input  splits/test_doclevel.jsonl \
      --out    eval_set_1000.json \
      --model_name google/gemma-2-2b \
      --n 1000 --seed 42

  # 2. score the BASE model
  python next_token_eval_gemma.py score \
      --eval_set eval_set_1000.json \
      --model_name google/gemma-2-2b \
      --out results_base.json

  # 3. score the CPT model (same eval set)
  python next_token_eval_gemma.py score \
      --eval_set eval_set_1000.json \
      --model_name google/gemma-2-2b \
      --adapter adapters/gemma2_2b_cpt_v1/final_adapter \
      --out results_cpt.json

  # 4. compare
  python next_token_eval_gemma.py compare \
      --results results_base.json results_cpt.json
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def cmd_build(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    rng = random.Random(args.seed)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            if not text:
                continue
            records.append(
                {
                    "source": r.get("final_cpt_source") or r.get("source") or "unknown",
                    "source_id": r.get("source_id") or r.get("id"),
                    "text": text,
                }
            )

    print(f"records available : {len(records):,}")
    rng.shuffle(records)

    examples = []
    skipped_short = 0

    for r in records:
        if len(examples) >= args.n:
            break

        ids = tok(r["text"], add_special_tokens=False)["input_ids"]

        # Need enough left context for the prediction to be meaningful.
        if len(ids) < args.min_context + 1:
            skipped_short += 1
            continue

        # Target position: uniformly inside the usable range, so predictions
        # are not all taken from document openings.
        lo = args.min_context
        hi = min(len(ids) - 1, args.max_context)
        if hi <= lo:
            skipped_short += 1
            continue
        pos = rng.randint(lo, hi)

        context = ids[max(0, pos - args.max_context) : pos]
        gold = ids[pos]

        examples.append(
            {
                "source": r["source"],
                "source_id": r["source_id"],
                "context_ids": context,
                "gold_id": gold,
                "gold_token": tok.decode([gold]),
            }
        )

    by_source = defaultdict(int)
    for e in examples:
        by_source[e["source"]] += 1

    meta = {
        "model_name_for_tokenizer": args.model_name,
        "n_examples": len(examples),
        "seed": args.seed,
        "min_context": args.min_context,
        "max_context": args.max_context,
        "input_file": str(args.input),
        "by_source": dict(sorted(by_source.items())),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "examples": examples}, f, ensure_ascii=False)

    print(f"skipped (too short): {skipped_short:,}")
    print(f"built              : {len(examples):,} examples -> {args.out}\n")
    print(f"{'source':<28}{'examples':>10}")
    print("-" * 38)
    for s, n in sorted(by_source.items()):
        print(f"{s:<28}{n:>10}")

    if len(examples) < args.n:
        print(
            f"\nWARNING: wanted {args.n}, built {len(examples)}. "
            "Lower --min_context or supply more text."
        )


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def cmd_score(args):
    import torch

    blob = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    examples = blob["examples"]
    meta = blob["meta"]

    if meta["model_name_for_tokenizer"] != args.model_name:
        raise ValueError(
            f"Eval set was tokenized with {meta['model_name_for_tokenizer']!r} "
            f"but you passed {args.model_name!r}. Token ids would not match."
        )

    print(f"eval set : {args.eval_set}  ({len(examples)} examples)")
    print(f"model    : {args.model_name}")
    print(f"adapter  : {args.adapter or '(none - base model)'}\n")

    # ---- load ------------------------------------------------------------
    if args.use_unsloth:
        from unsloth import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_context + 8,
            dtype=None,
            load_in_4bit=True,
        )
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=quant, device_map="auto"
        )
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    device = next(model.parameters()).device

    # ---- score -----------------------------------------------------------
    per_example = []

    with torch.no_grad():
        for i, ex in enumerate(examples):
            ids = torch.tensor([ex["context_ids"]], dtype=torch.long, device=device)
            logits = model(input_ids=ids).logits[0, -1].float()

            logprobs = torch.log_softmax(logits, dim=-1)
            gold = ex["gold_id"]

            gold_logprob = logprobs[gold].item()
            gold_prob = math.exp(gold_logprob)

            # rank: 1 = model's top choice
            rank = int((logits > logits[gold]).sum().item()) + 1
            top10 = torch.topk(logits, k=10).indices.tolist()

            per_example.append(
                {
                    "source": ex["source"],
                    "rank": rank,
                    "prob": gold_prob,
                    "nll": -gold_logprob,
                    "top1": rank == 1,
                    "top5": rank <= 5,
                    "top10": rank <= 10,
                    "pred_id": top10[0],
                }
            )

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(examples)}")

    results = {
        "model_name": args.model_name,
        "adapter": args.adapter,
        "eval_set": str(args.eval_set),
        "n_examples": len(per_example),
        "overall": summarize(per_example),
        "by_source": {
            s: summarize([e for e in per_example if e["source"] == s])
            for s in sorted({e["source"] for e in per_example})
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_results(results)
    print(f"\nwrote {args.out}")


def summarize(rows):
    """Metric set matching evaluation.md."""
    n = len(rows)
    if n == 0:
        return {}
    mean_nll = sum(r["nll"] for r in rows) / n
    return {
        "n": n,
        "top1": sum(r["top1"] for r in rows) / n,
        "top5": sum(r["top5"] for r in rows) / n,
        "top10": sum(r["top10"] for r in rows) / n,
        "avg_target_probability": sum(r["prob"] for r in rows) / n,
        "avg_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "avg_target_rank": sum(r["rank"] for r in rows) / n,
    }


def print_results(results):
    o = results["overall"]
    print("\n" + "=" * 78)
    print(f"{'':<28}{'Top-1':>8}{'Top-5':>8}{'Top-10':>8}{'AvgP':>8}{'NLL':>8}{'PPL':>10}{'Rank':>8}")
    print("=" * 78)
    row = lambda name, m: print(
        f"{name:<28}{m['top1']*100:>7.1f}%{m['top5']*100:>7.1f}%{m['top10']*100:>7.1f}%"
        f"{m['avg_target_probability']:>8.3f}{m['avg_nll']:>8.3f}"
        f"{m['perplexity']:>10.2f}{m['avg_target_rank']:>8.2f}"
    )
    row("OVERALL", o)
    print("-" * 78)
    for s, m in results["by_source"].items():
        row(f"  {s} (n={m['n']})", m)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def cmd_compare(args):
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.results]

    names = []
    for r in loaded:
        names.append("CPT" if r.get("adapter") else "Base")

    print(f"\n{'Model':<12}{'Top-1':>9}{'Top-5':>9}{'Top-10':>9}{'AvgP':>8}{'NLL':>8}{'PPL':>10}{'Rank':>8}")
    print("-" * 73)
    for name, r in zip(names, loaded):
        m = r["overall"]
        print(
            f"{name:<12}{m['top1']*100:>8.1f}%{m['top5']*100:>8.1f}%{m['top10']*100:>8.1f}%"
            f"{m['avg_target_probability']:>8.3f}{m['avg_nll']:>8.3f}"
            f"{m['perplexity']:>10.2f}{m['avg_target_rank']:>8.2f}"
        )

    if len(loaded) == 2:
        a, b = loaded[0]["overall"], loaded[1]["overall"]
        print("-" * 73)
        print(
            f"{'DELTA':<12}{(b['top1']-a['top1'])*100:>+8.1f}pp"
            f"{(b['top5']-a['top5'])*100:>+8.1f}pp{(b['top10']-a['top10'])*100:>+8.1f}pp"
            f"{b['avg_target_probability']-a['avg_target_probability']:>+8.3f}"
            f"{b['avg_nll']-a['avg_nll']:>+8.3f}"
            f"{b['perplexity']-a['perplexity']:>+10.2f}"
            f"{b['avg_target_rank']-a['avg_target_rank']:>+8.2f}"
        )

        print("\nPer-source top-1:")
        print(f"{'source':<28}{'Base':>9}{'CPT':>9}{'delta':>10}")
        print("-" * 56)
        for s in loaded[0]["by_source"]:
            if s not in loaded[1]["by_source"]:
                continue
            x = loaded[0]["by_source"][s]["top1"]
            y = loaded[1]["by_source"][s]["top1"]
            print(f"{s:<28}{x*100:>8.1f}%{y*100:>8.1f}%{(y-x)*100:>+9.1f}pp")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the evaluation set once")
    b.add_argument("--input", required=True, help="held-out clean JSONL")
    b.add_argument("--out", required=True)
    b.add_argument("--model_name", default="google/gemma-2-2b")
    b.add_argument("--n", type=int, default=1000)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--min_context", type=int, default=64)
    b.add_argument("--max_context", type=int, default=512)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score one model against a built eval set")
    s.add_argument("--eval_set", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model_name", default="google/gemma-2-2b")
    s.add_argument("--adapter", default=None, help="omit to score the base model")
    s.add_argument("--max_context", type=int, default=512)
    s.add_argument("--use_unsloth", action="store_true")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="print a base-vs-CPT table")
    c.add_argument("--results", nargs="+", required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
