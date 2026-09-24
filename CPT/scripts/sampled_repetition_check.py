"""
Does the base model's degenerate repetition survive stochastic decoding?

RESULTS.md originally hedged that the loops in the greedy qualitative comparison were
"partly a decoding artifact". This measures that claim instead of asserting it: the same
12 prompts, nucleus sampling, N samples per condition, with a quantitative loop metric.

Metric: max_rep = the highest number of times any whitespace 4-gram repeats in a
continuation. max_rep >= 3 counts as a degenerate loop. Both conditions come from one
loaded model via disable_adapter(), as in qualitative_compare.py.

Usage:
    python scripts/sampled_repetition_check.py \
        --adapter /workspace/adapters/gemma2_2b_cpt_v1/final_adapter \
        --out /workspace/eval/qualitative_sampled.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualitative_compare import PROMPTS  # noqa: E402  (also applies the peft torchao patch)

LOOP_THRESHOLD = 3


def max_ngram_repeat(text, n=4):
    """How many times the most-repeated n-gram occurs. 1 means no repetition."""
    words = text.split()
    if len(words) < n:
        return 1
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(grams.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="/workspace/eval/qualitative_sampled.json")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    torch.manual_seed(args.seed)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    def generate(prompt, adapter_on):
        enc = tok(prompt, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                  top_p=args.top_p, temperature=args.temperature,
                  pad_token_id=tok.eos_token_id)
        with torch.no_grad():
            if adapter_on:
                out = model.generate(**enc, **kw)
            else:
                with model.disable_adapter():
                    out = model.generate(**enc, **kw)
        return tok.decode(out[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    rows, loops = [], {"base": 0, "cpt": 0}
    total = len(PROMPTS) * args.samples
    for i, prompt in enumerate(PROMPTS, 1):
        for s in range(args.samples):
            row = {"prompt_index": i, "prompt": prompt, "sample": s}
            for cond, on in (("base", False), ("cpt", True)):
                text = generate(prompt, on)
                rep = max_ngram_repeat(text)
                row[cond] = text
                row[cond + "_max_rep"] = rep
                if rep >= LOOP_THRESHOLD:
                    loops[cond] += 1
            rows.append(row)
        print(f"[{i}/{len(PROMPTS)}] done", flush=True)

    summary = {
        "decoding": {"do_sample": True, "top_p": args.top_p,
                     "temperature": args.temperature, "seed": args.seed},
        "samples_per_condition": total,
        "loop_threshold_max_4gram_repeat": LOOP_THRESHOLD,
        "base_loop_rate": round(100.0 * loops["base"] / total, 1),
        "cpt_loop_rate": round(100.0 * loops["cpt"] / total, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f,
                  ensure_ascii=False, indent=1)

    print("\n" + "=" * 60)
    print(f"samples per condition : {total}")
    print(f"base loop rate        : {summary['base_loop_rate']}%")
    print(f"CPT  loop rate        : {summary['cpt_loop_rate']}%")
    print("=" * 60)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
