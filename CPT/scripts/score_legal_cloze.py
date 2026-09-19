"""
Score a model on the legal cloze probe built by build_legal_cloze.py.

Greedy continuation: the model is given the context up to the blank and must produce the
masked term. This matches how a base LM works -- no instruction format is involved, so the
probe is fair to the base model, which is the comparison point.

Every category reports a MAJORITY-CLASS BASELINE alongside accuracy. Without it the
numbers are uninterpretable: `statute_term` has only four possible answers and المرسوم
accounts for ~40% of them, so 40% is what always guessing the most common answer would
score. Only the margin above that baseline is evidence of domain knowledge.

Usage:
    # base model
    python score_legal_cloze.py score --cloze eval/legal_cloze.json \
        --model_name google/gemma-2-2b --out eval/cloze_base.json

    # CPT model, same probe file
    python score_legal_cloze.py score --cloze eval/legal_cloze.json \
        --model_name google/gemma-2-2b --adapter <...>/final_adapter \
        --out eval/cloze_cpt.json

    python score_legal_cloze.py compare --results eval/cloze_base.json eval/cloze_cpt.json
"""

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

# peft on the Colab stack hard-errors on the old torchao preinstall; we never use torchao.
try:
    import peft.import_utils as _pi, peft.tuners.lora.torchao as _pt
    _pi.is_torchao_available = _pt.is_torchao_available = lambda: False
except Exception:
    pass

ARABIC_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
ARABIC_WORD = re.compile(r"[ء-ي]+")


def normalize(value):
    """Same normalisation as build_legal_cloze.py and the project's reference script."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("ـ", "")
    value = ARABIC_DIACRITICS.sub("", value)
    # unify alef and yaa variants -- orthographic noise, not a wrong answer
    value = re.sub("[آأإ]", "ا", value)
    value = value.replace("ى", "ي").replace("ة", "ه")
    return value.strip()


def first_word(text):
    m = ARABIC_WORD.search(text or "")
    return m.group(0) if m else ""


def load_model(model_name, adapter, use_unsloth=False):
    import torch
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported() else torch.float16,
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map={"": 0},
            attn_implementation="sdpa",
        )

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"adapter attached: {adapter}")
    else:
        print("no adapter - scoring the BASE model")

    model.eval()
    tok.padding_side = "left"          # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def score(args):
    import torch
    payload = json.load(open(args.cloze, encoding="utf-8"))
    items = payload["items"]
    print(f"cloze items: {len(items)}   model: {args.model_name}")

    model, tok = load_model(args.model_name, args.adapter, args.use_unsloth)

    preds = []
    B = args.batch_size
    for start in range(0, len(items), B):
        batch = items[start:start + B]
        enc = tok(
            [b["context"] for b in batch],
            return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_context, add_special_tokens=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        for b, g in zip(batch, gen):
            text = tok.decode(g, skip_special_tokens=True)
            preds.append({
                "id": b["id"],
                "category": b["category"],
                "source": b["source"],
                "answer": b["answer"],
                "generated": text[:60],
                "predicted": first_word(text),
                "correct": normalize(first_word(text)) == normalize(b["answer"]),
            })
        print(f"  {min(start + B, len(items))}/{len(items)}", end="\r")
    print()

    def acc(rows):
        return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0

    def majority(rows):
        """Always-guess-the-most-common-answer baseline."""
        if not rows:
            return 0.0
        c = collections.Counter(normalize(r["answer"]) for r in rows)
        return c.most_common(1)[0][1] / len(rows)

    by_cat = collections.defaultdict(list)
    by_src = collections.defaultdict(list)
    for r in preds:
        by_cat[r["category"]].append(r)
        by_src[r["source"]].append(r)

    results = {
        "model_name": args.model_name,
        "adapter": args.adapter,
        "cloze_file": str(args.cloze),
        "n": len(preds),
        "accuracy": round(acc(preds), 4),
        "majority_baseline": round(majority(preds), 4),
        "by_category": {
            k: {"n": len(v), "accuracy": round(acc(v), 4),
                "majority_baseline": round(majority(v), 4)}
            for k, v in sorted(by_cat.items())
        },
        "by_source": {
            k: {"n": len(v), "accuracy": round(acc(v), 4)}
            for k, v in sorted(by_src.items())
        },
        "predictions": preds,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"\n{'='*64}")
    print(f"OVERALL  accuracy {results['accuracy']:.1%}   "
          f"(majority baseline {results['majority_baseline']:.1%})")
    print(f"{'='*64}")
    for k, v in results["by_category"].items():
        print(f"  {k:14} n={v['n']:<4} acc {v['accuracy']:.1%}   "
              f"baseline {v['majority_baseline']:.1%}")
    print()
    for k, v in results["by_source"].items():
        print(f"  {k:28} n={v['n']:<4} acc {v['accuracy']:.1%}")
    print(f"\nwrote {args.out}")


def compare(args):
    loaded = [json.load(open(p, encoding="utf-8")) for p in args.results]
    names = ["CPT" if r.get("adapter") else "Base" for r in loaded]

    files = {r["cloze_file"] for r in loaded}
    if len(files) > 1:
        print("*** WARNING: results were scored on DIFFERENT cloze files. "
              "These numbers are not comparable.\n")

    print(f"{'metric':<22}" + "".join(f"{n:>12}" for n in names) + f"{'delta':>12}")
    print("-" * (22 + 12 * (len(names) + 1)))

    def row(label, vals, pct=True):
        cells = "".join(f"{v:>11.1%}" if pct else f"{v:>12}" for v in vals)
        d = vals[-1] - vals[0] if len(vals) > 1 else 0.0
        print(f"{label:<22}{cells}{d:>+11.1%}")

    row("overall accuracy", [r["accuracy"] for r in loaded])
    print(f"{'  (majority baseline)':<22}"
          + "".join(f"{r['majority_baseline']:>11.1%}" for r in loaded))
    print()
    for cat in sorted(loaded[0]["by_category"]):
        row(cat, [r["by_category"][cat]["accuracy"] for r in loaded])
    print()
    for src in sorted(loaded[0]["by_source"]):
        row(src[:20], [r["by_source"][src]["accuracy"] for r in loaded])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--cloze", default="eval/legal_cloze.json")
    s.add_argument("--out", required=True)
    s.add_argument("--model_name", default="google/gemma-2-2b")
    s.add_argument("--adapter", default=None, help="omit to score the base model")
    s.add_argument("--max_context", type=int, default=512)
    s.add_argument("--max_new_tokens", type=int, default=6)
    s.add_argument("--batch_size", type=int, default=8)
    s.add_argument("--use_unsloth", action="store_true")
    s.set_defaults(func=score)

    c = sub.add_parser("compare")
    c.add_argument("--results", nargs="+", required=True)
    c.set_defaults(func=compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
