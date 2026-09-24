"""
Qualitative base-vs-CPT comparison on legal prompts.

Day-10 deliverable in FYP_Timeline.pdf: "10-15 legal prompts, base output vs CPT output
side by side. Acceptance: appendix section with Arabic examples."

Both columns come from ONE loaded model. `disable_adapter()` turns the CPT LoRA off, so the
base column is provably the same base weights rather than a separately loaded model --
faster, and it removes a confound.

Usage:
    python scripts/qualitative_compare.py \
        --adapter /workspace/adapters/gemma2_2b_cpt_v1/final_adapter \
        --out /workspace/eval/qualitative.json
"""

import argparse
import json

# peft on some stacks hard-errors on an old torchao preinstall; we never use torchao.
try:
    import peft.import_utils as _pi, peft.tuners.lora.torchao as _pt
    _pi.is_torchao_available = _pt.is_torchao_available = lambda: False
except Exception:
    pass

PROMPTS = [
    "بناء على المرسوم رقم",
    "إن رئيس الجمهورية، بناء على",
    "المادة الأولى: يحق لكل",
    "تنص المادة الثانية من قانون العمل على",
    "حكمت المحكمة",
    "الجريدة الرسمية اللبنانية",
    "وحيث أن الاجتهاد مستقر على",
    "يعاقب بالحبس من",
    "لجنة الخدمة المدنية",
    "على وزير الداخلية والبلديات",
    "عقد العمل هو",
    "تسري أحكام هذا القانون على",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="/workspace/eval/qualitative.json")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

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
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("adapter attached:", args.adapter)

    def generate(prompt, adapter_on):
        enc = tok(prompt, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  pad_token_id=tok.eos_token_id)
        with torch.no_grad():
            if adapter_on:
                out = model.generate(**enc, **kw)
            else:
                with model.disable_adapter():
                    out = model.generate(**enc, **kw)
        return tok.decode(out[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    rows = []
    for i, prompt in enumerate(PROMPTS, 1):
        base = generate(prompt, False)
        cpt = generate(prompt, True)
        rows.append({"prompt": prompt, "base": base, "cpt": cpt})
        print("\n" + "=" * 72)
        print(f"[{i}] {prompt}")
        print("-" * 72)
        print(f"BASE: {base}")
        print(f"CPT : {cpt}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {len(rows)} prompt pairs -> {args.out}")


if __name__ == "__main__":
    main()
