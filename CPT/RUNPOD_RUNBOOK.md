# RunPod runbook — Gemma 2-2B CPT

Replaces the Colab notebook. Copy-paste into the pod's **Web Terminal** or a JupyterLab
terminal, top to bottom.

Volume: `cpt_volume_rokaya`, 30 GB, **EU-RO-1**, id `0bs9wr1ndx`.
Every pod must be deployed in **EU-RO-1** or the volume will not be offered.

---

## 1. Which GPU

Available in EU-RO-1 with this volume attached:

| GPU | VRAM | $/hr | Arch | Verdict |
|---|---|---|---|---|
| **RTX 4090** | 24 GB | **$0.74** | Ada (sm_89) | ✅ **Take this** |
| L4 | 24 GB | $0.49 | Ada (sm_89) | ✅ safe fallback, ~2× slower |
| RTX PRO 4000 | 24 GB | $0.57 | Blackwell | ⚠️ see below |
| RTX PRO 4500 | 32 GB | $0.72 | Blackwell | ⚠️ see below |
| RTX PRO 6000 | 96 GB | $2.09 | Blackwell | ❌ 96 GB for a 7 GB job |

All five support bf16, so none is disqualified the way a T4 would be.

**Why the 4090.** Peak VRAM for this run is ~7 GB, so 24 GB is already generous — **VRAM is
not the constraint, time is**. The 4090 has roughly 3× the memory bandwidth of an L4 and is
the card the plan's timing estimates were calibrated against.

**Why not the RTX PRO cards, despite RTX PRO 4000 being cheaper.** They are Blackwell
(sm_120), which needs a recent CUDA and a PyTorch built for it. `bitsandbytes` 4-bit kernels
are where this bites — if the template's build predates Blackwell support, quantised loading
fails or silently falls back. Ada (sm_89) has been the mainstream QLoRA target for years. On
a run you need to simply work, that is not a variable worth introducing.

If you want to try one anyway, the smoke test in §6 tells you within five minutes. That is
what it is for.

**On cost:** L4 at $0.49 looks cheaper per hour, but at roughly twice the wall-clock the
total for a run lands in the same place — while doubling your exposure to an interruption.
Take the 4090.

### Budget at $0.74/hr

The setup guide assumed $0.34/hr Community pricing. These are Secure Cloud rates, so
recompute:

| Activity | Hours | Cost |
|---|---|---|
| Smoke test + environment | 0.5 | $0.37 |
| Baseline: next-token + cloze | 0.75 | $0.56 |
| **Run A — training** | ~2.5 | **$1.85** |
| CPT eval: next-token + cloze | 0.5 | $0.37 |
| Retention, 2 runs | 1.25 | $0.93 |
| Qualitative | 0.25 | $0.19 |
| **GPU subtotal** | **~5.75 h** | **~$4.27** |
| Volume, 30 GB | — | $2.10/month |
| **Total** | | **~$6.40** |

That leaves ~$3.60 for one retry, and **no room for the optional Run B**. Treat Run B as
out of scope unless Run A passes first time and credit remains.

---

## 2. Deploy the pod

Pods → Deploy:

| Setting | Value |
|---|---|
| Region | **EU-RO-1** (must match the volume) |
| GPU | RTX 4090, **1×** (not 2) |
| Template | RunPod PyTorch 2.x |
| Network volume | `cpt_volume_rokaya` → mounted at `/workspace` |
| Container disk | 20 GB |

> 🔴 **Terminate the pod whenever you are not actively using the GPU.** RunPod bills every
> running minute, idle or not. At $0.74/hr a pod left over a weekend is ~$37 — more than the
> entire budget. Redeploying takes under a minute and `/workspace` survives.

---

## 3. One-time setup

### 3.1 Verify the GPU before anything else

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('bf16:', torch.cuda.is_bf16_supported())"
```

`bf16: True` is required. If it prints `False`, terminate and redeploy.

### 3.2 Install packages so they actually persist

> ⚠️ The setup guide says `cd /workspace && pip install ...` persists the packages. **It does
> not.** `pip` installs into the system `site-packages` on the *container disk*, which is
> destroyed on termination — the working directory has no effect on where pip installs.

Use a venv on the volume instead. `--system-site-packages` inherits `torch` from the image,
so only our additions live on the network volume:

```bash
python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip install -U transformers peft bitsandbytes datasets accelerate sentencepiece
```

**Every new session starts with:**

```bash
source /workspace/venv/bin/activate
```

Verify:

```bash
python -c "import torch, transformers, peft, bitsandbytes; \
print(torch.__version__, transformers.__version__, peft.__version__)"
```

### 3.3 Hugging Face

```bash
huggingface-cli login          # paste your read token
```

`google/gemma-2-2b` is gated; you have already accepted the licence, so this is all that is
needed. The token is stored under `~/.cache` on the container disk, so **repeat it each
session** — or persist it:

```bash
export HF_HOME=/workspace/hf
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
```

### 3.4 Clone the repo

```bash
cd /workspace
git clone -b week2-cpt https://github.com/RokayaAlHarakeh/Arabic-WSD-LLM.git
cd Arabic-WSD-LLM/CPT
```

---

## 4. Get the data onto the volume

Stage A is **already done** — splits, packed files, eval set, cloze probe were all built on
Colab and are on Google Drive. Two routes.

### Route A — upload with no pod running (cheapest)

The volume has an S3 endpoint, so files can be pushed to it **without any pod**, and
therefore at zero GPU cost. Create an S3 API key in the RunPod console, then from your
laptop:

```bash
aws s3 cp legal_corpus_subset.jsonl.zip \
  s3://0bs9wr1ndx/ --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io

aws s3 cp eval/ s3://0bs9wr1ndx/eval/ --recursive \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
```

Upload the **small** files this way: the corpus zip (35 MB), `eval_set_1000.json` and
`legal_cloze.json`. Skip the packed files — they are ~400–600 MB and regenerate in minutes.

### Route B — regenerate on the pod

Deterministic and verified: the same seed produced byte-identical output on Windows and on
Colab Linux, so regeneration is safe.

```bash
cd /workspace
unzip -o legal_corpus_subset.jsonl.zip
wc -l legal_corpus_subset.jsonl          # must be 45636

cd /workspace/Arabic-WSD-LLM/CPT

python scripts/03b_split_by_document.py \
  --input  /workspace/legal_corpus_subset.jsonl \
  --outdir /workspace/splits \
  --report /workspace/reports/03b_split_by_document.txt

for SPLIT in train val test; do
  python cpt_student_bundle/01_data_prep/02_pack_dataset_4096.py \
    --model_name google/gemma-2-2b \
    --input_file  /workspace/splits/${SPLIT}_doclevel.jsonl \
    --output_file /workspace/packed_2048/${SPLIT}_packed_2048.jsonl \
    --report_file /workspace/reports/${SPLIT}_packed_2048.txt \
    --max_seq_length 2048 --append_eos --drop_remainder
done
```

**Expected — check against Stage A on Colab:**

| | Value |
|---|---|
| Train / val / test docs | 39,310 / 2,185 / 2,185 |
| Train blocks | **13,262** |
| Train tokens | **27,162,160** |
| Optimizer steps | **1,658** |

Anything different means the input differs — stop and find out why.

> 💡 This is ~40 minutes of pure CPU work. At $0.74/hr that is ~$0.50 on a 4090 doing
> nothing. If you want it free, deploy a **CPU-only pod** (the CPU tab at deploy time) with
> the same volume, run this, terminate, then deploy the 4090.

> 🔴 **Do not rebuild the eval set or the cloze probe.** Upload those two files. Rebuilding
> between base and CPT is the one mistake that silently invalidates the whole comparison.

---

## 5. Directory layout

```
/workspace/
├── venv/                       # packages (survives termination)
├── hf/                         # HF cache
├── legal_corpus_subset.jsonl
├── Arabic-WSD-LLM/CPT/         # scripts
├── splits/
├── packed_2048/
├── reports/
├── eval/                       # eval_set_1000.json, legal_cloze.json, results
└── adapters/gemma2_2b_cpt_v1/  # OUTPUT_DIR
```

The training script now defaults to `/workspace`; override with `WSD_WORKSPACE` if needed.

---

## 6. Smoke test — not optional

```bash
cd /workspace/Arabic-WSD-LLM/CPT
python scripts/05_full_training_gemma2_2b.py \
  --dry_run_samples 64 --num_epochs 1 \
  --output_dir /workspace/smoke_test --skip_s3_upload
```

Four things must hold:

| Check | Expected |
|---|---|
| trainable params | ~0.5–1.5%. 0% or ~100% → LoRA did not attach |
| sanity probe | `OK - well below random` (random ≈ 12.46) |
| loss | finite, decreasing, no NaN |
| adapter saved | `adapter_config.json` + `adapter_model.safetensors` |

**Write down seconds/step.** Full run ≈ that × 1,658. At 4–7 s/step expect 1.8–3.2 h.

---

## 7. Hypothesis, then baseline, then train

```bash
# 7.1 baseline - next-token (base model, no adapter)
python cpt_student_bundle/03_evaluation/next_token_eval_gemma.py score \
  --eval_set /workspace/eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --out /workspace/eval/results_base.json

# 7.2 baseline - cloze
python scripts/score_legal_cloze.py score \
  --cloze /workspace/eval/legal_cloze.json \
  --model_name google/gemma-2-2b \
  --out /workspace/eval/cloze_base.json

# 7.3 train  (~2-3 h; resumes automatically if interrupted)
python scripts/05_full_training_gemma2_2b.py --skip_s3_upload
```

Write the dated hypothesis into `CPT/HYPOTHESIS.md` and commit it **before** 7.3.

If the pod dies mid-run, redeploy and re-run 7.3 unchanged — checkpoints are on the volume
and resume needs no flags.

---

## 8. Evaluate

```bash
A=/workspace/adapters/gemma2_2b_cpt_v1/final_adapter

python cpt_student_bundle/03_evaluation/next_token_eval_gemma.py score \
  --eval_set /workspace/eval/eval_set_1000.json \
  --model_name google/gemma-2-2b --adapter $A \
  --out /workspace/eval/results_cpt.json

python cpt_student_bundle/03_evaluation/next_token_eval_gemma.py compare \
  --results /workspace/eval/results_base.json /workspace/eval/results_cpt.json

python scripts/score_legal_cloze.py score \
  --cloze /workspace/eval/legal_cloze.json \
  --model_name google/gemma-2-2b --adapter $A \
  --out /workspace/eval/cloze_cpt.json

python scripts/score_legal_cloze.py compare \
  --results /workspace/eval/cloze_base.json /workspace/eval/cloze_cpt.json
```

Then retention on Dataset A (base zero-shot, then base + CPT adapter) and the qualitative
prompts — same as notebook §11 and §11.5, with `/workspace` paths.

The gate is read **after all four measurements**, not before.

---

## 9. End of every session

```bash
ls -la /workspace/adapters/gemma2_2b_cpt_v1/        # checkpoints present?
cd /workspace/Arabic-WSD-LLM && git add -A && git commit -m "..." && git push
```

Then **terminate the pod**. Leave the network volume alone.

| Action | Billing | Keeps `/workspace`? |
|---|---|---|
| Terminate pod | stops | ✅ |
| Leave pod idle | $0.74/hr | ✅ |
| Delete network volume | stops | ❌ everything gone |
