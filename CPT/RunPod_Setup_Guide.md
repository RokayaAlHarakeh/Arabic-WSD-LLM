# RunPod Setup — Step by Step

**Goal:** give the student a dedicated bf16-capable GPU for ~9 hours of work, on a $10 budget, with the supervisor keeping control of billing.

---

## Account structure: team account, supervisor as owner

**Do this, not the alternatives.**

| Option | Verdict |
|---|---|
| **Team account — you own it, she joins with the `dev` role** | ✅ **Use this.** You control billing; she can start, stop and manage pods herself; no shared password. Team accounts are free to create, with no per-member fees — billing stays per-GPU-hour. |
| Her own account, you send her money | ❌ No visibility, no control, and you cannot stop a forgotten pod |
| Share your login with her | ❌ She sees your billing and payment method; no audit trail |

### Roles

Runpod has four roles. The two that matter:

| Role | Can do | Cannot do |
|---|---|---|
| `basic` | Connect to **already-deployed** pods | **Start or stop pods**, create resources, see billing |
| **`dev`** | Deploy, start, stop and manage pods | See or change billing |

> ⚠️ **Give her `dev`, not `basic`.** The `basic` role cannot stop a pod — and stopping the pod is the single most important cost-control action. She must be able to do it herself without waiting for you.

---

# PART A — Supervisor setup (~10 minutes)

## A1. Create the account

Go to `console.runpod.io/signup`. Sign up with your email.

## A2. ⚠️ Convert to a team account BEFORE adding any credit

Team page → **Convert to a Team Account** → enter a team name (e.g. `CoLawyer-FYP`) → confirm.

> 🔴 **This order matters.** Credit added to a *personal* account does not transfer to a *team* account, and people have had to open support tickets to get it moved. Convert first, fund second.

## A3. Add credit

Billing → add **$10**.

Two things to know:
- Credits can take **30 seconds to 2 minutes** to appear. Don't panic, and don't try to deploy immediately.
- Runpod returns a generic *"Unauthorized"* / *"Invalid API key"* error for an **unfunded account** as well as for a bad key. If something fails early, check the balance before debugging anything else.

New accounts often receive **$5–10 in starting credits**, so your $10 may go further than expected.

## A4. Invite the student

Top-left account selector → make sure you are **in the team**, not your personal account → **Teams** → **Invite New Member** → assign role **`dev`** → **Copy Link** → send it to her.

## A5. Set a low-balance alert

Billing → notifications → alert at **$3 remaining**.

Runpod has **no built-in per-member spending limits**, so this alert is your only automatic backstop.

## A6. Create the network volume

**Do this before deploying any pod.**

Storage → **New Network Volume**:

| Setting | Value |
|---|---|
| Size | **30 GB** |
| Region | an **EU region** (lowest latency from Beirut; EU-RO-1 usually has the best 4090 availability) |
| Name | `fyp-cpt` |

**Write down the region.** Pods can only attach a network volume in the same region, so every pod from now on goes in that region.

Cost is roughly $0.07/GB/month → about **$2.10 for two weeks** at 30 GB.

### Why a network volume rather than container disk

The container disk is **destroyed when the pod is terminated**. The network volume is not. It holds the corpus, the installed packages, the checkpoints and the adapter — so she can terminate the pod every evening and lose nothing.

---

# PART B — Student setup (~15 minutes, once)

## B1. Join the team

Click the invite link → **Join Team**.

> ⚠️ Then click the **top-left account selector and switch to the team account**. Accepting the invite is not enough — until she switches, she is still in her personal account and will not see the shared volume or pods.

## B2. Deploy the pod

Pods → **Deploy**:

| Setting | Value |
|---|---|
| GPU | **RTX 4090 (24 GB)** |
| Cloud type | **Community Cloud** — $0.34/hr vs $0.74/hr on Secure |
| Region | **the same region as the network volume** |
| Template | `RunPod PyTorch 2.x` |
| Network volume | attach `fyp-cpt` at `/workspace` |
| Container disk | 20 GB is plenty |

If no 4090 is available in that region, the fallbacks in order: **RTX A5000** (cheaper, ~30% slower), **L4**, **A40**. All support bf16. **Do not take a T4, V100 or P100** — no bf16.

## B3. Connect

Pod card → **Connect** → **Jupyter Lab**. Opens in the browser, no SSH key needed.

For a terminal instead: **Connect → Web Terminal**.

## B4. Install once

```bash
cd /workspace
pip install unsloth trl peft bitsandbytes datasets accelerate transformers
```

Installing into `/workspace` means it survives pod termination. Verify:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(torch.cuda.is_bf16_supported())"   # must print True
```

> 🔴 If `torch.cuda.is_bf16_supported()` prints `False`, the wrong GPU was assigned. Terminate the pod and redeploy. Do not proceed.

## B5. Upload the data and code

JupyterLab file browser → navigate to `/workspace` → drag and drop:

- `legal_corpus_subset.jsonl.gz` (~38 MB)
- `cpt_student_bundle.zip` (~73 KB)

```bash
cd /workspace
gunzip legal_corpus_subset.jsonl.gz
unzip cpt_student_bundle.zip
```

## B6. Working directory layout

```
/workspace/
├── legal_corpus_subset.jsonl
├── cpt_student_bundle/
├── splits/                    # output of 03b_split_by_document.py
├── packed_2048/               # output of the packing script
├── eval/                      # eval set + results JSON
└── adapters/gemma2_2b_cpt_v1/ # OUTPUT_DIR for training
```

> 🔴 **Everything must live under `/workspace`.** Anything written elsewhere is lost when the pod is terminated.

Set `OUTPUT_DIR` in the training script to `/workspace/adapters/gemma2_2b_cpt_v1`.

---

# PART C — The cost rules

Runpod bills for **every minute a pod is running**, whether or not the GPU is doing anything. A 4090 left running for a weekend is about **$16** — more than the whole budget.

| Action | Billing | Keeps `/workspace`? |
|---|---|---|
| **Terminate pod** | stops | ✅ yes (network volume) |
| Leave pod running idle | **$0.34/hr** | ✅ |
| Delete the network volume | stops volume charge | ❌ **everything gone** |

### The rule

> **Terminate the pod whenever she is not actively using the GPU.** Redeploying takes under a minute, packages are already installed on `/workspace`, and nothing is lost.

At the end of every session:

1. Confirm checkpoints are under `/workspace/adapters/`
2. Terminate the pod
3. Leave the network volume alone

### Budget

| Item | Hours | Cost |
|---|---|---|
| Smoke tests, environment checks | 0.5 | $0.17 |
| Baseline evaluation | 0.5 | $0.17 |
| **Run A — CPT training, 27M tokens** | ~2.5 | $0.85 |
| Evaluation of Run A | 0.5 | $0.17 |
| Optional Run B | ~2.5 | $0.85 |
| Retry margin | ~2.5 | $0.85 |
| **GPU subtotal** | **~9 h** | **~$3.06** |
| Network volume, 30 GB, 2 weeks | | ~$2.10 |
| **Total** | | **~$5.20** |

**$10 leaves roughly 90% headroom.**

Data prep and building the evaluation set need **no GPU**. She can do those on her own laptop or in free Colab — no reason to run a 4090 while tokenizing.

---

# PART D — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| She joined but sees no pods or volume | Did not switch to the team account | Top-left account selector → select the team |
| Cannot stop a pod | Assigned `basic` role | Change her role to `dev` |
| Credit added but balance is $0 in the team | Funded the **personal** account | Convert to team **first**; contact support if already done |
| "Unauthorized" / "Invalid API key" | Often just a $0 balance | Check Billing before debugging keys |
| Network volume not offered at deploy | Pod region ≠ volume region | Deploy in the volume's region |
| `is_bf16_supported()` returns False | T4/V100/P100 assigned | Terminate, redeploy on 4090/A5000/L4/A40 |
| Packages gone after redeploy | Installed outside `/workspace` | Reinstall into `/workspace` |
| No 4090 in the region | Community capacity varies | Try A5000/L4/A40, or Secure Cloud at $0.74/hr |
| Pod disappeared mid-run | Community host interruption | Redeploy; the training script auto-resumes from the last checkpoint |

---

# Quick reference

**Supervisor:** sign up → **convert to team** → fund $10 → invite her as `dev` → low-balance alert → create 30 GB EU network volume

**Student:** join → **switch to team** → deploy 4090 Community in the volume's region → attach volume at `/workspace` → JupyterLab → install into `/workspace` → upload data → **terminate when done**
