# ── 0. Imports & model load ──────────────────────────────────────────────
import os, re, json
from tqdm import tqdm
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from transformers import GenerationConfig
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")          # optional; passed to from_pretrained

# ── CONFIG (same scheme as finetuning.py; override via env vars) ──────────
PROJECT_DIR = os.environ.get("WSD_PROJECT_DIR", "/content/drive/MyDrive/WSD_Project")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODEL_TAG   = os.environ.get("WSD_MODEL_TAG", "gemma2_2b")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", MODEL_TAG)
# The merged 16-bit dir written by finetuning.py:
MODEL_DIR   = os.environ.get("WSD_MODEL_DIR", os.path.join(OUTPUT_DIR, "merged_16bit"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=int(os.environ.get("WSD_MAX_SEQ_LEN", 1024)),
    load_in_4bit=True,
    dtype=None,
    token=HF_TOKEN,
    local_files_only=True,
)
FastLanguageModel.for_inference(model)        # 2× faster kernels

EOS = tokenizer.eos_token
EOS_ID = tokenizer.eos_token_id       # int, like 128001


# ── 1. Alpaca-style prompt template ──────────────────────────────────────
ALPACA_PROMPT = """Below is an instruction that describes a task, \
paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

WSD_INSTRUCTION = (
    "You are tasked with performing Word Sense Disambiguation (WSD). Your job is to analyze the given sentence and identify the correct sense for the target word based on the context. For each sense, you are provided with a Sense ID and its definition. Using the context of the sentence, choose the most appropriate sense definition and provide the corresponding Sense ID."
)

def build_input_block(sentence, word, senses, dictionary):
    """Format the block that goes into the '### Input:' section.

    MUST match create_finetuning_dataset.py byte-for-byte (single quotes,
    bracketed senses joined by ', ' on one line) — otherwise the model sees an
    out-of-distribution prompt and never emits a clean ID (all preds → none).
    """
    possible_senses = [
        f"[Sense ID: {sid}, Definition: {dictionary[sid]}]"
        for sid in senses
        if sid in dictionary
    ]
    if not possible_senses:
        possible_senses = ["none"]
    return (
        f"Sentence: '{sentence}'\n"
        f"Target Word: '{word}'\n"
        f"Possible Senses:\n" + ", ".join(possible_senses)
    )

def make_prompt(sentence, word, senses, dictionary):
    input_block = build_input_block(sentence, word, senses, dictionary)
    return ALPACA_PROMPT.format(WSD_INSTRUCTION, input_block, "") 

# ── 2. Regex patterns for Sense-ID extraction ────────────────────────────
REGEX_PATTERNS = [
    r"^\s*(\d+)\s*<eos>",       # case 1: 303048730<eos>
    r"^\s*(\d+)\s*$",           # case 2: just 303048730
]


def extract_sense_id(text, sense_candidates):
    """Return the first valid ID from text that exists in candidate senses."""
    for pattern in REGEX_PATTERNS:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        for match in matches:
            sid = int(match)
            if sid in sense_candidates:
                return sid
    return None


# ── 3. Prediction helper ────────────────────────────────────────────────
GEN_CONFIG = GenerationConfig(
    max_new_tokens = 128,        # one or two tokens is enough for an ID
    do_sample      = False,
    eos_token_id   = EOS_ID,
    pad_token_id   = EOS_ID,
    use_cache      = True,
)

def predict_sense(sentence, word, senses, dictionary):
    prompt = make_prompt(sentence, word, senses, dictionary)
    # inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True     # adds Gemma's <bos>
    ).to("cuda")
    outputs = model.generate(**inputs, generation_config=GEN_CONFIG)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=False).strip()
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    sense_id = extract_sense_id(response, set(senses))
    return sense_id if sense_id is not None else "none", response

# ── 4. Batch runner with logging + resumable checkpoint ──────────────────
def run_prediction(test_path, dict_path, output_path, debug_path, ckpt_path, limit=None):
    test_data = json.load(open(test_path, encoding="utf-8"))
    dictionary = {
        d["sense_id"]: d["definition"]
        for d in json.load(open(dict_path, encoding="utf-8"))
    }

    if limit is not None:
        test_data = test_data[:limit]
        print(f"⚠️ WSD_LIMIT set — running only the first {limit} sentences (smoke test)")

    # ── Resume: reload sentences finished by a previous (interrupted) run. ──
    # Each sentence is appended to ckpt_path (JSONL) only AFTER all its words
    # are done, so a disconnect mid-sentence just re-runs that whole sentence.
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["sentence_id"]] = rec
        print(f"▶ Resuming: {len(done)} sentences already done; skipping them")

    fresh = not done
    with open(debug_path, "w" if fresh else "a", encoding="utf-8") as log, \
         open(ckpt_path,  "w" if fresh else "a", encoding="utf-8") as ckpt:
        if fresh:
            log.write("===== MODEL PREDICTIONS =====\n")
        for sentence in tqdm(test_data, desc="Predicting"):
            sent_id   = sentence["sentence_id"]
            if sent_id in done:
                continue
            sent_txt  = sentence["sentence"]
            word_preds = []

            for w in sentence["words"]:
                senses = w.get("senses", [])
                if senses:
                    pred_id, resp = predict_sense(sent_txt, w["word"], senses, dictionary)
                else:
                    pred_id, resp = "none", ""

                word_preds.append({
                    "word_id": w["word_id"],
                    "word": w["word"],
                    "target_sense": pred_id,
                })

                # Debug log
                log.write(f"\n===== Query for Word: {w['word']} =====\n")
                log.write(f"===== Sentence ID: {sent_id} | Word: {w['word']} =====\n")
                log.write(f"Prompt:\n{make_prompt(sent_txt, w['word'], senses, dictionary)}\n")
                log.write(f"Response:\n{resp}\n")
                log.write(f"Prediction: {pred_id}\n")

            sent_pred = {
                "sentence_id": sent_id,
                "sentence": sent_txt,
                "words": word_preds,
            }
            done[sent_id] = sent_pred
            # Persist this sentence immediately so a disconnect can't lose it.
            ckpt.write(json.dumps(sent_pred, ensure_ascii=False) + "\n")
            ckpt.flush()
            log.flush()

    # ── Assemble final predictions in test_set order (eval.py aligns by position). ──
    predictions = [done[s["sentence_id"]] for s in test_data if s["sentence_id"] in done]
    json.dump(predictions, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"✅ Predictions saved to {output_path}  ({len(predictions)} sentences)")
    print(f"🪵 Debug log saved to {debug_path}")
    print(f"🧷 Resume checkpoint: {ckpt_path}  (delete it to force a clean re-run)")

# ── 5. Run! ──────────────────────────────────────────────────────────────
# Set WSD_LIMIT=5 for a ~30s smoke test before committing to the full pass.
os.makedirs(OUTPUT_DIR, exist_ok=True)
_limit = os.environ.get("WSD_LIMIT")
run_prediction(
    test_path   = os.path.join(DATA_DIR, "test_set.json"),
    dict_path   = os.path.join(DATA_DIR, "test_dictionary.json"),
    output_path = os.path.join(OUTPUT_DIR, f"predictions_{MODEL_TAG}.json"),
    debug_path  = os.path.join(OUTPUT_DIR, f"debug_{MODEL_TAG}.txt"),
    ckpt_path   = os.path.join(OUTPUT_DIR, f"predictions_{MODEL_TAG}.partial.jsonl"),
    limit       = int(_limit) if _limit else None,
)