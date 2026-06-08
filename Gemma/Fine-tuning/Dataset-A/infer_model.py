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
    """Format the block that goes into the '### Input:' section."""
    senses_lines = [
        f"- Sense ID: {sid}, Definition: {dictionary.get(sid, '[Missing]')}"
        for sid in senses
    ]
    return (
        f"Sentence: \"{sentence}\"\n"
        f"Target Word: \"{word}\"\n"
        f"Possible Senses:\n" + "\n".join(senses_lines)
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

# ── 4. Batch runner with logging ─────────────────────────────────────────
def run_prediction(test_path, dict_path, output_path, debug_path):
    test_data = json.load(open(test_path, encoding="utf-8"))
    dictionary = {
        d["sense_id"]: d["definition"]
        for d in json.load(open(dict_path, encoding="utf-8"))
    }

    predictions = []
    with open(debug_path, "w", encoding="utf-8") as log:
        log.write("===== MODEL PREDICTIONS =====\n")
        for sentence in tqdm(test_data, desc="Predicting"):
            sent_txt  = sentence["sentence"]
            sent_id   = sentence["sentence_id"]
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

            predictions.append({
                "sentence_id": sent_id,
                "sentence": sent_txt,
                "words": word_preds,
            })

    json.dump(predictions, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"✅ Predictions saved to {output_path}")
    print(f"🪵 Debug log saved to {debug_path}")

# ── 5. Run! ──────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
run_prediction(
    test_path   = os.path.join(DATA_DIR, "test_set.json"),
    dict_path   = os.path.join(DATA_DIR, "test_dictionary.json"),
    output_path = os.path.join(OUTPUT_DIR, f"predictions_{MODEL_TAG}.json"),
    debug_path  = os.path.join(OUTPUT_DIR, f"debug_{MODEL_TAG}.txt"),
)