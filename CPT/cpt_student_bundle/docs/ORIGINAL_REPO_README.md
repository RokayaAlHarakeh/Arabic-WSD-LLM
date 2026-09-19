# Lebanese Legal CPT with Gemma 4 12B

## Project overview

This repository documents and organizes the data-preparation, smoke-testing, and evaluation-analysis layer of a Lebanese legal continued pretraining (CPT) project.

The project is based on:

- **Base model:** `google/gemma-4-12B`
- **Training framework:** Unsloth / QLoRA preparation
- **Training method:** 4-bit LoRA continued pretraining
- **Domain:** Lebanese legal Arabic
- **Final sequence length:** 4096 tokens

The purpose of the CPT phase is to adapt Gemma 4 12B to Lebanese legal language, terminology, document structure, legislation, jurisprudence, Official Gazette material, and related legal documents.

This repository is designed to make the preprocessing and project analysis reproducible while keeping very large datasets, model checkpoints, adapters, and generated runtime artifacts outside GitHub.

---

## Repository scope

This repository currently contains the **preprocessing and documentation layer** of the project:

- source-specific data inspection scripts;
- source-specific preprocessing and chunking scripts;
- quality-control scripts;
- dataset combination and splitting scripts;
- tokenization and 4096-token packing scripts;
- dataset documentation and preprocessing reports;
- Unsloth environment verification scripts;
- packed-dataset validation scripts;
- a short CPT smoke-test script;
- evaluation documentation, including Claude Sonnet baseline results and Gemma CPT checkpoint analysis.

The repository does **not** contain the full training dataset, raw source documents, model checkpoints, LoRA adapters, Hugging Face caches, or large evaluation prediction outputs.

---

## Current status

The preprocessing pipeline has been completed and organized under:

```text
data_preprocessing/
```

The final cleaned and packed CPT dataset was prepared under:

```text
data_preprocessing/data_cpt/final_cpt_v1/
```

A short Unsloth smoke-test workflow is prepared under:

```text
training_codes/
```

Evaluation documentation is organized under:

```text
evaluation/
```

The full training and full evaluation execution are intended to run on the EC2 training/evaluation environment. This GitHub repository keeps the scripts, documentation, selected small files, and result summaries needed to understand and reproduce the project logic.

---

## Repository structure

```text
CPT/
├── data_preprocessing/
│   ├── data_raw/
│   │   ├── adl/
│   │   ├── bibliographic_rulings/
│   │   ├── gazette/
│   │   ├── legislations/
│   │   ├── provisions/
│   │   └── studies/
│   │
│   ├── data_cpt/
│   │   ├── combined_data/
│   │   │   └── cpt_full_legal_v1.jsonl
│   │   │
│   │   ├── final_cpt_v1/
│   │   │   ├── clean/
│   │   │   ├── packed/
│   │   │   └── reports/
│   │   │
│   │   ├── samples/
│   │   ├── source_cpt_files/
│   │   └── split_4096_archive/
│   │
│   ├── data_preprocessing_scripts/
│   │   ├── 01_data_inspection/
│   │   ├── 02_data_processing/
│   │   ├── 03_quality_checks/
│   │   ├── 04_full_dataset_building/
│   │   ├── 05_tokenization_4096/
│   │   ├── 06_cleaning_and_packing/
│   │   └── 99_utilities_archive/
│   │
│   ├── reports/
│   └── DATA_PREPROCESSING_PHASE.md
│
├── evaluation/
│   ├── continuous/
│   │   ├── data/
│   │   ├── predictions/
│   │   ├── continuous_next_word_build_dataset.py
│   │   └── continuous_next_word_inference_sonnet_bedrock.py
│   │
│   ├── next_token_analysis/
│   │   ├── data/
│   │   ├── readable_reports_html/
│   │   ├── s3_results/
│   │   ├── scripts/
│   │   └── README_selected_token.md
│   │
│   ├── next_word/
│   │   ├── predictions/
│   │   └── scripts/
│   │
│   └── evaluation.md
│
├── training_codes/
│   ├── 00_unsloth_verify.py
│   ├── 00_testing_dataset.py
│   ├── 00_5_step_training.py
│   └── requirements.txt
│
├── .gitignore
├── README.md
└── visual.py
```

---

## Data sources

The CPT dataset was built from several Lebanese legal data sources:

| Source | Description |
|---|---|
| ADL rulings | Lebanese jurisprudence / court ruling material |
| Official Gazette legal core | Official Gazette legal material, mainly legal sections |
| Official Gazette Section 2 sample | Sampled Section 2 Gazette material |
| Related provisions | Legal provisions linked to legislation or jurisprudence |
| Bibliographic rulings | Ruling references and legal bibliography material |
| Legislations | Lebanese legislation records |
| Associated Arabic studies | Arabic legal studies and related legal research material |

Each source was first inspected, then processed into source-specific CPT JSONL files, then checked before being merged into the full CPT dataset.

---

## Preprocessing pipeline

The preprocessing workflow follows this structure:

```text
Raw legal documents
        ↓
Source-specific inspection
        ↓
Source-specific extraction, cleaning, and chunking
        ↓
Quality checks
        ↓
Combined legal CPT dataset
        ↓
Long-record correction
        ↓
Train / validation / test split
        ↓
Header cleaning
        ↓
Gemma tokenization
        ↓
Packing into 4096-token blocks
        ↓
Dataset validation
        ↓
Training / evaluation handoff
```

---

## 1. Data inspection

Inspection scripts are located under:

```text
data_preprocessing/data_preprocessing_scripts/01_data_inspection/
```

This phase was used to understand the structure, size, fields, and quality of each raw source before processing.

Scripts include:

```text
01_inspect_adl.py
02_inspect_gazette.py
03_inspect_provisions.py
04_inspect_bibliographic_rulings.py
05_inspect_legislations.py
06_inspect_studies.py
```

The goal of this phase was to answer:

- What fields exist in each source?
- Which fields are useful for CPT?
- Which sources require chunking?
- Which records are too short or too long?
- Which metadata should be preserved?
- Which documents are Arabic and legally relevant?

---

## 2. Source-specific processing

Processing scripts are located under:

```text
data_preprocessing/data_preprocessing_scripts/02_data_processing/
```

Scripts include:

```text
01_chunk_adl.py
02_chunk_gazette.py
03_process_provisions.py
04_process_bibliographic_rulings.py
05_process_gazette_section2.py
06_prepare_legislations.py
07_prepare_studies.py
```

This phase converts raw source files into source-specific CPT JSONL files.

The source-specific outputs are stored under:

```text
data_preprocessing/data_cpt/source_cpt_files/
```

Examples:

```text
cpt_chunked_adl_v1.jsonl
cpt_gazette_legal_core_v1.jsonl
cpt_gazette_section2_sample10_v1.jsonl
cpt_related_provisions_v1.jsonl
cpt_bibliographic_rulings_v1.jsonl
cpt_legislations_v1.jsonl
cpt_associated_studies_ar_v1.jsonl
```

---

## Chunking configuration

Most long legal sources were chunked using approximately the same character-based configuration:

```python
MAX_CHARS = 20000
OVERLAP_CHARS = 1200
MIN_CHARS = 300
MIN_CHUNK_CHARS = 3000
```

Meaning:

| Parameter | Purpose |
|---|---|
| `MAX_CHARS = 20000` | Maximum approximate character length for one source-level chunk |
| `OVERLAP_CHARS = 1200` | Character overlap between consecutive chunks to preserve continuity |
| `MIN_CHARS = 300` | Very short records below this threshold are usually skipped |
| `MIN_CHUNK_CHARS = 3000` | Minimum preferred size for generated chunks when splitting long records |

This stage is **character-based chunking** and should not be confused with the final 4096-token packing stage.

Character-based chunking prepares readable legal text records. Token packing later converts the cleaned text into fixed-length 4096-token model training blocks.

---

## 3. Quality checks

Quality-check scripts are located under:

```text
data_preprocessing/data_preprocessing_scripts/03_quality_checks/
```

Scripts include:

```text
01_check_adl_chunked.py
02_check_gazette_chunked.py
03_check_provisions_processed.py
04_check_bibliographic_rulings_processed.py
05_check_gazette_section2.py
06_check_legislations_processed.py
07_check_studies_processed.py
08_check_gemma_tokenizer.py
09_check_4096_token_dataset.py
```

The checks verify:

- valid JSONL formatting;
- no empty text records;
- no unexpected missing fields;
- source distribution;
- approximate length distribution;
- tokenizer compatibility;
- 4096-token suitability;
- duplicate or problematic records;
- whether all expected sources are represented.

---

## 4. Full dataset building

Dataset-building scripts are located under:

```text
data_preprocessing/data_preprocessing_scripts/04_full_dataset_building/
```

Scripts include:

```text
01_combine_cpt_full_legal_v1.py
02_fix_long_records.py
03_split_train_val_test.py
04_create_sample_dataset.py
```

This phase combines all approved source-specific CPT files into one full CPT dataset.

The combined dataset is stored under:

```text
data_preprocessing/data_cpt/combined_data/
```

The split archive is stored under:

```text
data_preprocessing/data_cpt/split_4096_archive/
```

The train / validation / test split follows approximately:

```text
Train: 97%
Validation: 2%
Test: 1%
```

---

## 5. Tokenization and 4096-token preparation

Tokenization scripts are located under:

```text
data_preprocessing/data_preprocessing_scripts/05_tokenization_4096/
```

The dataset was prepared for the Gemma tokenizer using:

```text
google/gemma-4-12B
```

The target context length is:

```text
4096 tokens
```

The 4096-token preparation ensures that legal text records are compatible with the CPT training context window.

---

## 6. Header cleaning and final packing

Cleaning and packing scripts are located under:

```text
data_preprocessing/data_preprocessing_scripts/06_cleaning_and_packing/
```

Scripts include:

```text
01_clean_text_headers.py
02_pack_dataset_4096.py
```

### Header cleaning

The purpose of header cleaning is to keep useful metadata inside the training text while removing noisy tags.

Useful compact metadata includes fields such as:

```text
source
type
authority
year
journal_section
legal_domain / domain
```

Noisy identifiers and internal processing fields are removed from the model-visible text, such as:

```text
issue
part
doc_index
page_numbers
nb_tables
chunk
source_id
original_id_before_final_combine
final_cpt_file
```

This improves the quality of the training signal by preserving legal context while reducing artificial metadata noise.

### Packing

After cleaning, the dataset is tokenized and packed into fixed 4096-token blocks.

The final packed train file contains:

```text
Input records before packing: 196,983
Packed train blocks: 31,151
Input tokens including EOS: 127,594,907
Output useful tokens: 127,594,496
Dropped tokens: 411
Approx optimizer steps with gradient_accumulation_steps=8: 3,893.88
```

Packing reduces the number of training rows while preserving almost all useful tokens.

Before packing:

```text
196,983 training rows
```

After packing:

```text
31,151 packed training rows
```

This is approximately a 6.3x reduction in optimizer steps compared with training directly on the un-packed records.

---

## Final dataset paths

The final cleaned dataset is stored under:

```text
data_preprocessing/data_cpt/final_cpt_v1/clean/
```

Expected files:

```text
train_cleanheaders.jsonl
val_cleanheaders.jsonl
test_cleanheaders.jsonl
```

The final packed dataset is stored under:

```text
data_preprocessing/data_cpt/final_cpt_v1/packed/
```

Expected files:

```text
train_cleanheaders_packed_4096.jsonl
val_cleanheaders_packed_4096.jsonl
test_cleanheaders_packed_4096.jsonl
```

The full packed training file is excluded from GitHub because of size:

```text
train_cleanheaders_packed_4096.jsonl
```

The validation and test packed files may be included to allow schema validation, smoke testing, and evaluation reproduction.

---

## GitHub data policy

Large legal datasets and generated artifacts are excluded from GitHub.

Excluded from this repository:

- raw legal source documents;
- full combined CPT datasets;
- source-specific full CPT JSONL files;
- split archive datasets;
- full cleaned training file;
- full packed training file;
- backup files;
- temporary files;
- model checkpoints;
- LoRA adapters;
- Hugging Face model caches;
- S3 checkpoint downloads;
- large prediction JSON/JSONL files;
- runtime logs.

Included or recommended for inclusion:

- preprocessing scripts;
- quality-control scripts;
- dataset-building scripts;
- tokenization and packing scripts;
- documentation;
- README files;
- small validation/test files if needed for schema verification;
- evaluation summaries and result tables;
- smoke-test scripts;
- requirements files for the lightweight test workflow.

This keeps the repository readable and usable while avoiding GitHub size limits and accidental publication of large or sensitive data.

---

## Training smoke-test workflow

Training smoke-test scripts are located under:

```text
training_codes/
```

Files:

```text
00_unsloth_verify.py
00_testing_dataset.py
00_5_step_training.py
requirements.txt
```

These scripts are used to check the EC2/Unsloth environment before starting any long training run.

The checks verify:

- Unsloth imports correctly;
- GPU is visible;
- `google/gemma-4-12B` can load in 4-bit;
- LoRA adapters attach correctly;
- packed `input_ids` and `attention_mask` files can be read;
- a short CPT smoke test runs;
- loss is not NaN;
- adapter saving works;
- the script prints seconds per optimizer step.

The smoke test is intentionally short and is not intended to measure final model quality.

---

## EC2 training handoff

The EC2 instance receives the required files through a compressed archive such as:

```text
cpt_ec2_upload.zip
```

The archive should contain:

```text
training_codes/
data_preprocessing/data_cpt/final_cpt_v1/packed/


Run checks in order:

```bash
python training_codes/00_unsloth_verify.py
python training_codes/00_testing_dataset.py
python training_codes/00_5_step_training.py
```

The full training time can be estimated from the smoke-test speed:

```text
full training time = seconds_per_optimizer_step × 3894 optimizer steps
```
All details about training have been rewritten in the other repository, so the training code can be discarded for now.


---

## Evaluation documentation

Evaluation documentation is organized under:

```text
evaluation/
```

The repository includes analysis and documentation for:

- Claude Sonnet 4.5 rolling continuous complete next-word evaluation;
- Claude Sonnet 4.5 standalone next-word evaluation;
- Gemma CPT selected-token checkpoint analysis;
- comparison between base Gemma and CPT checkpoints;
- checkpoint learning dynamics;
- convergence behavior across the first CPT epoch.

### Claude Sonnet baseline summary

Claude Sonnet 4.5 was evaluated using two protocols:

| Benchmark | Accuracy |
|---|---:|
| Rolling continuous complete next-word | 35.5% normalized accuracy |
| Standalone next-word top-1 | 43.2% top-1 accuracy |

### Gemma CPT checkpoint summary

The selected-token checkpoint analysis showed that continued pretraining substantially improved Gemma's legal-text prediction ability.

| Model | Top-1 | Top-5 | Top-10 | Avg NLL | Perplexity |
|---|---:|---:|---:|---:|---:|
| Base Gemma | 31.4% | 50.7% | 61.0% | 6.321 | 556.01 |
| Final CPT checkpoint | ~62.0% | ~80.3% | ~86.1% | ~3.15 | ~23.3 |

This means the CPT checkpoints nearly doubled the base model's top-1 next-token accuracy and strongly reduced perplexity.

The detailed evaluation documentation is located at:

```text
evaluation/evaluation.md
evaluation/next_token_analysis/README_selected_token.md
```

Large evaluation execution artifacts are excluded from GitHub when necessary, including:

- full prediction JSON/JSONL files;
- large dashboards if they exceed GitHub limits;
- S3 checkpoint downloads;
- model checkpoints;
- LoRA adapters;
- runtime logs.

---

## Reproducibility notes

To reproduce the preprocessing logic:

1. Place the raw source files under:

```text
data_preprocessing/data_raw/
```

2. Run source inspection scripts from:

```text
data_preprocessing/data_preprocessing_scripts/01_data_inspection/
```

3. Run source-specific processing scripts from:

```text
data_preprocessing/data_preprocessing_scripts/02_data_processing/
```

4. Run quality checks from:

```text
data_preprocessing/data_preprocessing_scripts/03_quality_checks/
```

5. Build the full dataset with:

```text
data_preprocessing/data_preprocessing_scripts/04_full_dataset_building/
```

6. Run tokenization, header cleaning, and packing from:

```text
data_preprocessing/data_preprocessing_scripts/05_tokenization_4096/
data_preprocessing/data_preprocessing_scripts/06_cleaning_and_packing/
```

Training and full evaluation should be executed on the EC2 GPU environment, not from a local CPU environment.

---

## Future work

The next phase is maintained in the EC2 training/evaluation environment or a separate training repository.

Planned and ongoing work includes:

- full CPT training using the excluded packed training dataset;
- checkpoint saving and storage;
- LoRA adapter management;
- selected-token evaluation;
- complete next-word evaluation;
- rolling complete next-word evaluation;
- comparison against external baselines such as Claude Sonnet;
- generation of prediction files, logs, dashboards, and final analysis reports;
- later SFT/RAG-based legal reasoning development.

This repository keeps the preprocessing logic, dataset documentation, smoke-test preparation, and evaluation conclusions. The executable full training/evaluation workflow and large generated artifacts remain outside GitHub or are stored separately.

---

## Important notes

- The full packed training file is intentionally excluded from GitHub.
- Raw legal documents are intentionally excluded from GitHub.
- Model checkpoints and LoRA adapters are intentionally excluded from GitHub.
- The tokenizer and base model are not stored in this repository.
- The tokenizer and model are loaded from Hugging Face on the EC2 environment.
- Training and full evaluation require a GPU environment.
- The current repository is mainly for preprocessing, reproducibility, documentation, smoke testing, and evaluation analysis.
