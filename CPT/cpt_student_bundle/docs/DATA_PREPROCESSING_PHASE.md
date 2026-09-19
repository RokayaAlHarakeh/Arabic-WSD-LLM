# Data Preprocessing Phase — Lebanese Legal CPT Dataset

## 1. Purpose of this phase

The objective of the data preprocessing phase is to transform several heterogeneous Lebanese legal text sources into a clean, balanced, tokenized, and packed dataset suitable for **Continued Pre-Training (CPT)** of a base language model.

The target model for the current CPT preparation is:

```text
google/gemma-4-12B
```

The preprocessing phase does **not** create supervised instruction-answer examples. Its role is to prepare continuous legal text so the model can learn Lebanese legal language, structure, terminology, and writing style before any later SFT or RAG-based reasoning stage.

The final output of this phase is a set of 4096-token packed JSONL files:

```text
data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/val_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/test_cleanheaders_packed_4096.jsonl
```

## 1.1 Repository and storage note

The preprocessing work is organized under one project folder:

```text
data_preprocessing/
```

This folder contains the preprocessing scripts, raw data folders, progressively generated CPT files, samples, archives, and the preprocessing documentation.

The full dataset is **not fully pushed to GitHub** because the raw, intermediate, and final JSONL files are large and can exceed normal GitHub storage limits. The GitHub repository should therefore mainly track:

```text
source code
preprocessing scripts
documentation
small samples
quality reports
configuration notes
.gitignore rules
```

Large files should be kept locally, on the EC2 instance, or in external storage such as S3 or another project storage location. This avoids making the Git repository too large and keeps the preprocessing repository focused on reproducibility rather than raw data hosting.

---

## 2. Preprocessing folder organization

The whole preprocessing phase is organized under:

```text
data_preprocessing/
```

Current high-level structure:

```text
data_preprocessing/
├── data_raw/
│   ├── adl/
│   ├── bibliographic_rulings/
│   ├── gazette/
│   ├── legislations/
│   ├── provisions/
│   └── studies/
├── data_cpt/
│   ├── combined_data/
│   ├── final_cpt_v1/
│   ├── samples/
│   ├── source_cpt_files/
│   └── split_4096_archive/
├── data_preprocessing_scripts/
│   ├── 01_data_inspection/
│   ├── 02_data_processing/
│   ├── 03_quality_checks/
│   ├── 04_full_dataset_building/
│   ├── 05_tokenization_4096/
│   ├── 06_cleaning_and_packing/
│   └── 99_utilities_archive/
└── DATA_PREPROCESSING_PHASE.md
```

This structure keeps the preprocessing phase self-contained. Raw sources, progressively generated CPT files, samples, archived intermediate versions, scripts, and documentation are all grouped in the same `data_preprocessing/` folder.

Each scripts folder represents one logical stage in the data preparation pipeline.

All paths in this document are written relative to the `data_preprocessing/` folder unless otherwise stated. For example:

```text
data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
```

means:

```text
data_preprocessing/data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
```

---

# 3. Stage 1 — Data inspection

Folder:

```text
data_preprocessing_scripts/01_data_inspection/
```

Scripts:

```text
01_inspect_adl.py
02_inspect_gazette.py
03_inspect_provisions.py
04_inspect_bibliographic_rulings.py
05_inspect_legislations.py
06_inspect_studies.py
```

## Purpose

This stage inspects the raw data sources before any transformation. The goal is to understand:

- file format
- number of records
- available metadata fields
- text length distribution
- missing fields
- duplicate risks
- source-specific problems
- whether the source is useful for CPT

## Sources inspected

The main sources inspected were:

```text
ADL rulings
Official Gazette
Related provisions
Bibliographic rulings
Lebanese legislations
Associated legal studies
```

## What this stage answers

For each source, the inspection scripts help answer:

```text
Is the file valid JSONL?
How many records are available?
Which fields exist?
Where is the main legal text stored?
Are there empty or very short records?
Are there unusually long records?
Are metadata fields consistent?
Should this source be included in CPT?
```

## Expected outputs

This stage usually produces console summaries or reports that guide the next processing step. It does not produce the final training dataset directly.

---

# 4. Stage 2 — Source-specific data processing

Folder:

```text
data_preprocessing_scripts/02_data_processing/
```

Scripts:

```text
01_chunk_adl.py
02_chunk_gazette.py
03_process_provisions.py
04_process_bibliographic_rulings.py
05_process_gazette_section2.py
06_prepare_legislations.py
07_prepare_studies.py
```

## Purpose

This stage converts raw legal sources into CPT-ready source files. Each source has its own structure and therefore needs its own processing logic.

The output of this stage is not yet the final train/validation/test dataset. Instead, it produces cleaned or chunked source-level JSONL files that can later be combined.

## Approximate chunking configuration

Most long-text sources were chunked using approximately the same character-based chunking policy before the later tokenizer-based 4096-token preparation.

The common configuration was:

```python
MAX_CHARS = 20000
OVERLAP_CHARS = 1200
MIN_CHARS = 300
MIN_CHUNK_CHARS = 3000
```

Meaning:

```text
MAX_CHARS = 20000
Maximum approximate size of a source-level text chunk before tokenization.

OVERLAP_CHARS = 1200
Repeated character overlap between consecutive chunks from the same long document. This helps preserve continuity across chunk boundaries.

MIN_CHARS = 300
Very short records below this threshold were usually skipped because they are often not useful for CPT.

MIN_CHUNK_CHARS = 3000
When splitting long documents, very small leftover chunks were avoided or merged when possible so that the resulting chunks remain meaningful.
```

This chunking stage is different from the final 4096-token packing stage. Character chunking prepares readable legal text chunks at the source level. Token packing later converts the cleaned train/validation/test text into dense 4096-token training blocks for Gemma.

---

## 4.1 ADL rulings

Script:

```text
01_chunk_adl.py
```

Raw source:

```text
data_raw/adl/all_rulings.jsonl
```

Purpose:

- process judicial rulings from Majallat Al-Adl
- clean ruling text
- preserve useful legal metadata
- split long rulings into manageable CPT chunks
- avoid empty or invalid records

The ADL rulings were kept because they contain valuable Lebanese judicial language and court decision style.

---

## 4.2 Official Gazette legal core

Script:

```text
02_chunk_gazette.py
```

Purpose:

- process Official Gazette documents
- keep legally useful sections
- remove or reduce irrelevant administrative noise
- chunk long legal texts
- preserve useful metadata such as source, type, year, authority, and section when useful

Processing strategy used for the Gazette:

```text
Section 1 and unknown legal sections: kept fully
Section 2: handled separately as a sample
Very short records: skipped
Very long records: fixed/chunked
```

The Gazette legal core is one of the largest and most important parts of the CPT dataset because it contains legislation, decrees, decisions, and other formal Lebanese legal texts.

---

## 4.3 Related provisions

Script:

```text
03_process_provisions.py
```

Purpose:

- process related legal provisions
- clean text
- preserve legal references
- prepare provision-level text for CPT

This source helps the model learn statutory phrasing and cross-reference language.

---

## 4.4 Bibliographic rulings

Script:

```text
04_process_bibliographic_rulings.py
```

Purpose:

- process bibliographic court ruling records
- clean ruling descriptions and legal summaries
- preserve source and legal metadata
- prepare the records for inclusion in CPT

This source contributes a large number of legal ruling-style records.

---

## 4.5 Gazette Section 2 sample

Script:

```text
05_process_gazette_section2.py
```

Purpose:

- process a sampled subset of Official Gazette Section 2
- avoid overloading the CPT dataset with lower-priority or repetitive Gazette material
- preserve useful Gazette-style legal language while controlling dataset balance

The Section 2 Gazette data was sampled rather than fully included to avoid making the model over-learn less central administrative publication patterns.

---

## 4.6 Legislations

Script:

```text
06_prepare_legislations.py
```

Purpose:

- process Lebanese legislation records
- clean legal text
- preserve legislative metadata
- prepare high-priority statutory material for CPT

The legislation source is a high-priority source because it teaches the model formal Lebanese legislative language, articles, legal structure, and statutory terminology.

---

## 4.7 Associated legal studies

Script:

```text
07_prepare_studies.py
```

Purpose:

- process associated legal studies
- keep Arabic legal studies relevant to the CPT objective
- exclude unsuitable or non-Arabic material when needed
- prepare academic/legal analysis text for CPT

This source is much smaller than the others but useful because it contributes legal commentary and analytical writing style.

---

# 5. Stage 3 — Quality checks

Folder:

```text
data_preprocessing_scripts/03_quality_checks/
```

Scripts:

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

## Purpose

This stage validates the processed source files before they are combined into the final dataset.

Quality checks are important because a CPT dataset can silently become corrupted if there are:

- invalid JSON lines
- empty records
- duplicate records
- badly chunked records
- extremely long records
- missing text fields
- broken metadata
- tokenizer problems
- records exceeding the target context length

---

## 5.1 Source-level checks

The first seven scripts check each processed source separately:

```text
ADL rulings
Gazette legal core
Related provisions
Bibliographic rulings
Gazette Section 2 sample
Legislations
Associated studies
```

These checks confirm that each source is clean enough to be included in the full CPT dataset.

Typical checks include:

```text
invalid JSON count
empty text count
duplicate count
record count
token or character length distribution
source field consistency
sample examples
maximum length
```

---

## 5.2 Gemma tokenizer check

Script:

```text
08_check_gemma_tokenizer.py
```

Purpose:

- verify that the target tokenizer loads correctly
- confirm tokenization behavior for Arabic and Lebanese legal text
- check token counts before creating 4096-token examples

Target tokenizer:

```text
google/gemma-4-12B
```

This is important because the dataset must be prepared according to the same tokenizer that will be used during training.

---

## 5.3 4096-token dataset check

Script:

```text
09_check_4096_token_dataset.py
```

Purpose:

- verify that records are compatible with the 4096-token training context
- check that no record exceeds the model context after tokenization
- validate final JSONL structure before packing

---

# 6. Stage 4 — Full dataset building

Folder:

```text
data_preprocessing_scripts/04_full_dataset_building/
```

Scripts:

```text
01_combine_cpt_full_legal_v1.py
02_fix_long_records.py
03_split_train_val_test.py
04_create_sample_dataset.py
```

## Purpose

This stage combines all approved processed sources into one full CPT dataset, fixes problematic long records, creates train/validation/test splits, and creates smaller sample datasets for testing.

---

## 6.1 Combining all CPT sources

Script:

```text
01_combine_cpt_full_legal_v1.py
```

Purpose:

- combine all processed source-level CPT files
- preserve source metadata
- produce a unified full legal CPT dataset

The combined dataset includes:

```text
legislations
bibliographic_rulings
gazette_legal_core
gazette_section2_sample
related_provisions
adl_rulings
associated_studies_ar
```

---

## 6.2 Fixing long records

Script:

```text
02_fix_long_records.py
```

Purpose:

- detect records that are too long for 4096-token training
- split or fix long records safely
- avoid losing legal content
- prevent training errors caused by overlength examples

This step was especially important for Gazette data, where some legal records were very long.

---

## 6.3 Train/validation/test split

Script:

```text
03_split_train_val_test.py
```

Purpose:

- split the final 4096-ready dataset into train, validation, and test files
- preserve source representation across splits
- avoid using only one source in validation or test

The split used:

```text
Train: 97%
Validation: 2%
Test: 1%
```

Final split files before header cleaning:

```text
data_cpt/train_cpt_full_legal_v1_4096.jsonl
data_cpt/val_cpt_full_legal_v1_4096.jsonl
data_cpt/test_cpt_full_legal_v1_4096.jsonl
```

After later cleaning, the active files became:

```text
data_cpt/final_cpt_v1/clean/train_cleanheaders.jsonl
data_cpt/final_cpt_v1/clean/val_cleanheaders.jsonl
data_cpt/final_cpt_v1/clean/test_cleanheaders.jsonl
```

---

## 6.4 Sample dataset creation

Script:

```text
04_create_sample_dataset.py
```

Purpose:

- create small sample datasets for testing scripts
- avoid testing every script on the full dataset
- allow quick validation of tokenization, packing, and training scripts

Samples are useful for:

```text
checking tokenizer behavior
checking packing logic
running short training tests
checking EC2/Unsloth/QLoRA setup
```

---

# 7. Stage 5 — Tokenization to 4096-token records

Folder:

```text
data_preprocessing_scripts/05_tokenization_4096/
```

Script:

```text
02_create_4096_token_dataset.py
```

## Purpose

This stage creates a dataset compatible with the 4096-token context length of the target model.

The target maximum sequence length is:

```text
4096 tokens
```

The tokenizer used is:

```text
google/gemma-4-12B
```

## Why 4096 tokens?

4096 tokens was selected to:

- fit the model context length used for CPT
- keep long Lebanese legal documents meaningful
- reduce unnecessary truncation
- allow the model to see longer legal structures, articles, decrees, and rulings

## Final 4096-ready full dataset

The full 4096-ready dataset contained:

```text
Total records: 203,073
Total tokens: 141,115,308
Maximum length: 4096
Invalid JSON: 0
Empty text: 0
Duplicates: 0
```

Source distribution:

```text
legislations:              62,504 records / 51,525,352 tokens
bibliographic_rulings:     55,325 records / 10,755,042 tokens
gazette_legal_core:        53,849 records / 52,718,098 tokens
gazette_section2_sample:   21,326 records / 9,670,875 tokens
related_provisions:         4,949 records / 2,590,556 tokens
adl_rulings:                4,922 records / 13,186,557 tokens
associated_studies_ar:        198 records / 668,828 tokens
```

---

# 8. Stage 6 — Header cleaning and dataset packing

Folder:

```text
data_preprocessing_scripts/06_cleaning_and_packing/
```

Scripts:

```text
01_clean_text_headers.py
02_pack_dataset_4096.py
```

This is the final preprocessing stage before training.

---

## 8.1 Cleaning text headers

Script:

```text
01_clean_text_headers.py
```

## Purpose

Earlier source files contained metadata headers inside the text, such as source, type, year, authority, issue number, part number, chunk number, and other source-specific fields.

Because CPT trains only on the `text` content, metadata inside the text is learned by the model. Therefore, the headers needed to be cleaned so that the model learns useful legal context but not noisy internal processing details.

## Header cleaning strategy

Useful metadata kept inside the model text:

```text
source
type
authority
year
journal_section, when useful for Gazette records
legal_domain/domain, when available
```

Noisy processing metadata removed from the model text:

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
internal processing identifiers
```

## Why keep some metadata?

Compact metadata helps the model distinguish between different legal text types:

```text
legislation
court ruling
Official Gazette document
legal provision
legal study
```

This is useful because each type has a different style and legal function.

## Why remove noisy metadata?

Noisy metadata can make the model learn irrelevant patterns, such as internal file names, chunk numbers, or processing IDs. These are not legal knowledge and should not influence the trained model.

## Cleaned files

Final cleaned files:

```text
data_cpt/final_cpt_v1/clean/train_cleanheaders.jsonl
data_cpt/final_cpt_v1/clean/val_cleanheaders.jsonl
data_cpt/final_cpt_v1/clean/test_cleanheaders.jsonl
```

Cleaning reports:

```text
data_cpt/final_cpt_v1/reports/01_train_cleanheaders.txt
data_cpt/final_cpt_v1/reports/02_val_cleanheaders.txt
data_cpt/final_cpt_v1/reports/03_test_cleanheaders.txt
```

## Cleaning result summary

Train:

```text
Records: 196,983
Invalid JSON: 0
Empty text: 0
Changed records: all records
```

Validation:

```text
Records: 4,061
Invalid JSON: 0
Empty text: 0
Changed records: all records
```

Test:

```text
Records: 2,029
Invalid JSON: 0
Empty text: 0
Changed records: all records
```

---

## 8.2 Packing the dataset

Script:

```text
02_pack_dataset_4096.py
```

## Purpose

Packing combines many shorter tokenized examples into full 4096-token blocks.

Before packing, each row is one cleaned legal record or chunk.

After packing, each row is one dense 4096-token training block.

This greatly improves training efficiency.

---

## Why packing is important

Without packing, many rows are shorter than 4096 tokens. This causes inefficient training because the GPU processes many short examples separately.

Packing creates full blocks like:

```text
Packed row 1: 4096 tokens
Packed row 2: 4096 tokens
Packed row 3: 4096 tokens
...
```

This reduces the number of training rows and optimizer steps while keeping almost all useful tokens.

---

## Train packing report

Final train packing report:

```text
Model: google/gemma-4-12B
Input file: data_cpt/final_cpt_v1/clean/train_cleanheaders.jsonl
Output file: data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
Max sequence length: 4096
Append EOS: True
Drop remainder: True

Input records: 196,983
Invalid JSON: 0
Empty text skipped: 0
Input tokens including EOS: 127,594,907
Output blocks: 31,151
Output useful tokens: 127,594,496
Remainder tokens: 411
Dropped tokens: 411
Average tokens per packed block: 4096.0
Approx optimizer steps with gradient_accumulation_steps=8: 3,893.88
```

## Interpretation

Before packing:

```text
196,983 rows
```

After packing:

```text
31,151 rows
```

Rows reduced by:

```text
196,983 - 31,151 = 165,832 fewer rows
```

Average compression in row count:

```text
196,983 / 31,151 ≈ 6.32x
```

This means each packed row contains, on average, the token content of about 6.3 original cleaned rows.

---

## Training efficiency impact

Without packing:

```text
196,983 rows / gradient_accumulation_steps 8 ≈ 24,623 optimizer steps
```

With packing:

```text
31,151 packed rows / gradient_accumulation_steps 8 ≈ 3,894 optimizer steps
```

Packing reduced optimizer steps by approximately:

```text
24,623 / 3,894 ≈ 6.3x
```

Assuming 3 minutes per optimizer step:

```text
Without packing: approximately 51 days
With packing:    approximately 8.1 days
```

This is why packing is essential for the CPT training phase.

---

# 9. Final active preprocessing outputs

The active final dataset for training is located in:

```text
data_cpt/final_cpt_v1/
```

## Clean text files

```text
data_cpt/final_cpt_v1/clean/train_cleanheaders.jsonl
data_cpt/final_cpt_v1/clean/val_cleanheaders.jsonl
data_cpt/final_cpt_v1/clean/test_cleanheaders.jsonl
```

## Packed files

```text
data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/val_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/test_cleanheaders_packed_4096.jsonl
```

## Reports

```text
data_cpt/final_cpt_v1/reports/01_train_cleanheaders.txt
data_cpt/final_cpt_v1/reports/02_val_cleanheaders.txt
data_cpt/final_cpt_v1/reports/03_test_cleanheaders.txt
data_cpt/final_cpt_v1/reports/04_train_packed_4096.txt
data_cpt/final_cpt_v1/reports/05_val_packed_4096.txt
data_cpt/final_cpt_v1/reports/06_test_packed_4096.txt
```

---

# 10. Final preprocessing dataset summary

The final CPT dataset is:

```text
Arabic/Lebanese legal text
source-aware
cleaned
validated
Gemma-tokenized
4096-context compatible
packed for efficient training
ready for QLoRA CPT
```

The final packed training file contains:

```text
31,151 full 4096-token blocks
127,594,496 useful training tokens
only 411 dropped tokens
approximately 3,894 optimizer steps per epoch with gradient_accumulation_steps=8
```

This is the dataset that should be used for the CPT training phase.

---

# 11. Recommended next phase

After preprocessing, the next phase is:

```text
QLoRA CPT training
```

The training scripts should use:

```text
data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/val_cleanheaders_packed_4096.jsonl
```

The test file should be kept for final evaluation:

```text
data_cpt/final_cpt_v1/packed/test_cleanheaders_packed_4096.jsonl
```

Before full training, a short EC2/Unsloth or standard PEFT test should confirm:

```text
model loading works
4-bit quantization works
LoRA attaches correctly
packed input_ids + attention_mask are accepted
loss is not NaN
adapter saves correctly
seconds per optimizer step are measured
```

---

# 12. Key design decisions

## 12.1 CPT before SFT/RAG

This dataset is for continued pre-training, not supervised question answering. The goal is to adapt the base model to Lebanese legal language before instruction tuning or RAG reasoning.

## 12.2 Use base model, not instruction model

The CPT target is the base model:

```text
google/gemma-4-12B
```

This is preferred because CPT is language adaptation, not instruction-response training.

## 12.3 Keep useful source headers

Useful compact headers help the model understand the type and origin of the text.

## 12.4 Remove noisy internal metadata

Internal identifiers, chunk numbers, issue numbers, and processing IDs were removed from the model text to avoid teaching the model irrelevant artifacts.

## 12.5 Pack to 4096 tokens

Packing is essential because it greatly reduces training steps and improves GPU efficiency.

## 12.6 Preserve validation and test sets

Validation and test files are kept separate to monitor training quality and avoid evaluating on training data.

---

# 13. Future work and downstream repositories

The preprocessing repository is limited to the data preparation phase. The next phases are expected to be maintained separately because they require GPU execution, model checkpoints, adapters, evaluation outputs, and experiment logs.

Future work includes:

```text
QLoRA CPT training on EC2
Unsloth or standard PEFT training tests
training speed benchmarking
checkpoint saving and adapter management
validation loss tracking
base vs CPT model evaluation
next-token, next-word, and rolling evaluation
final comparison reports
```

Training and evaluation are performed on the EC2 instance and can be stored in a separate training/evaluation repository or external project folder. This keeps the preprocessing repository clean and prevents large model artifacts, checkpoints, predictions, and logs from being mixed with the data preparation code.

The preprocessing output files are therefore the handoff point between repositories:

```text
data_cpt/final_cpt_v1/packed/train_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/val_cleanheaders_packed_4096.jsonl
data_cpt/final_cpt_v1/packed/test_cleanheaders_packed_4096.jsonl
```

These files are the inputs used by the EC2 training and evaluation phase.

---

# 14. One-line summary

The preprocessing phase transformed raw Lebanese legal sources into a clean, validated, source-aware, 4096-token packed CPT dataset ready for efficient QLoRA training of `google/gemma-4-12B`, while keeping large data and later EC2 training/evaluation artifacts outside the preprocessing GitHub repository.
