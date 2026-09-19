# Evaluation Results

## Overview

This document summarizes the evaluation performed on **Claude Sonnet 4.5** as an external baseline and the checkpoint analysis conducted for the **Gemma CPT** model.

Two evaluation protocols were used for Claude Sonnet 4.5:

1. **Rolling Continuous Complete Next-Word Evaluation**
2. **Standalone Next-Word Top-1 Evaluation**

In addition, every checkpoint produced during the first epoch of Gemma CPT training was evaluated to analyze the evolution of the model throughout continued pretraining.

---

# 1. Claude Sonnet 4.5 Rolling Continuous Next-Word Evaluation

## Model Information

| Property           | Value                                          |
| ------------------ | ---------------------------------------------- |
| Model              | Claude Sonnet 4.5                              |
| Model ID           | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| AWS Region         | `eu-west-1`                                    |
| Benchmark          | Rolling Continuous Complete Next-Word          |
| Number of Examples | **1000**                                       |

---

## Evaluation Methodology

The rolling benchmark follows a **teacher-forced next-word prediction** protocol.

For every evaluation example:

* the model receives the complete ground-truth context preceding the prediction point;
* it is instructed to generate **only the immediate next complete word**;
* each prediction is evaluated independently;
* previous model predictions are **never** reused as input.

Consequently, this benchmark measures the model's ability to predict the next word under teacher-forced conditions without allowing prediction errors to accumulate over time.

---

## Overall Results

| Metric                         |     Value |
| ------------------------------ | --------: |
| Number of Examples             |      1000 |
| Strict Correct Predictions     |       354 |
| Strict Accuracy                | **35.4%** |
| Normalized Correct Predictions |       355 |
| Normalized Accuracy            | **35.5%** |
| Macro Passage Accuracy         | **35.5%** |

Only one prediction differed between strict and normalized scoring, indicating that normalization (primarily Arabic diacritics and formatting differences) had a negligible effect on the final accuracy.

---

## Accuracy by Passage

| Passage    | Accuracy |
| ---------- | -------: |
| Passage 1  |      34% |
| Passage 2  |      40% |
| Passage 3  |      38% |
| Passage 4  |      50% |
| Passage 5  |      22% |
| Passage 6  |      22% |
| Passage 7  |      34% |
| Passage 8  |      36% |
| Passage 9  |      30% |
| Passage 10 |  **56%** |
| Passage 11 |      22% |
| Passage 12 |      46% |
| Passage 13 |      30% |
| Passage 14 |      54% |
| Passage 15 |      22% |
| Passage 16 |      28% |
| Passage 17 |      34% |
| Passage 18 |      34% |
| Passage 19 |      30% |
| Passage 20 |      48% |

### Observations

* Highest passage accuracy: **56%** (Passage 10)
* Lowest passage accuracy: **22%** (Passages 5, 6, 11 and 15)

The noticeable variation across passages suggests that prediction difficulty depends strongly on the linguistic characteristics and complexity of the underlying legal text.

---

## Accuracy by Source

| Source               | Examples | Correct | Accuracy |
| -------------------- | -------: | ------: | -------: |
| Legislation          |      200 |      88 |  **44%** |
| Official Gazette     |      200 |      76 |      38% |
| ADL Rulings          |      200 |      70 |      35% |
| Related Provisions   |      150 |      51 |      34% |
| Associated Study     |      100 |      34 |      34% |
| Bibliographic Ruling |      150 |      36 |  **24%** |

### Observations

The model achieved its highest accuracy on **Legislation** documents (44%), while **Bibliographic Ruling** documents proved to be the most challenging, with an accuracy of only 24%.

---

# 2. Claude Sonnet 4.5 Standalone Next-Word Evaluation

## Evaluation

A second benchmark evaluated Claude Sonnet 4.5 using isolated next-word prediction examples.

| Metric              |     Value |
| ------------------- | --------: |
| Number of Examples  |      1000 |
| Correct Predictions |       432 |
| Top-1 Accuracy      | **43.2%** |

---

## Comparison of the Two Benchmarks

| Evaluation                            |  Accuracy |
| ------------------------------------- | --------: |
| Rolling Continuous Complete Next-Word | **35.5%** |
| Standalone Next-Word Top-1            | **43.2%** |

The standalone benchmark achieves a higher accuracy than the rolling benchmark because the two protocols evaluate different prediction scenarios.

The rolling benchmark evaluates prediction within continuous passages using teacher forcing, whereas the standalone benchmark evaluates isolated prediction points. Consequently, the rolling benchmark represents a more challenging evaluation setting for exact next-word prediction.

---

# 3. Gemma CPT Checkpoint Analysis

## Overview

To analyze the effect of Continued Pretraining (CPT), every checkpoint produced during the **first training epoch** was evaluated on the same held-out next-token benchmark.

Each checkpoint was downloaded from Amazon S3 and evaluated independently using the selected-token evaluation pipeline.

Every checkpoint was evaluated on **1,000 held-out examples** using the following metrics:

* Top-1 Accuracy
* Top-5 Accuracy
* Top-10 Accuracy
* Average Probability Assigned to the Correct Token
* Negative Log-Likelihood (NLL)
* Perplexity of the Correct Token
* Average Rank of the Correct Token

---

## Interactive Visualization

A complete interactive HTML dashboard was generated to facilitate checkpoint comparison.

### Location

```text
evaluation/
└── next_token_analysis/
    └── readable_reports_html/
        └── index.html
```

The dashboard provides interactive visualizations of:

* Top-1 Accuracy
* Top-5 Accuracy
* Top-10 Accuracy
* Average Probability Assigned to the Correct Token
* Negative Log-Likelihood (NLL)
* Perplexity
* Average Rank of the Correct Token

The interactive report provides a considerably clearer visualization of the learning dynamics than inspecting raw JSON files individually.

---

# Checkpoint Results

| Model           |     Top-1 | Top-5 | Top-10 | Avg. Target Probability |  Avg. NLL | Perplexity | Avg. Target Rank |
| --------------- | --------: | ----: | -----: | ----------------------: | --------: | ---------: | ---------------: |
| Base            |     31.4% | 50.7% |  61.0% |                   0.293 |     6.321 |     556.01 |           144.52 |
| Checkpoint-100  |     48.2% | 70.6% |  77.2% |                   0.463 |     4.402 |      81.61 |            22.13 |
| Checkpoint-200  |     52.8% | 73.6% |  80.7% |                   0.517 |     3.919 |      50.33 |            19.00 |
| Checkpoint-1000 |     58.4% | 77.5% |  83.6% |                   0.571 |     3.602 |      36.65 |            15.29 |
| Checkpoint-1500 |     58.5% | 78.1% |  84.1% |                   0.580 |     3.507 |      33.36 |            13.46 |
| Checkpoint-2000 |     59.6% | 79.3% |  85.4% |                   0.587 |     3.460 |      31.81 |            12.59 |
| Checkpoint-2500 |     60.6% | 80.0% |  86.1% |                   0.598 |     3.234 |      25.39 |            11.48 |
| Checkpoint-2600 |     60.7% | 80.1% |  86.2% |                   0.597 |     3.197 |      24.46 |            11.14 |
| Checkpoint-2700 |     60.9% | 79.6% |  85.6% |                   0.600 |     3.311 |      27.42 |            11.59 |
| Checkpoint-2800 |     61.5% | 80.5% |  86.2% |                   0.605 |     3.213 |      24.85 |            11.57 |
| Checkpoint-2900 |     61.0% | 80.9% |  85.7% |                   0.604 |     3.237 |      25.47 |            11.27 |
| Checkpoint-3000 |     61.0% | 79.9% |  86.3% |                   0.604 |     3.165 |      23.69 |            11.42 |
| Checkpoint-3100 |     61.4% | 80.2% |  86.2% |                   0.607 |     3.181 |      24.07 |            11.43 |
| Checkpoint-3200 |     61.2% | 79.9% |  86.2% |                   0.605 |     3.188 |      24.23 |            11.37 |
| Checkpoint-3300 |     61.6% | 80.1% |  86.1% |                   0.605 |     3.156 |      23.47 |            11.29 |
| Checkpoint-3400 |     61.9% | 80.1% |  86.1% |                   0.607 | **3.147** |  **23.28** |            11.14 |
| Checkpoint-3500 |     61.6% | 80.1% |  86.1% |                   0.606 |     3.150 |      23.34 |        **11.10** |
| Checkpoint-3600 |     62.0% | 80.0% |  86.1% |                   0.607 |     3.152 |      23.37 |            11.25 |
| Checkpoint-3700 | **62.1%** | 80.3% |  85.8% |                   0.607 |     3.148 |      23.28 |            11.27 |
| Checkpoint-3800 |     61.9% | 80.3% |  86.0% |                   0.607 |     3.154 |      23.44 |            11.27 |
| Checkpoint-3894 |     62.0% | 80.3% |  86.1% |                   0.607 |     3.150 |      23.34 |            11.28 |

---

# Learning Dynamics

The checkpoint evaluation demonstrates a clear and consistent improvement throughout continued pretraining.

## Top-1 Accuracy

Top-1 accuracy increased from **31.4%** for the base model to **62.0–62.1%** after continued pretraining.

* **Absolute improvement:** +30.7 percentage points
* **Relative improvement:** approximately **98%**

This nearly doubles the model's exact next-token prediction accuracy.

---

## Top-5 and Top-10 Accuracy

Prediction quality improved consistently across larger candidate sets.

| Metric          |  Base | Final |
| --------------- | ----: | ----: |
| Top-5 Accuracy  | 50.7% |  ≈80% |
| Top-10 Accuracy | 61.0% |  ≈86% |

These improvements indicate that the correct token consistently moved closer to the top of the model's prediction distribution.

---

## Average Probability Assigned to the Correct Token

The average probability assigned to the ground-truth token increased from

**0.293 → 0.607**

representing more than a twofold increase in model confidence.

---

## Negative Log-Likelihood (NLL)

Average NLL decreased from

**6.321 → approximately 3.15**

representing roughly a **50% reduction**.

Lower NLL indicates that the model assigns substantially higher likelihood to the correct continuation after continued pretraining.

---

## Perplexity

Perplexity decreased dramatically from

**556.0 → approximately 23.3**

representing approximately a **96% reduction**, demonstrating a substantial improvement in predictive certainty.

---

## Average Target Rank

The average rank of the correct token improved from

**144.5 → approximately 11**

This indicates that, on average, the correct token moved from being well outside the model's most probable predictions to consistently appearing among its highest-ranked candidates.

---

# Training Behaviour

The largest performance improvements occurred during the early stages of continued pretraining.

Rapid gains were observed between the base model and approximately **checkpoint 2000** across all evaluation metrics.

Beyond **checkpoint 2500**, the metrics largely stabilized, with only minor fluctuations between checkpoints. The final checkpoints exhibited very similar performance, suggesting that the model had effectively converged by the end of the first training epoch.

No evidence of performance degradation was observed during the evaluated checkpoints.

---

# 4. Conclusion

The evaluation demonstrates that continued pretraining substantially improves Gemma's ability to model Arabic legal text.

Across the evaluated checkpoints, continued pretraining consistently increased Top-1, Top-5, and Top-10 accuracy while simultaneously reducing Negative Log-Likelihood, perplexity, and the average rank of the correct token. The convergence observed during the latter part of the first epoch indicates that the model had largely reached a stable performance plateau.

Claude Sonnet 4.5 provides a strong external baseline for comparison, achieving **35.5% normalized accuracy** on the rolling continuous next-word benchmark and **43.2% Top-1 accuracy** on the standalone next-word benchmark. Together, these evaluations provide complementary perspectives on model performance, with the rolling benchmark assessing teacher-forced prediction in continuous legal text and the standalone benchmark measuring isolated next-word prediction accuracy.
