# LLM Capability Learning: Skill Acquisition, Systematic Generalisation, and Verbalisation Dynamics

[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An empirical study probing post-training adaptation dynamics in small causal language models (`Qwen2.5-0.5B-Instruct`). This project examines how parameter-efficient fine-tuning (LoRA) introduces deterministic algorithmic capabilities, measures distributional robustness under factorized out-of-distribution (OOD) shifts, monitors catastrophic forgetting, and evaluates whether operational capability implies declarative self-verbalization.

---

## 📌 Research Questions

1. **Skill Acquisition (RQ1):** Can a small instruction-tuned model acquire a non-natural, deterministic permutation function (`PAIRFLIP`) via low-rank parameter adaptation?
2. **Systematic Generalisation (RQ2):** Does the learned transformation generalize when subjected to independent shifts in numerical magnitude versus sequence length?
3. **Retention & Forgetting (RQ3):** Does adapting targeted projections cause interference or catastrophic forgetting in baseline natural language and code generation faculties?
4. **Declarative Verbalisation (RQ4):** Does operational mastery of an algorithmic rule enable the model to explain or introspect on the mechanics of that rule?

---

## 🔬 Experimental Setup

* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (~494M parameters)
* **Adaptation Mechanism:** Low-Rank Adaptation (LoRA) applied across all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
* **LoRA Hyperparameters:** Rank $r=16$, $\alpha=32$, Dropout $=0.05$ (8,798,208 trainable parameters $\approx$ 1.75% of model capacity)
* **Training Details:** 3 Epochs, AdamW ($\text{LR}=2\times 10^{-4}$), dynamic collation with assistant-only loss masking (`labels=-100` on prompt tokens), gradient clipping (`max_norm=1.0`)
* **Task Definition (`PAIRFLIP`):** Swap adjacent token pairs in a sequence while preserving trailing un-paired elements:
  $$\text{PAIRFLIP}([x_1, x_2, \dots, x_{2k}, x_{2k+1}]) = [x_2, x_1, \dots, x_{2k+1}, x_{2k}]$$

---

## 📊 Benchmark Results

### 1. In-Distribution & Controlled Factorized Evaluations

| Evaluation Regime | Condition Description | Sequence Length ($N$) | Value Range ($V$) | Exact Match (%) | Length Match (%) | Pair-Level Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Pre-trained)** | Zero-shot base model | 4–12 | 1–500 | 2.0% | — | — |
| **In-Distribution (LoRA)** | Identical to train split | 4–12 | 1–1,000 | **100.0%** | **100.0%** | **100.0%** |
| **Condition B (`short_large`)** | Value magnitude shift | 4–12 | 1,000–100,000 | **93.0%** | **98.0%** | **97.7%** |
| **Condition C (`long_small`)** | Context horizon expansion | 13–30 | 1–500 | **5.0%** | **12.0%** | **55.8%** |
| **Condition D (`long_large`)** | Joint magnitude & horizon shift | 13–30 | 1,000–100,000 | **0.0%** | **0.0%** | **41.2%** |

---

## 🔍 Core Findings

### 1. Asymmetric Generalisation Breakdown
The model exhibits near-perfect invariance to multi-token numerical magnitudes (**93.0% exact match** on numbers up to $10^5$), showing that tokenization complexity does not degrade the core operational mapping. However, extending sequence length beyond the training horizon ($N > 12$) induces an immediate collapse (**5.0%** and **0.0%** exact match).

### 2. Local Circuit Retention vs. Global Autoregressive Drift
While full-sequence exact matches fail on long sequences, **pair-level accuracy remains between 41.2% and 55.8%**. The internal attention circuits still execute the pairwise swap locally for early indices, but autoregressive generation accumulates positional errors, eventually degenerating into repetitive token loops.

### 3. Mechanistic vs. Declarative Decoupling
Probing capability verbalization (RQ4) revealed a stark dissociation:
* **Operational Execution:** 100.0% accuracy on in-distribution tasks.
* **Declarative Understanding:** When prompted to define or explain `PAIRFLIP`, the model hallucinates generic single-value swaps or arbitrary arithmetic operations.
Procedural competence acquired via low-rank weight updates does not induce semantic or introspective understanding of the learned behavior without explicit verbalization supervision.

### 4. Preservation of Pre-trained Priors
Evaluating 5 zero-shot baseline questions (machine learning concepts, recursive Python factorials, neural network definitions) before and after fine-tuning confirmed **zero catastrophic forgetting**. Restricting gradient updates to low-rank adapter matrices preserved general linguistic and coding competence.

---

## 📁 Repository Structure

```text
├── LLM_Capability_Learning.ipynb         # Full reproducible training & evaluation notebook
├── baseline_results.json                 # Pre-training baseline responses (RQ3 control)
├── pairflip_baseline_results.json        # Zero-shot task execution results (2% exact match)
├── pairflip_finetuned_results.json       # In-distribution LoRA test results (100% exact match)
├── controlled_generalisation_results.json # Factorized OOD evaluation metrics (Conditions B, C, D)
├── retention_results.json                # Post-training retention outputs (RQ3)
├── verbalisation_results.json            # Model explanations of the learned rule (RQ4)
└── README.md                             # Project overview and empirical summary
