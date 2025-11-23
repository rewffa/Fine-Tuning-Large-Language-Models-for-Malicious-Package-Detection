# Fine-Tuning Large Language Models for Malicious Package Detection

> Fine-tune code/LLM models on behavior-sequence / code-based representations to detect malicious packages in open-source ecosystems (PyPI / npm / RubyGems).

---

## 1. Overview

Open-source package ecosystems (e.g., PyPI, npm, RubyGems) dramatically boost software productivity, but they also expand the attack surface. Adversaries frequently poison packages via typosquatting, dependency confusion, backdoor injection, and obfuscated/encrypted payloads. Traditional defenses based on handcrafted rules or shallow features struggle to keep up with rapidly evolving malicious behaviors.

This project provides an end-to-end pipeline to address the problem:

1. Extract semantic/behavioral sequences from package source code across multiple ecosystems.
2. Build labeled malicious/benign datasets and merge them into a unified format.
3. Fine-tune large language models (e.g., Llama-3-8B, GPT-OSS) on these sequences/texts for binary malicious package detection.

Goal: develop a detector that is transferable, explainable, and adaptable across ecosystems and languages.

---

## 2. Repository Structure

```
.
├── merged_dataset/                 # Merged cross-ecosystem train/test data
├── Fine_tuning_llama_3_8B.ipynb    # Example fine-tuning notebook for Llama-3-8B
├── train_gpt_oss_lora.py           # GPT-OSS fine-tuning script with LoRA
├── generate_python.py              # Behavioral sequence generation for PyPI
├── generate_npm.py                 # Behavioral sequence generation for npm
├── generate_rubygems.py            # Behavioral sequence generation for RubyGems
├── make_pypi_labels.py             # Label construction for PyPI
├── make_npm_labels.py              # Label construction for npm
└── make_rubygems_labels.py         # Label construction for RubyGems
```
