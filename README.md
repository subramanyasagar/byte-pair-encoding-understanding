# Byte Pair Encoding (BPE) — learning demo

This repository is a small **educational** Python example of **Byte Pair Encoding**: learning merge rules from text (**training**), then applying those rules to turn new text into tokens and numeric IDs (**inference**).

## What problem does BPE solve?

Language models do not read raw text directly. They need **tokens**—chunks of text represented as numbers. If you only used single letters, sequences would be very long and would miss common patterns like `th` or `ing`. If you only used huge fixed dictionaries, you would need enormous vocabularies.

**BPE** offers a middle path: start from something simple (here, **one character = one token**), then **repeatedly merge** the most common **pairs** of neighboring tokens into new, longer tokens. Over many merges, frequent words and subwords become single tokens, which makes sequences shorter and more meaningful.

## What this code does

**Training** ([`byte_pair_encoding_training.py`](byte_pair_encoding_training.py)) walks through that idea on a short sample string (`TRAINING_TEXT`):

1. Split the text into characters and record them in a growing **vocabulary** (string → integer ID).
2. Count how often each adjacent pair appears (e.g. `lo`, `ow`, …).
3. Pick the pair that appears often enough and **merge** every occurrence of that pair into one token.
4. Repeat for a fixed number of rounds. The result includes **`merge_order`** (which merges happened first) and **`vocab`**.

**Inference** ([`byte_pair_encoding_inference.py`](byte_pair_encoding_inference.py)) takes **new** text (`INFERENCE_TEXT`) plus the **trained** `merge_order` and `vocab`. It starts from characters and repeatedly applies merges allowed by `merge_order` until no more applicable merges remain, then maps each token to its ID using `vocab`.

Shared helpers live in [`byte_pair_encoding_common.py`](byte_pair_encoding_common.py) (character-level tokenization, merging one pair, counting adjacent pairs).

**End-to-end demo:** [`byte_pair_encoding_pipeline.py`](byte_pair_encoding_pipeline.py) runs training first, then inference on the learned artifacts and prints both stages.

Real systems used in AI (e.g. GPT-style tokenizers) use the same core idea at scale on huge corpora, with extra engineering for efficiency, special tokens, and Unicode. This repo is **not** a production tokenizer; it is a **compact illustration** of training, merge order, vocabulary, and encoding new text.

## Project layout

| File | Role |
|------|------|
| `byte_pair_encoding_common.py` | Shared utilities (token list helpers, pair counts, merge step) |
| `byte_pair_encoding_training.py` | `train(...)` → `merge_order`, `vocab`, training token list |
| `byte_pair_encoding_inference.py` | `infer(text, merge_order, vocab)` → tokens and ID list |
| `byte_pair_encoding_pipeline.py` | Calls `train` then `infer`; main demo entry point |

## How to run it

Requires Python 3. From this directory:

**Full pipeline (recommended):**

```bash
python3 byte_pair_encoding_pipeline.py
```

**Training only** (vocabulary, merge precedence, last training token list):

```bash
python3 byte_pair_encoding_training.py
```

**Inference only** (trains a fresh model on `TRAINING_TEXT`, then encodes `INFERENCE_TEXT`):

```bash
python3 byte_pair_encoding_inference.py
```

You can import `train`, `infer`, `TRAINING_TEXT`, and `INFERENCE_TEXT` from the modules above and call them from your own code if you want different strings or to chain steps manually.
