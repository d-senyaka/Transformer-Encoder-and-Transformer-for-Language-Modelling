# Transformer Encoder & Transformer for Language Modelling (PyTorch)

A Python (PyTorch) implementation of core Transformer building blocks, applied to:
1) **A sequence labeling task (Letter Counting)** using a **Transformer Encoder-style model** (non-causal self-attention).
2) **Character-level Language Modeling** scaffolding (with a provided uniform baseline + a neural LM interface to implement).

> Repo language: **100% Python**  
> Repo: https://github.com/d-senyaka/Transformer-Encoder-and-Transformer-for-Language-Modelling

---

## Highlights (What’s implemented here)

### ✅ Part A — Transformer for **Letter Counting** (Sequence Labeling)
This repository includes a working Transformer-based classifier for the classic **letter-counting** task:

Given a fixed-length input string (length **20**) over a character vocabulary (**a–z + space = 27 tokens**), the model predicts **one label per position**:

- **0** → the current character has appeared **0** times previously
- **1** → the current character has appeared **1** time previously
- **2** → the current character has appeared **2+** times previously

Implemented components:
- **Character embedding** (`nn.Embedding`)
- **Learned positional embeddings** (`PositionalEncoding`)
- **Stackable Transformer layers** (self-attention + feed-forward + residuals + layer norm)
- **Per-position classifier head** (`Linear(d_model → 3)` + `log_softmax`)
- **Training loop** using **NLLLoss** and **Adam**
- **Attention map extraction** and saving attention heatmaps into `plots/`

Key code:
- `transformer.py` — main Transformer model + TransformerLayer + training routine
- `letter_counting.py` — driver that builds examples, trains, decodes, and plots attentions
- `utils.py` — `Indexer` utility for mapping tokens ↔ ids

---

### ✅ Part B — Language Modeling Framework (Baseline + Neural LM skeleton)
The repo also includes a language-modeling framework:
- A **UniformLanguageModel** baseline that assigns equal probability to every character
- A **NeuralLanguageModel** interface **(skeleton)** to be implemented
- Evaluation utilities for:
  - sanity checking probabilities
  - normalization testing
  - perplexity calculation

Key code:
- `transformer_lm.py` — LM interfaces (Uniform LM implemented; Neural LM + training left as TODO)
- `lm.py` — driver for LM evaluation (UNIFORM or NEURAL)

---

## Repository Structure

```text
.
├── transformer.py               # Transformer model (letter counting) + training + attention plotting support
├── letter_counting.py           # Task driver: builds dataset, trains, decodes, plots attention maps
├── transformer_lm.py            # Language model interfaces (uniform baseline + neural skeleton)
├── lm.py                        # LM driver: sanity checks + perplexity evaluation
├── utils.py                     # Indexer + Beam utilities
├── data/                        # datasets (text8 subsets, letter counting files, etc.)
├── plots/                       # generated attention heatmaps (saved by decode())
├── artifacts/                   # optional outputs / saved files (repo-specific)
├── Framework Code/              # provided framework reference code (mirrors key files)
└── Member 1 - Updated work/     # member-specific work area
```

---

## Quickstart

### 1) Environment
Install dependencies (typical):
- Python 3.9+
- `torch`
- `numpy`
- `matplotlib`

Example:
```bash
pip install torch numpy matplotlib
```

---

## Run: Letter Counting Transformer

The `letter_counting.py` script:
- reads training/dev examples
- creates labels
- trains the Transformer classifier
- prints predictions on a few dev examples
- plots attention maps into `plots/`
- reports accuracy on train/dev

### Command
```bash
python letter_counting.py --task BEFORE
```

Options:
- `--task BEFORE` (default): label counts based on **previous occurrences only**
- `--task BEFOREAFTER`: label counts based on **all other occurrences** in the sequence

Data files used by default:
- `data/lettercounting-train.txt`
- `data/lettercounting-dev.txt`

---

## Run: Language Model (Uniform baseline)

The LM driver supports:
- `--model UNIFORM` (works out-of-the-box)
- `--model NEURAL` (requires implementing `NeuralLanguageModel` + `train_lm` in `transformer_lm.py`)

### Uniform model
```bash
python lm.py --model UNIFORM
```

This prints:
- sanity check result
- normalization test result
- log probability, average log probability
- perplexity

---

## How the Transformer (Letter Counting) Works

At a high level:

1. **Embed characters**: `indices → [L, d_model]`
2. **Add positional embeddings**
3. **Apply N TransformerLayers**:
   - self-attention (single-head in this implementation)
   - dropout (if enabled)
   - residual + layer norm
   - feed-forward (2-layer MLP)
   - residual + layer norm
4. **Classify each position**: `d_model → 3` classes

The model returns:
- `log_probs`: shape `[L, 3]` (log-probabilities per position)
- `attn_maps`: list of attention matrices `[L, L]` (one per layer)

---

## Outputs

### Attention Heatmaps
If enabled in decoding (`do_plot_attn=True`), attention maps are saved to:
- `plots/<example_index>_attns<layer_index>.png`

### Metrics
The `decode(...)` function reports accuracy:
- training subset accuracy (first 100 examples by default)
- full dev accuracy

---

## Acknowledgements
This repository contains both:
- **framework/provided starter code** (see `Framework Code/`)
- **implemented solutions and updates** (root-level files + member work directories)

---

## License
No license file detected in the repository. If you plan to reuse or publish this work, consider adding a license (e.g., MIT, Apache-2.0).
