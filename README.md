# URR

This repository contains the compact implementation used for the URR method:

1. **Stage 1a, SID alignment.** Add hierarchical SID tokens to a frozen LLM and
   align the SID-path representation with an EMA item-text representation.
2. **Stage 1b, recommendation.** Unfreeze the aligned LLM and train it with
   recommendation cross entropy plus the **point** and **plan** objectives.
3. **Evaluation.** Generate 50 constrained beams using the MiniOneRec-style
   train prompt and report HR/NDCG at 1, 3, 5, 10, 20, and 50.

The included configuration reproduces the selected Qwen3-0.6B setting on the
Amazon Beauty data. Model checkpoints and datasets are intentionally excluded.

## Method At A Glance

### Stage 1a: SID-text alignment

For every catalog item, Stage 1a creates two views:

- SID view: `<|a_i|> <|b_j|> <|c_k|> <|d_l|> [ITEM]`
- text view: title, categories, brand, and description

The online branch reads the hidden state at `[ITEM]`; the EMA target branch
reads the final text token. Only newly added SID embedding rows and the
predictor are optimized. The released 0.6B configuration uses the raw
1024-dimensional hidden space, a linear predictor, and cosine distance.

### Stage 1b: point and plan

Stage 1b combines language-model cross entropy with:

- **Point objective:** the history state predicts the fixed Stage 1a SID hidden
  representation of the next item.
- **Remaining-plan objective:** the state before each target SID token predicts
  the current and all remaining RQ-SID levels in the LM-head embedding space.

`point_grad_ratio` and `plan_grad_ratio` are target gradient ratios relative to
cross entropy, not raw additive loss weights. The trainer calibrates the actual
weights from the first batch. For Qwen3-0.6B/Beauty they are 0.015 and 0.03.

## Repository Layout

```text
configs/
  beauty_0.6b_stage1a.yaml
  beauty_0.6b_stage1b.yaml
scripts/
  check_data.py
  evaluate.sh
  run_beauty_0.6b.sh
urr/
  data/                       # SID/text and recommendation datasets
  models/                     # JEPA aligner, projection, predictor, EMA
  train/stage1a_jepa.py       # Stage 1a training
  train/build_stage1b_targets.py
  train/stage1b.py            # point + plan training
  eval/evaluate.py            # constrained beam generation
  eval/calc.py                # HR/NDCG calculation
```

## Installation

Python 3.10 or newer and CUDA-capable PyTorch are required. The reference run
uses eight 80 GB GPUs, although Qwen3-0.6B can also be trained with fewer GPUs
after reducing the global batch size in the YAML files.

```bash
git clone <repository-url> KDD27
cd KDD27
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch wheels are platform-specific. Install the correct CUDA build from the
official PyTorch channel before the remaining requirements when necessary.

## Data

Place the processed Beauty files under `data/Beauty`, or set `DATA_DIR` to an
existing directory:

```text
data/Beauty/
  Beauty.index.json
  Beauty.item.json
  Beauty.train.inter
  Beauty.valid.inter
  Beauty.test.inter
```

`Beauty.index.json` maps each item ID to four RQ-SID tokens, with an optional
fifth collision-resolution token:

```json
{
  "item_1": ["<|a_12|>", "<|b_7|>", "<|c_93|>", "<|d_4|>"]
}
```

Interaction files are tab-separated and contain a header followed by
`user_id`, space-separated history IDs, and target item ID. Validate the data
before training:

```bash
python scripts/check_data.py --data_dir data/Beauty --dataset Beauty
```

## Complete 0.6B Beauty Run

The full command trains both stages, builds the fixed target bank, evaluates the
validation split, and finally evaluates the test split:

```bash
GPUS=0,1,2,3,4,5,6,7 \
BASE_MODEL=Qwen/Qwen3-0.6B \
DATA_DIR="$PWD/data/Beauty" \
bash scripts/run_beauty_0.6b.sh all
```

For a local model checkpoint, set `BASE_MODEL=/path/to/Qwen3-0.6B`. Outputs are
written to `outputs/qwen3_0.6b_beauty` by default. Each phase is resumable: a
complete checkpoint, target cache, or metric file is detected and skipped.

Run individual phases when needed:

```bash
bash scripts/run_beauty_0.6b.sh stage1a
bash scripts/run_beauty_0.6b.sh targets
bash scripts/run_beauty_0.6b.sh stage1b
bash scripts/run_beauty_0.6b.sh valid
bash scripts/run_beauty_0.6b.sh test
```

Inspect the commands without launching training:

```bash
DRY_RUN=1 bash scripts/run_beauty_0.6b.sh all
```

## Selected Hyperparameters

| Phase | Setting | Value |
|---|---|---:|
| Stage 1a | representation space | raw hidden (1024) |
| Stage 1a | predictor / loss | linear / cosine |
| Stage 1a | learning rate / epochs | 1.5e-3 / 10 |
| Stage 1a | EMA momentum | 0.999 |
| Stage 1b | learning rate | 2e-4 |
| Stage 1b | point gradient ratio | 0.015 |
| Stage 1b | plan gradient ratio | 0.03 |
| Stage 1b | plan scope / temperature | remaining / 0.07 |
| Stage 1b | selected checkpoint | epoch 3 |
| Evaluation | prompt / beams | train wording / 50 |

Stage 1b runs three epochs while retaining the four-epoch cosine scheduler. This
reproduces the original epoch-3 checkpoint without spending time or disk on the
unused fourth epoch.

## Outputs

```text
outputs/qwen3_0.6b_beauty/
  stage1a/aligned_base_model/
  targets/stage1b_targets.pt
  stage1b/epoch_3/
  evaluation/valid_metrics.json
  evaluation/test_metrics.json
```

The evaluation is deterministic constrained beam search. Sampling is disabled,
and model-specific Qwen generation defaults are explicitly prevented from
overriding the evaluation configuration.
