# SAGE

Official implementation of **SAGE**, a representation-space training framework
for SID-based generative recommendation.

SAGE has two training components:

1. **Item-level Semantic Grounding (ISG)** treats an item's SID sequence and
   text as two views of the same item. A frozen LLM learns grounded SID-token
   embeddings through joint-embedding predictive alignment.
2. **Generation-time Semantic Anchoring (GSA)** fine-tunes the recommender with
   next-token cross-entropy and two training-only objectives. Decision-State
   Anchoring (DSA) identifies the target item against the fixed ISG anchor bank,
   while Hierarchical Path Prediction (HPP) predicts the current and remaining
   SID codes from every prefix state.

Inference uses the autoregressive LLM with standard trie-constrained beam
search. The anchor bank and HPP heads are not used for decoding.

## Repository structure

```text
configs/                 Reference experiment configurations
data/                    Expected processed-data layout
data_process/            Optional upstream Amazon-to-GRRM preprocessing code
sage/
  data/                  Item views, prompts, and recommendation datasets
  models/                ISG projection, predictor, and EMA modules
  train/                 ISG, anchor-bank construction, and GSA trainers
  eval/                  Constrained decoding and HR/NDCG evaluation
  utils/                 Shared utilities
scripts/                 Data checks and end-to-end launchers
tests/                   Lightweight unit and configuration tests
```

Every functional directory contains a short `README.md` describing its role.

## Installation

The code requires Python 3.10 or newer. Create an isolated environment and
install the package in editable mode:

```bash
git clone <anonymous-repository-url> SAGE
cd SAGE
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

PyTorch wheels are platform-specific. Install the appropriate CUDA build from
the official PyTorch channel first if the pinned wheel in `requirements.txt`
does not match the machine.

## Data

The paper experiments start from the processed data release and do **not** run
the raw Amazon preprocessing pipeline. Place one dataset under `data/<name>`:

```text
data/Beauty/
  Beauty.index.json
  Beauty.item.json
  Beauty.train.inter
  Beauty.valid.inter
  Beauty.test.inter
```

The three datasets used in the paper are `Beauty`, `Instruments`, and `Sports`.
See [`data/README.md`](data/README.md) for the file schemas. Validate a dataset
before training:

```bash
python scripts/check_data.py --data_dir data/Beauty --dataset Beauty
```

The optional [`data_process/`](data_process/) directory retains the upstream
GRRM preprocessing pipeline for users who need to reconstruct these files from
raw Amazon data. It is not required to reproduce the SAGE experiments.

## End-to-end run

The reference configuration uses Qwen3-0.6B on Beauty. `run_sage.sh` performs
ISG, builds the fixed item-anchor bank, trains GSA, and evaluates both
validation and test splits:

```bash
GPUS=0,1,2,3,4,5,6,7 \
BASE_MODEL=Qwen/Qwen3-0.6B \
DATASET=Beauty \
DATA_DIR="$PWD/data/Beauty" \
OUTPUT_ROOT="$PWD/outputs/qwen3_0.6b_beauty" \
bash scripts/run_sage.sh all
```

Each phase is resumable and can be run separately:

```bash
bash scripts/run_sage.sh check
bash scripts/run_sage.sh isg
bash scripts/run_sage.sh anchors
bash scripts/run_sage.sh gsa
bash scripts/run_sage.sh valid
bash scripts/run_sage.sh test
```

Inspect all commands without launching a model:

```bash
DRY_RUN=1 bash scripts/run_sage.sh all
```

Use `ISG_CONFIG` and `GSA_CONFIG` to select other configuration files. The
dataset, model, paths, GPU list, and output directory can be overridden through
the environment variables shown above.

## Reference settings

| Component | Setting | Value |
|---|---|---:|
| ISG | representation space | raw hidden space |
| ISG | predictor / loss | linear / cosine |
| ISG | learning rate / epochs | 1.5e-3 / 10 |
| ISG | EMA momentum | 0.999 |
| GSA | learning rate | 2e-4 |
| DSA | objective / temperature | catalog CE / 0.07 |
| DSA | target gradient ratio | 0.010 |
| HPP | scope / temperature | remaining / 0.07 |
| HPP | target gradient ratio | 0.03 |
| Evaluation | beams | 50 |

The gradient ratios are targets relative to the next-token cross-entropy
gradient at the supervised hidden states. The trainer calibrates the concrete
auxiliary-loss weights on its first optimization step.

## Outputs

```text
outputs/qwen3_0.6b_beauty/
  isg/grounded_model/
  anchors/item_anchors.pt
  gsa/epoch_3/
  evaluation/valid_metrics.json
  evaluation/test_metrics.json
```

Generated data, model weights, logs, and experiment-tracking files are excluded
from version control.

## Testing

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The tests cover SID parsing, metric calculation, the released configurations,
and DSA/HPP forward and gradient diagnostics using a small synthetic model.

## Reproducibility notes

- Random seeds are explicit in the YAML files and evaluation is deterministic.
- Evaluation uses 50 trie-constrained beams without sampling or auxiliary-head
  reranking.
- The reference launcher uses the training prompt wording during evaluation,
  matching the paper protocol.
- Raw datasets and pretrained model checkpoints are not redistributed.

This repository is prepared for anonymous review and intentionally contains no
author or institutional metadata.
