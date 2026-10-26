# SAGE

Official implementation of **SAGE**, a representation-space training framework
for SID-based generative recommendation.

SAGE has two training components:

1. **Item-level Semantic Grounding (ISG)** treats an item's SID sequence and
   text as two views of the same item. With the LLM frozen, only the new SID
   embeddings and a linear predictor are trained so that the SID view predicts
   the frozen text view.
2. **Predictive Semantic Anchoring (PSA)** fine-tunes the recommender with
   next-token cross-entropy and one shared linear predictor applied to the four
   SID decoding states. Its semantic readout aligns each prediction with the
   fixed ISG anchor of the target item, and its identity readout predicts the
   remaining SID codes against the model's own SID token embeddings.

Inference uses the autoregressive LLM with standard trie-constrained beam
search. The anchor bank and the PSA predictor are not used for decoding.

## Repository structure

```text
configs/                 Reference experiment configurations
data/                    Expected processed-data layout
data_process/            Optional upstream Amazon-to-GRRM preprocessing code
sage/
  data/                  Item views, prompts, and recommendation datasets
  models/                ISG aligner and predictors
  train/                 ISG, anchor-bank construction, and PSA trainers
  eval/                  Constrained decoding and HR/NDCG evaluation
  utils/                 Shared utilities
scripts/                 Data checks and end-to-end launchers
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
ISG, builds the fixed item-anchor bank, trains PSA, and evaluates both
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
bash scripts/run_sage.sh psa
bash scripts/run_sage.sh valid
bash scripts/run_sage.sh test
```

Inspect all commands without launching a model:

```bash
DRY_RUN=1 bash scripts/run_sage.sh all
```

Use `ISG_CONFIG` and `PSA_CONFIG` to select other configuration files. The
dataset, model, paths, GPU list, and output directory can be overridden through
the environment variables shown above.

## Reference settings

| Component | Setting | Value |
|---|---|---:|
| ISG | trainable parameters | SID embeddings + linear predictor |
| ISG | text view / loss | frozen LLM last token / cosine |
| ISG | learning rate / epochs | 1.5e-3 / 10 |
| PSA | learning rate / epochs | 2e-4 / 3 of a 4-epoch schedule |
| PSA | identity readout scope / temperature | future codes / 0.07 |
| PSA | semantic gradient ratio | 0.010 |
| PSA | identity gradient ratio | 0.030 |
| Evaluation | beams | 50 |

The gradient ratios are targets relative to the next-token cross-entropy
gradient at the decoding states each readout supervises. The trainer calibrates
the two loss weights on its first optimization step and keeps them fixed. The
Instruments and Sports values are listed in `configs/beauty_qwen3_0.6b_psa.yaml`.

## Outputs

```text
outputs/qwen3_0.6b_beauty/
  isg/grounded_model/
  anchors/item_anchors.pt
  psa/epoch_3/
  evaluation/valid_metrics.json
  evaluation/test_metrics.json
```

Generated data, model weights, logs, and experiment-tracking files are excluded
from version control.

## Reproducibility notes

- Random seeds are explicit in the YAML files and evaluation is deterministic.
- Evaluation uses 50 trie-constrained beams without sampling or auxiliary-head
  reranking.
- The reference launcher uses the training prompt wording during evaluation,
  matching the paper protocol.
