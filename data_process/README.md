# Optional upstream data preparation

This directory preserves the data-preparation code distributed with the GRRM
pipeline. It provides a path from raw Amazon review data to the processed
metadata, interactions, and residual-quantized SIDs expected by SAGE.

The SAGE experiments in the paper use the already processed data release, so
this directory is **not part of the default reproduction path**. The code is
included for completeness and for users who need to rebuild the data from raw
Amazon files.

## Scope and provenance

The scripts in this directory originate from the upstream data pipeline rather
than the SAGE method. They are kept separate from the `sage` Python package and
have their own optional dependencies. Before public redistribution, retain the
upstream attribution and confirm that its license permits inclusion.

Some later steps synthesize preferences or reasoning traces through an OpenAI-
compatible remote or local model endpoint. These steps are not needed by SAGE,
which consumes only the item metadata, interaction splits, and SID index.

## Raw data layout

```text
data/
  Amazon/
    Ratings/
      ratings_Sports_and_Outdoors.csv
    Review/
      reviews_Sports_and_Outdoors_5.json.gz
    Metadata/
      meta_Sports_and_Outdoors.json.gz
```

Install the optional dependencies separately:

```bash
pip install -r data_process/requirements.txt
```

## Amazon preprocessing

Run one of the following commands from the repository root:

```bash
# Amazon-2014
python data_process/preprocess/amazon14_data_process.py \
  --dataset <dataset-name> \
  --input data/Amazon \
  --output data

# Amazon-2018
python data_process/preprocess/amazon18_data_process.py \
  --dataset <dataset-name> \
  --input data/Amazon \
  --output data
```

## SID construction

Encode item text and run constrained residual k-means:

```bash
torchrun --nproc_per_node=8 data_process/preprocess/text_emb.py \
  --dataset <dataset-name> \
  --plm_checkpoint <model-path>

python data_process/preprocess/rkmeans_constrained.py \
  --dataset <dataset-name>
```

The scripts inherit the upstream pipeline's relative-path assumptions. Run them
from the repository root and inspect each command's arguments before launching
a long job.

## Upstream-only generation steps

`construct_preference.py`, `construct_cot.py`, `get_align.py`, `transform.py`,
and `transform_rl.py` prepare preference, reasoning, SFT, or RL artifacts used
by the broader upstream pipeline. They are retained verbatim for provenance but
are not called by `scripts/run_sage.sh`. In particular, `get_align.py` expects
the broader upstream training repository's `utils.utils` module and is not a
standalone SAGE entry point.

Never commit API keys, generated responses, raw reviews, or processed datasets.
Pass credentials through the command line or environment only on a trusted
machine.
