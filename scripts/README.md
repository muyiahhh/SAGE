# Run scripts

- `check_data.py` validates the processed dataset schema and SID ranges.
- `run_sage.sh` launches ISG, anchor-bank construction, PSA, and evaluation.
- `evaluate.sh` shards constrained beam search across the requested GPUs and
  computes HR/NDCG metrics.

Run `DRY_RUN=1 bash scripts/run_sage.sh all` to inspect the generated commands.
