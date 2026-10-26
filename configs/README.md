# Configurations

The YAML files define the reference Qwen3-0.6B Beauty run.

- `beauty_qwen3_0.6b_isg.yaml` trains item-level semantic grounding.
- `beauty_qwen3_0.6b_psa.yaml` trains recommendation with predictive semantic
  anchoring. Its comments list the per-dataset values used in the paper.

Paths, dataset names, and model checkpoints can be overridden by
`scripts/run_sage.sh`.
