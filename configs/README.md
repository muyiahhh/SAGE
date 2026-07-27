# Configurations

The YAML files define the reference Qwen3-0.6B Beauty run.

- `beauty_qwen3_0.6b_isg.yaml` trains item-level semantic grounding.
- `beauty_qwen3_0.6b_gsa.yaml` trains recommendation with DSA and HPP.

Paths, dataset names, and model checkpoints can be overridden by
`scripts/run_sage.sh`. When changing model size, set `d_proj` to the model
hidden size if `projection_type: identity` is retained.
