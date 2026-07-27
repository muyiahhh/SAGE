"""SAGE evaluation metric calculation.

Computes HR@K and NDCG@K from JSON result files produced by evaluate.py.
Compatible with multi-shard output (pass multiple --path arguments).

The result JSON format is:
  [{"output": "<|a_i|><|b_j|><|c_k|><|d_l|>[<|e_n|>]",
    "predict": ["<|a_i|><|b_j|><|c_k|><|d_l|>[<|e_n|>]", ...]}, ...]

Usage:
  python -m sage.eval.calc \\
    --path results/instruments_sage.json \\
    --index_path /path/to/Instruments.index.json \\
    --save_path results/metrics_sage.json

Merge multiple shards first:
  python -m sage.eval.calc \\
    --path results/shard_0.json results/shard_1.json ... \\
    --index_path /path/to/Instruments.index.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from tqdm import tqdm
import numpy as np


TOPK_LIST = [1, 3, 5, 10, 20, 50]


def load_index_sid_set(index_path: str) -> set:
    """Build the set of all valid full SID strings from index.json."""
    with open(index_path) as f:
        index = json.load(f)
    return {"".join(v) for v in index.values() if v}


def calc_metrics(
    result_files: list[str],
    index_path: str,
    save_path: str = "",
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_tags: str = "",
):
    # Load all valid SID strings for OOV detection
    valid_sids = load_index_sid_set(index_path)

    use_wandb = bool(wandb_project)
    if use_wandb:
        import wandb

        tags = [t.strip() for t in wandb_tags.split(",")] if wandb_tags else []
        wandb.init(project=wandb_project, name=wandb_run_name or None, tags=tags)

    all_results = {}
    for path in result_files:
        with open(path) as f:
            test_data = json.load(f)

        n_beam = -1
        oov_total = 0
        valid_topk = []
        ALLNDCG = None
        ALLHR = None

        for sample in tqdm(test_data, desc=f"Scoring {os.path.basename(path)}"):
            predictions = [p.strip() for p in sample["predict"]]
            if n_beam == -1:
                n_beam = len(predictions)
                valid_topk = [k for k in TOPK_LIST if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))

            target_value = sample.get("output", sample.get("target"))
            if target_value is None:
                raise KeyError(
                    "Evaluation sample must contain `output` (standard) or `target` "
                    f"(legacy alias). Available keys: {sorted(sample.keys())}"
                )
            target = target_value.strip()

            # Count OOV predictions
            oov_total += sum(1 for p in predictions if p not in valid_sids)

            # Find first hit position
            hit_pos = n_beam  # no hit
            for i, pred in enumerate(predictions):
                if pred == target:
                    hit_pos = i
                    break

            for idx, topk in enumerate(valid_topk):
                if hit_pos < topk:
                    ALLNDCG[idx] += 1.0 / math.log2(hit_pos + 2)
                    ALLHR[idx] += 1

        n = len(test_data)
        ndcg = ALLNDCG / n / (1.0 / math.log2(2))
        hr = ALLHR / n

        print(f"\n=== {path} ===")
        print(f"Beams: {n_beam}  |  Samples: {n}  |  OOV predictions: {oov_total}")
        print(f"K:    {valid_topk}")
        print(f"NDCG: {[round(v, 4) for v in ndcg.tolist()]}")
        print(f"HR:   {[round(v, 4) for v in hr.tolist()]}")

        all_results[path] = {
            "n_beam": n_beam,
            "n_samples": n,
            "oov_predictions": oov_total,
            "topk": valid_topk,
            "NDCG": ndcg.tolist(),
            "HR": hr.tolist(),
        }

        if use_wandb:
            import wandb

            log_dict = {"n_beam": n_beam, "oov_predictions": oov_total}
            for k, nd, h in zip(valid_topk, ndcg.tolist(), hr.tolist()):
                log_dict[f"NDCG@{k}"] = nd
                log_dict[f"HR@{k}"] = h
            wandb.log(log_dict)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n[INFO] Metrics saved to {save_path}")

    if use_wandb:
        import wandb

        wandb.finish()

    return all_results


def merge_shards(shard_paths: list[str], merged_path: str):
    """Merge shard JSONs into one file."""
    merged = []
    for p in shard_paths:
        with open(p) as f:
            merged.extend(json.load(f))
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[INFO] Merged {len(shard_paths)} shards → {merged_path}  ({len(merged)} samples)")


def main():
    parser = argparse.ArgumentParser(description="SAGE metric calculation")
    parser.add_argument("--path", nargs="+", required=True, help="Result JSON file(s)")
    parser.add_argument("--index_path", required=True, help="Path to {dataset}.index.json")
    parser.add_argument("--save_path", default="", help="Output metrics JSON path")
    parser.add_argument(
        "--merge", default="", help="If set, merge --path files into this path first"
    )
    parser.add_argument(
        "--merge_only", action="store_true", help="Only merge shards and skip metric calculation"
    )
    parser.add_argument("--wandb_project", default="", help="W&B project name (disabled if empty)")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name")
    parser.add_argument("--wandb_tags", default="", help="Comma-separated W&B tags")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.path, args.merge)
        if args.merge_only:
            return
        paths = [args.merge]
    else:
        paths = args.path

    calc_metrics(
        paths,
        args.index_path,
        args.save_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )


if __name__ == "__main__":
    main()
