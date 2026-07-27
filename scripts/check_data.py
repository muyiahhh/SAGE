#!/usr/bin/env python3
"""Validate the processed dataset files expected by SAGE."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SID_PATTERN = re.compile(r"^<\|([a-e])_(\d+)\|>$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a SAGE dataset directory")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", default="Beauty")
    args = parser.parse_args()

    root = Path(args.data_dir)
    required = [
        root / f"{args.dataset}.index.json",
        root / f"{args.dataset}.item.json",
        root / f"{args.dataset}.train.inter",
        root / f"{args.dataset}.valid.inter",
        root / f"{args.dataset}.test.inter",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n  " + "\n  ".join(missing))

    with open(required[0], encoding="utf-8") as handle:
        index = json.load(handle)
    with open(required[1], encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(index, dict) or not index:
        raise ValueError("The SID index must be a non-empty JSON object")
    if not isinstance(items, dict) or not items:
        raise ValueError("The item metadata must be a non-empty JSON object")

    for item_id, tokens in index.items():
        if not isinstance(tokens, list):
            raise ValueError(f"SID path for item {item_id} is not a list")
        by_level = {}
        for token in tokens:
            match = SID_PATTERN.match(token)
            if match:
                by_level[match.group(1)] = int(match.group(2))
        missing_levels = [level for level in "abcd" if level not in by_level]
        if missing_levels:
            raise ValueError(f"Item {item_id} is missing SID levels {missing_levels}")
        if any(not 1 <= by_level[level] <= 256 for level in "abcd"):
            raise ValueError(f"Item {item_id} has an SID code outside 1..256")

    split_counts = {}
    for split_path in required[2:]:
        split = split_path.name.split(".")[-2]
        count = 0
        with open(split_path, encoding="utf-8") as handle:
            header = handle.readline().strip()
            if not header:
                raise ValueError(f"Missing header in {split_path}")
            for line_number, line in enumerate(handle, start=2):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    raise ValueError(f"Malformed row {split_path}:{line_number}")
                unknown = [
                    item_id
                    for item_id in parts[1].split() + [parts[2]]
                    if str(item_id) not in index
                ]
                if unknown:
                    raise ValueError(f"Unknown item {unknown[0]!r} at {split_path}:{line_number}")
                count += 1
        if count == 0:
            raise ValueError(f"No interactions found in {split_path}")
        split_counts[split] = count

    print(
        f"Dataset OK: {args.dataset} | items={len(index)} | "
        + " | ".join(f"{split}={count}" for split, count in split_counts.items())
    )


if __name__ == "__main__":
    main()
