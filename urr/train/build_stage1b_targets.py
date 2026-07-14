"""Build the fixed SID target bank used by URR Stage 1b.

Stage 1b's point objective predicts the Stage 1a-aligned hidden representation
of the target item's SID path.  This command encodes every catalog item once
and stores both that hidden representation and its four RQ-SID code indices.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urr.data.views import build_sid_sequence, lookup_sid_tokens
from urr.models.jepa import ITEM_MARKER, SID_LEVELS


_SID_RE = re.compile(r"^<\|([a-d])_(\d+)\|>$")


def extract_sid_codes(index: dict, item_ids: list[str]) -> torch.Tensor:
    """Convert ``<|a_1|>...<|d_256|>`` paths to zero-based code rows."""
    rows: list[list[int]] = []
    for item_id in item_ids:
        by_level: dict[str, int] = {}
        for token in lookup_sid_tokens(index, item_id):
            match = _SID_RE.match(token)
            if match:
                by_level[match.group(1)] = int(match.group(2)) - 1
        missing = [level for level in SID_LEVELS if level not in by_level]
        if missing:
            raise ValueError(f"Item {item_id} is missing SID levels: {missing}")
        row = [by_level[level] for level in SID_LEVELS]
        if not all(0 <= code < 256 for code in row):
            raise ValueError(f"Item {item_id} has an out-of-range SID code: {row}")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def encode_sid_hidden(
    model,
    tokenizer,
    sequences: list[str],
    *,
    batch_size: int,
    max_sid_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the final-layer hidden state at the ``[ITEM]`` marker."""
    marker_id = tokenizer.convert_tokens_to_ids(ITEM_MARKER)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        encoded = tokenizer(
            sequences[start : start + batch_size],
            add_special_tokens=True,
            truncation=True,
            max_length=max_sid_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        marker_mask = input_ids.eq(marker_id) & attention_mask.bool()
        if not bool(marker_mask.any(dim=1).all()):
            raise ValueError("At least one SID sequence lost [ITEM] during tokenization")
        item_position = marker_mask.long().argmax(dim=1)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = output.hidden_states[-1]
        batch_index = torch.arange(hidden.size(0), device=device)
        chunks.append(hidden[batch_index, item_position].float().cpu())
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build URR Stage 1b SID targets")
    parser.add_argument("--stage1a_model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_sid_len", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.stage1a_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.stage1a_model,
            dtype=dtype,
            trust_remote_code=True,
            use_cache=False,
        )
        .to(device)
        .eval()
    )

    index_path = os.path.join(args.data_dir, f"{args.dataset}.index.json")
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    item_ids = [str(item_id) for item_id in index]
    sid_codes = extract_sid_codes(index, item_ids)
    sid_sequences = [
        build_sid_sequence(lookup_sid_tokens(index, item_id), ITEM_MARKER) for item_id in item_ids
    ]

    print(f"[INFO] Encoding {len(item_ids)} Stage 1a SID targets")
    sid_hidden = encode_sid_hidden(
        model,
        tokenizer,
        sid_sequences,
        batch_size=args.batch_size,
        max_sid_len=args.max_sid_len,
        device=device,
    )
    print(
        "[INFO] SID target statistics: "
        f"dim={sid_hidden.size(1)} "
        f"norm={sid_hidden.norm(dim=-1).mean():.3f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "item_ids": item_ids,
            "sid_codes": sid_codes.contiguous(),
            "sid_hidden": sid_hidden.contiguous(),
            "meta": {
                "version": 1,
                "stage1a_model": str(Path(args.stage1a_model).resolve()),
                "dataset": args.dataset,
                "hidden_size": int(model.config.hidden_size),
                "max_sid_len": int(args.max_sid_len),
            },
        },
        output_path,
    )
    print(f"[INFO] Saved Stage 1b target cache -> {output_path}")


if __name__ == "__main__":
    main()
