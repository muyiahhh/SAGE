"""Dataset that produces View A (SID sequence) and View B (text description).

View A:  <|a_i|> <|b_j|> <|c_k|> <|d_l|> [<|e_n|>] [ITEM]
         → hidden state at [ITEM] token position = z_SID

View B:  "title: <title>, categories: <cats>, description: <desc>"  <eos>
         → hidden state at EOS position = z_text

The dataset also returns SID generation sequences for Stage 1b SFT:
  Input:  user interaction history as SID tokens
  Label:  next item's SID tokens

Data source: a processed dataset directory supplied by the user.
Files used:
  - <DATASET>.index.json   : item_id → [SID tokens]
  - <DATASET>.item.json    : item_id → {title, description, categories, ...}
  - <DATASET>.train.inter  : user_id | item_id_list | next_item_id
  - <DATASET>.valid.inter  : same format
  - <DATASET>.test.inter   : same format
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ──────────────────────────────────────────────────────────────────────────────
# Item-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_index_sid_tokens(
    index: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    n_levels: Optional[int] = None,
    context: str = "index",
) -> None:
    """Fail fast if SID paths are missing or do not tokenize atomically."""
    vocab = tokenizer.get_vocab()
    missing: list[tuple[str, str]] = []
    non_atomic: list[tuple[str, list[str], list[int], list[int]]] = []

    for item_id, sid_tokens in index.items():
        levels = sid_tokens if n_levels is None else sid_tokens[:n_levels]
        token_ids = []
        for token in levels:
            if token not in vocab:
                missing.append((item_id, token))
            else:
                token_ids.append(tokenizer.convert_tokens_to_ids(token))

        if len(token_ids) == len(levels):
            encoded = tokenizer.encode("".join(levels), add_special_tokens=False)
            if encoded != token_ids:
                non_atomic.append((item_id, list(levels), token_ids, encoded))

    if missing:
        examples = ", ".join(f"{iid}:{tok}" for iid, tok in missing[:10])
        raise ValueError(
            f"{context} contains {len(missing)} SID token(s) missing from tokenizer "
            f"vocab. Examples: {examples}. Check SID code range and rerun Stage 1a "
            "from a tokenizer that registers all index tokens."
        )

    if non_atomic:
        item_id, levels, token_ids, encoded = non_atomic[0]
        raise ValueError(
            f"{context} contains SID paths that do not tokenize atomically. "
            f"Example item {item_id}: levels={levels}, token_ids={token_ids}, "
            f"encoded={encoded}. Register SID strings as special added tokens and "
            "rerun Stage 1a/Stage 1b."
        )


def lookup_sid_tokens(index: dict[str, list[str]], item_id: str) -> list[str]:
    sid_tokens = index.get(str(item_id)) or index.get(item_id)
    if sid_tokens is None:
        raise KeyError(f"Item id {item_id!r} is missing from SID index.")
    return sid_tokens


def build_item_text(item: dict) -> str:
    """Convert item metadata dict into a single text string for View B."""
    parts = []
    if item.get("title"):
        parts.append(f"title: {item['title']}")
    if item.get("categories"):
        cats = item["categories"]
        if isinstance(cats, list):
            cats = ", ".join(cats)
        parts.append(f"categories: {cats}")
    if item.get("brand"):
        parts.append(f"brand: {item['brand']}")
    if item.get("description"):
        desc = item["description"]
        # Truncate very long descriptions to keep token count manageable
        if len(desc) > 300:
            desc = desc[:300] + "..."
        parts.append(f"description: {desc}")
    return ", ".join(parts)


def build_sid_sequence(sid_tokens: list[str], item_marker: str) -> str:
    """Build the SID view-A input string: '<|a_i|> <|b_j|> ... [ITEM]'."""
    return " ".join(sid_tokens) + " " + item_marker


def build_history_sequence(
    history_sids: list[list[str]],
    target_sids: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_history: int = 20,
) -> tuple[list[int], list[int], list[int]]:
    """Encode user history as SID token sequence for Stage 1b SFT.

    Format:
      [BOS] <|a_i|> <|b_j|> ... [SEP] <|a_k|> ... [SEP] ... [EOS]
              ← past items →           ← target item →

    Returns (input_ids, labels, attention_mask) where labels mask the history
    portion with -100 and only supervise the target SID tokens.
    """
    sep = tokenizer.eos_token or "</s>"
    bos = tokenizer.bos_token or "<s>"

    # Truncate to max_history items
    if len(history_sids) > max_history:
        history_sids = history_sids[-max_history:]

    # Build history string — SID tokens concatenated without spaces, sep as item delimiter
    # Format matches MiniOneRec: no spaces between SID tokens within one item
    history_str = bos + " "
    for sid_list in history_sids:
        history_str += "".join(sid_list) + sep

    # Target string — same no-space format, sep (eos) appended directly
    target_str = "".join(target_sids) + sep

    history_ids = tokenizer.encode(history_str, add_special_tokens=False)
    target_ids = tokenizer.encode(target_str, add_special_tokens=False)

    input_ids = history_ids + target_ids
    labels = [-100] * len(history_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    return input_ids, labels, attention_mask


# ──────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────────────

class JEPAViewDataset(Dataset):
    """Produces (View A, View B) pairs for JEPA Stage 1a training.

    Each sample contains:
      sid_input_ids, sid_attention_mask, item_pos  → View A
      text_input_ids, text_attention_mask, eos_pos → View B
    """

    def __init__(
        self,
        data_dir: str,
        dataset: str,
        tokenizer: PreTrainedTokenizerBase,
        item_marker: str = "[ITEM]",
        max_text_len: int = 128,
        max_sid_len: int = 16,
    ):
        self.tokenizer = tokenizer
        self.item_marker = item_marker
        self.max_text_len = max_text_len
        self.max_sid_len = max_sid_len

        # Load index (item_id → SID tokens) and items (item_id → metadata)
        with open(os.path.join(data_dir, f"{dataset}.index.json")) as f:
            self.index: dict[str, list[str]] = json.load(f)
        with open(os.path.join(data_dir, f"{dataset}.item.json")) as f:
            self.items: dict[str, dict] = json.load(f)

        validate_index_sid_tokens(self.index, tokenizer, context=f"{dataset}.index.json")

        # Build ordered list of item ids that have both SID and text metadata
        self.item_ids = [
            iid for iid in self.index.keys() if str(iid) in self.items or iid in self.items
        ]

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> dict:
        iid = self.item_ids[idx]
        sid_tokens = self.index[iid]                    # e.g. ['<|a_3|>', '<|b_7|>', '<|c_5|>', '<|d_2|>']
        item_meta = self.items.get(str(iid)) or self.items.get(iid, {})

        # ── View A: SID sequence ─────────────────────────────────────────────
        sid_seq = build_sid_sequence(sid_tokens, self.item_marker)
        sid_enc = self.tokenizer(
            sid_seq,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_sid_len,
            padding=False,
            return_tensors=None,
        )
        sid_input_ids = sid_enc["input_ids"]
        sid_attention_mask = sid_enc["attention_mask"]

        # Find [ITEM] marker position in tokenized sequence
        item_marker_id = self.tokenizer.convert_tokens_to_ids(self.item_marker)
        try:
            item_pos = sid_input_ids.index(item_marker_id)
        except ValueError:
            # Fallback: use last non-pad position
            item_pos = len(sid_input_ids) - 1

        # ── View B: text sequence ─────────────────────────────────────────────
        text = build_item_text(item_meta)
        text_enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_text_len,
            padding=False,
            return_tensors=None,
        )
        text_input_ids = text_enc["input_ids"]
        text_attention_mask = text_enc["attention_mask"]

        # EOS position: last real token (last 1 in attention_mask)
        eos_pos = len(text_input_ids) - 1

        return {
            "sid_input_ids": sid_input_ids,
            "sid_attention_mask": sid_attention_mask,
            "item_pos": item_pos,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "eos_pos": eos_pos,
        }


class SIDSeqRecDataset(Dataset):
    """SID sequence recommendation dataset for Stage 1b SFT.

    Each sample:
      input_ids, labels, attention_mask
    where labels supervise only the next-item SID tokens (history is masked -100).
    """

    def __init__(
        self,
        data_dir: str,
        dataset: str,
        split: str,   # "train", "valid", or "test"
        tokenizer: PreTrainedTokenizerBase,
        item_marker: str = "[ITEM]",
        max_seq_len: int = 512,
        max_history: int = 20,
        # Also provide JEPA view pairs for the Stage 1b regularizer
        include_jepa_views: bool = False,
        max_text_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.item_marker = item_marker
        self.max_seq_len = max_seq_len
        self.max_history = max_history
        self.include_jepa_views = include_jepa_views
        self.max_text_len = max_text_len

        with open(os.path.join(data_dir, f"{dataset}.index.json")) as f:
            self.index: dict[str, list[str]] = json.load(f)
        with open(os.path.join(data_dir, f"{dataset}.item.json")) as f:
            self.items: dict[str, dict] = json.load(f)

        validate_index_sid_tokens(self.index, tokenizer, context=f"{dataset}.index.json")

        # Load interaction file
        inter_file = os.path.join(data_dir, f"{dataset}.{split}.inter")
        self.samples = self._load_inter(inter_file)

    def _load_inter(self, path: str) -> list[dict]:
        """Parse .inter file: user_id, item_id_list (space-sep), next_item_id."""
        samples = []
        with open(path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                user_id = parts[0]
                history_ids = parts[1].split()
                target_id = parts[2]
                samples.append({
                    "user_id": user_id,
                    "history_ids": history_ids,
                    "target_id": target_id,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        history_ids = sample["history_ids"]
        target_id = sample["target_id"]

        # Convert item IDs to SID token lists
        history_sids = [
            lookup_sid_tokens(self.index, iid)
            for iid in history_ids
        ]
        target_sids = lookup_sid_tokens(self.index, target_id)

        input_ids, labels, attention_mask = build_history_sequence(
            history_sids, target_sids, self.tokenizer, self.max_history
        )

        # Truncate to max_seq_len
        if len(input_ids) > self.max_seq_len:
            # Keep the target portion intact — truncate from the left of history
            excess = len(input_ids) - self.max_seq_len
            input_ids = input_ids[excess:]
            labels = labels[excess:]
            attention_mask = attention_mask[excess:]

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if self.include_jepa_views:
            # JEPA views for the target item (used as regularizer in Stage 1b)
            sid_seq = build_sid_sequence(target_sids, self.item_marker)
            sid_enc = self.tokenizer(
                sid_seq,
                add_special_tokens=True,
                truncation=True,
                max_length=16,
                padding=False,
            )
            item_marker_id = self.tokenizer.convert_tokens_to_ids(self.item_marker)
            sid_ids = sid_enc["input_ids"]
            try:
                item_pos = sid_ids.index(item_marker_id)
            except ValueError:
                item_pos = len(sid_ids) - 1

            target_meta = self.items.get(str(target_id)) or self.items.get(target_id, {})
            text = build_item_text(target_meta)
            text_enc = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_text_len,
                padding=False,
            )
            eos_pos = len(text_enc["input_ids"]) - 1

            result.update({
                "sid_input_ids": sid_ids,
                "sid_attention_mask": sid_enc["attention_mask"],
                "item_pos": item_pos,
                "text_input_ids": text_enc["input_ids"],
                "text_attention_mask": text_enc["attention_mask"],
                "eos_pos": eos_pos,
            })

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────────────────────────────────────

def pad_to_max(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


def jepa_collate_fn(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate a batch from JEPAViewDataset or SIDSeqRecDataset (with JEPA views)."""
    result = {}

    # Standard SFT fields (may be absent in JEPA-only dataset)
    if "input_ids" in batch[0]:
        result["input_ids"] = pad_to_max([b["input_ids"] for b in batch], pad_id)
        result["labels"] = pad_to_max([b["labels"] for b in batch], -100)
        result["attention_mask"] = pad_to_max([b["attention_mask"] for b in batch], 0)

    # JEPA view fields
    if "sid_input_ids" in batch[0]:
        result["sid_input_ids"] = pad_to_max([b["sid_input_ids"] for b in batch], pad_id)
        result["sid_attention_mask"] = pad_to_max([b["sid_attention_mask"] for b in batch], 0)
        result["item_pos"] = torch.tensor([b["item_pos"] for b in batch], dtype=torch.long)
        result["text_input_ids"] = pad_to_max([b["text_input_ids"] for b in batch], pad_id)
        result["text_attention_mask"] = pad_to_max([b["text_attention_mask"] for b in batch], 0)
        result["eos_pos"] = torch.tensor([b["eos_pos"] for b in batch], dtype=torch.long)

    return result
