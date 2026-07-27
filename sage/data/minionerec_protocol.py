"""MiniOneRec-compatible SID recommendation prompt/data utilities.

This module keeps prompt construction separate from the item-view dataset so
the SID-only recommendation protocol remains easy to inspect.
"""

from __future__ import annotations

import json
import os

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from sage.data.views import (
    build_item_text,
    build_sid_sequence,
    lookup_sid_tokens,
    validate_index_sid_tokens,
)


MINIONEREC_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""

TRAIN_USER_INPUT_TEMPLATE = (
    "The user has interacted with items {history} in chronological order. "
    "Can you predict the next possible item that the user may expect?"
)

EVAL_USER_INPUT_TEMPLATE = (
    "Can you predict the next possible item the user may expect, given the "
    "following chronological interaction history: {history}"
)

RESPONSE_PREFIX = "### Response:\n"


def encode_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    bos: bool = False,
    eos: bool = False,
) -> list[int]:
    """Mirror MiniOneRec's local Tokenizer.encode helper.

    MiniOneRec strips tokenizer-added BOS/EOS, then conditionally adds BOS/EOS.
    Using ``add_special_tokens=False`` plus explicit BOS/EOS gives the same
    behavior for Qwen-style tokenizers and avoids accidental extra chat tokens.
    """

    ids = tokenizer.encode(text, add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        ids = [int(tokenizer.bos_token_id)] + ids
    if eos and tokenizer.eos_token_id is not None:
        ids = ids + [int(tokenizer.eos_token_id)]
    return ids


def sid_string(index: dict[str, list[str]], item_id: str) -> str:
    return "".join(lookup_sid_tokens(index, str(item_id)))


def history_sid_string(
    history_ids: list[str],
    index: dict[str, list[str]],
    max_history: int = 20,
) -> str:
    history_ids = [str(iid) for iid in history_ids]
    if len(history_ids) > max_history:
        history_ids = history_ids[-max_history:]
    return ", ".join(sid_string(index, iid) for iid in history_ids)


def build_minionerec_prompt_text(
    history_ids: list[str],
    index: dict[str, list[str]],
    *,
    max_history: int = 20,
    eval_style: bool = False,
    include_response_prefix: bool = True,
) -> str:
    """Build MiniOneRec-style prompt without the target response."""

    history = history_sid_string(history_ids, index, max_history=max_history)
    template = EVAL_USER_INPUT_TEMPLATE if eval_style else TRAIN_USER_INPUT_TEMPLATE
    user_input = template.format(history=history)
    prompt = (
        MINIONEREC_INSTRUCTION
        + "### User Input: \n"
        + user_input
        + "\n\n"
    )
    if include_response_prefix:
        prompt += RESPONSE_PREFIX
    return prompt


def build_minionerec_train_sequence(
    history_ids: list[str],
    target_id: str,
    index: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_history: int = 20,
    eval_style: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Encode MiniOneRec-style SFT sequence.

    Labels supervise only ``target_sid + "\\n" + EOS``; the instruction, user
    input, and response prefix are masked with ``-100``.
    """

    prompt = build_minionerec_prompt_text(
        history_ids,
        index,
        max_history=max_history,
        eval_style=eval_style,
    )
    target = sid_string(index, str(target_id)) + "\n"
    prompt_ids = encode_text(tokenizer, prompt, bos=True, eos=False)
    target_ids = encode_text(tokenizer, target, bos=False, eos=True)
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, labels, attention_mask


class MiniOneRecSIDSeqRecDataset(Dataset):
    """SAGE .inter data rendered with MiniOneRec's instruction prompt.

    It keeps the optional ISG target-item views used by the training pipeline.
    """

    def __init__(
        self,
        data_dir: str,
        dataset: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        item_marker: str = "[ITEM]",
        max_seq_len: int = 512,
        max_history: int = 20,
        include_jepa_views: bool = False,
        max_text_len: int = 128,
        eval_prompt_style_for_sft: bool = False,
    ):
        self.tokenizer = tokenizer
        self.item_marker = item_marker
        self.max_seq_len = int(max_seq_len)
        self.max_history = int(max_history)
        self.include_jepa_views = include_jepa_views
        self.max_text_len = int(max_text_len)
        self.eval_prompt_style_for_sft = bool(eval_prompt_style_for_sft)

        with open(os.path.join(data_dir, f"{dataset}.index.json"), encoding="utf-8") as f:
            self.index: dict[str, list[str]] = json.load(f)
        with open(os.path.join(data_dir, f"{dataset}.item.json"), encoding="utf-8") as f:
            self.items: dict[str, dict] = json.load(f)

        validate_index_sid_tokens(self.index, tokenizer, context=f"{dataset}.index.json")
        self.samples = self._load_inter(os.path.join(data_dir, f"{dataset}.{split}.inter"))

    def _load_inter(self, path: str) -> list[dict]:
        samples: list[dict] = []
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                samples.append(
                    {
                        "user_id": parts[0],
                        "history_ids": parts[1].split(),
                        "target_id": parts[2],
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        history_ids = [str(iid) for iid in sample["history_ids"]]
        target_id = str(sample["target_id"])
        target_sids = lookup_sid_tokens(self.index, target_id)

        input_ids, labels, attention_mask = build_minionerec_train_sequence(
            history_ids,
            target_id,
            self.index,
            self.tokenizer,
            max_history=self.max_history,
            eval_style=self.eval_prompt_style_for_sft,
        )

        if len(input_ids) > self.max_seq_len:
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
            sid_seq = build_sid_sequence(target_sids, self.item_marker)
            sid_enc = self.tokenizer(
                sid_seq,
                add_special_tokens=True,
                truncation=True,
                max_length=16,
                padding=False,
            )
            sid_ids = sid_enc["input_ids"]
            item_marker_id = self.tokenizer.convert_tokens_to_ids(self.item_marker)
            try:
                item_pos = sid_ids.index(item_marker_id)
            except ValueError:
                item_pos = len(sid_ids) - 1

            target_meta = self.items.get(target_id) or self.items.get(str(target_id), {})
            text = build_item_text(target_meta)
            text_enc = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_text_len,
                padding=False,
            )
            result.update(
                {
                    "sid_input_ids": sid_ids,
                    "sid_attention_mask": sid_enc["attention_mask"],
                    "item_pos": item_pos,
                    "text_input_ids": text_enc["input_ids"],
                    "text_attention_mask": text_enc["attention_mask"],
                    "eos_pos": len(text_enc["input_ids"]) - 1,
                }
            )

        return result
