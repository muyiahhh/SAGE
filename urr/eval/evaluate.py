"""Evaluate URR with constrained MiniOneRec-style beam search."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

from urr.data.minionerec_protocol import (
    RESPONSE_PREFIX,
    build_minionerec_prompt_text,
    encode_text,
    history_sid_string,
    sid_string,
)
from urr.data.views import validate_index_sid_tokens


def get_hash(token_ids: list[int] | Tensor) -> str:
    if isinstance(token_ids, Tensor):
        token_ids = token_ids.tolist()
    return "-".join(str(token_id) for token_id in token_ids)


class SIDConstrainedLogitsProcessor(LogitsProcessor):
    """Restrict every beam to a valid catalog SID prefix."""

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, list[int]], list[int]],
        num_beams: int,
        prefix_length: int,
        eos_token_id: int,
    ):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.num_beams = int(num_beams)
        self.prefix_length = int(prefix_length)
        self.eos_token_id = int(eos_token_id)
        self.generated_count = 0

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))

        # One device-to-host transfer per decoding step avoids a synchronization
        # for every beam while preserving the original prefix-trie behavior.
        grouped = input_ids.view(-1, self.num_beams, input_ids.shape[-1]).tolist()
        for batch_id, beam_group in enumerate(grouped):
            for beam_id, sentence in enumerate(beam_group):
                row = batch_id * self.num_beams + beam_id
                if self.generated_count > 0 and sentence[-1] == self.eos_token_id:
                    mask[row, self.eos_token_id] = 0
                    continue
                if self.generated_count == 0:
                    hash_key = sentence[-self.prefix_length :]
                else:
                    hash_key = sentence[-self.generated_count :]
                allowed = self.prefix_allowed_tokens_fn(batch_id, hash_key)
                mask[row, allowed or [self.eos_token_id]] = 0

        self.generated_count += 1
        return scores + mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_interactions(path: str) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                samples.append(
                    {
                        "user_id": parts[0],
                        "history_ids": parts[1].split(),
                        "target_id": parts[2],
                    }
                )
    return samples


def build_prefix_trie(
    index: dict[str, list[str]],
    tokenizer,
    response_prefix_ids: list[int],
    eos_token_id: int,
) -> dict[str, list[int]]:
    prefix_length = len(response_prefix_ids)
    trie: dict[str, set[int]] = {}
    for sid_tokens in index.values():
        if not sid_tokens:
            continue
        target_ids = encode_text(tokenizer, "".join(sid_tokens) + "\n", bos=False, eos=True)
        full_sequence = list(response_prefix_ids) + target_ids
        for position in range(prefix_length, len(full_sequence)):
            if position == prefix_length:
                key = get_hash(full_sequence[:position])
            else:
                key = get_hash(full_sequence[prefix_length:position])
            trie.setdefault(key, set()).add(int(full_sequence[position]))

    valid_ids = set(tokenizer.get_added_vocab().values()) | {int(eos_token_id)}
    valid_ids.update(encode_text(tokenizer, "\n", bos=False, eos=False))
    filtered = {
        key: sorted(token_ids & valid_ids)
        for key, token_ids in trie.items()
        if token_ids & valid_ids
    }
    return filtered


def decode_prediction(
    token_ids: list[int], tokenizer, eos_token_id: int, newline_ids: set[int]
) -> str:
    sid_ids: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == eos_token_id:
            break
        if token_id in newline_ids or token_id == tokenizer.pad_token_id:
            continue
        sid_ids.append(token_id)
    return "".join(token for token in tokenizer.convert_ids_to_tokens(sid_ids) if token)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    print(f"[INFO] Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_cache=True,
        )
        .to(device)
        .eval()
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    index_path = os.path.join(args.data_dir, f"{args.dataset}.index.json")
    with open(index_path, encoding="utf-8") as handle:
        index: dict[str, list[str]] = json.load(handle)
    validate_index_sid_tokens(index, tokenizer, context=index_path)

    split_path = os.path.join(args.data_dir, f"{args.dataset}.{args.split}.inter")
    samples = load_interactions(split_path)
    if args.n_shards > 1:
        chunk_size = len(samples) // args.n_shards
        start = args.shard * chunk_size
        end = start + chunk_size if args.shard < args.n_shards - 1 else len(samples)
        samples = samples[start:end]
        print(f"[INFO] Shard {args.shard}/{args.n_shards}: samples {start}-{end} ({len(samples)})")
    else:
        print(f"[INFO] {args.split} samples: {len(samples)}")

    response_prefix_ids = encode_text(tokenizer, RESPONSE_PREFIX, bos=False, eos=False)
    prefix_trie = build_prefix_trie(
        index,
        tokenizer,
        response_prefix_ids,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"[INFO] Constraint trie keys: {len(prefix_trie)}")

    def allowed_tokens(_batch_id: int, token_ids: list[int]) -> list[int]:
        return prefix_trie.get(get_hash(token_ids), [])

    generation_config = GenerationConfig(
        num_beams=int(args.num_beams),
        num_return_sequences=int(args.num_beams),
        max_new_tokens=int(args.max_new_tokens),
        length_penalty=float(args.length_penalty),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    newline_ids = set(encode_text(tokenizer, "\n", bos=False, eos=False))
    results: list[dict] = []
    batch_size = int(args.batch_size)
    batch_count = (len(samples) + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, len(samples), batch_size),
        total=batch_count,
        desc="URR evaluation",
    ):
        batch = samples[batch_start : batch_start + batch_size]
        prompts = [
            encode_text(
                tokenizer,
                build_minionerec_prompt_text(
                    sample["history_ids"],
                    index,
                    max_history=int(args.max_history),
                    eval_style=bool(args.eval_prompt_style),
                ),
                bos=True,
                eos=False,
            )
            for sample in batch
        ]
        max_length = max(len(prompt) for prompt in prompts)
        padded = [
            [tokenizer.pad_token_id] * (max_length - len(prompt)) + prompt for prompt in prompts
        ]
        masks = [[0] * (max_length - len(prompt)) + [1] * len(prompt) for prompt in prompts]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        logits_processor = LogitsProcessorList(
            [
                SIDConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn=allowed_tokens,
                    num_beams=int(args.num_beams),
                    prefix_length=len(response_prefix_ids),
                    eos_token_id=tokenizer.eos_token_id,
                )
            ]
        )
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            use_model_defaults=False,
        )
        completions = generated[:, max_length:].detach().cpu().tolist()
        for batch_index, sample in enumerate(batch):
            predictions = [
                decode_prediction(
                    completions[batch_index * int(args.num_beams) + beam_index],
                    tokenizer,
                    eos_token_id=tokenizer.eos_token_id,
                    newline_ids=newline_ids,
                )
                for beam_index in range(int(args.num_beams))
            ]
            results.append(
                {
                    "user_id": sample["user_id"],
                    "input": history_sid_string(
                        sample["history_ids"],
                        index,
                        max_history=int(args.max_history),
                    ),
                    "output": sid_string(index, sample["target_id"]),
                    "predict": predictions,
                }
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[INFO] Results saved to {output_path} ({len(results)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an URR checkpoint")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument(
        "--eval_prompt_style",
        action="store_true",
        help="Use the alternate evaluation wording; train wording is the default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
