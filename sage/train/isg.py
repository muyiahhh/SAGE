"""Train SAGE item-level semantic grounding (ISG).

Trains:
  - the newly added SID embedding rows (weight decay 0)
  - the linear SID-to-text predictor

Keeps frozen:
  - all original LLM parameters, which also encode the text view

Output: ``grounded_model/`` containing
  - the LLM with grounded SID embeddings and its tokenizer
  - ``isg_heads.pt`` with the predictor state (not used after ISG)

Usage:
  python -m sage.train.isg --config configs/beauty_qwen3_0.6b_isg.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from sage.data.views import ISGViewDataset, jepa_collate_fn
from sage.models.isg import (
    ITEM_MARKER,
    ISGAligner,
    add_sid_tokens_to_model,
    build_sid_tokens,
    freeze_backbone,
)
from sage.utils.checkpoint import staged_checkpoint

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "model_name": "Qwen/Qwen3-0.6B",
    "dataset": "Beauty",
    "data_dir": "data/Beauty",
    "output_dir": "outputs/qwen3_0.6b_beauty/isg",
    "predictor_type": "linear",  # linear (paper) | identity | mlp2 | mlp3
    "sid_init_strategy": "mean",  # mean (paper) | normal
    "batch_size": 256,
    "learning_rate": 1.5e-3,
    "weight_decay": 0.01,  # predictor only; SID embedding rows use 0
    "num_epochs": 10,
    "warmup_ratio": 0.05,
    "max_text_len": 128,
    "max_sid_len": 32,
    "log_steps": 50,
    "bf16": True,
    "grad_clip": 1.0,
    "dataloader_num_workers": 4,
    "seed": 42,
    "save_each_epoch": False,
}

_FLOAT_KEYS = ["learning_rate", "weight_decay", "warmup_ratio", "grad_clip"]
_INT_KEYS = [
    "batch_size",
    "num_epochs",
    "max_text_len",
    "max_sid_len",
    "log_steps",
    "dataloader_num_workers",
    "seed",
]


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    user_cfg = {}
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            user_cfg = yaml.safe_load(handle) or {}
    unknown = sorted(set(user_cfg).difference(DEFAULT_CFG))
    if unknown:
        raise ValueError(f"Unknown ISG config keys: {unknown}")
    cfg = {**DEFAULT_CFG, **user_cfg, **(overrides or {})}
    # PyYAML parses values such as ``1e-4`` as strings.
    for key in _FLOAT_KEYS:
        cfg[key] = float(cfg[key])
    for key in _INT_KEYS:
        cfg[key] = int(cfg[key])
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────


def train_isg(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        logger.info("SAGE ISG config: %s", cfg)

    # ── Base model + SID vocabulary ──────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=dtype,
        trust_remote_code=True,
        use_cache=False,
    )
    model, tokenizer, orig_vocab_size = add_sid_tokens_to_model(
        model,
        tokenizer,
        build_sid_tokens(),  # 4 SID levels + collision tokens + [ITEM]
        init_strategy=cfg["sid_init_strategy"],
    )
    freeze_backbone(model, orig_vocab_size)
    if is_main:
        logger.info(
            "Vocabulary extended: %d -> %d (+%d SID tokens)",
            orig_vocab_size,
            len(tokenizer),
            len(tokenizer) - orig_vocab_size,
        )

    aligner = ISGAligner(
        language_model=model,
        hidden_size=int(model.config.hidden_size),
        predictor_type=cfg["predictor_type"],
    ).to(device=device, dtype=dtype)

    if world_size > 1:
        aligner = nn.parallel.DistributedDataParallel(
            aligner, device_ids=[local_rank], find_unused_parameters=False
        )
    raw_aligner = aligner.module if world_size > 1 else aligner

    # ── Optimizer: SID embedding rows (wd 0) and predictor (wd > 0) ──────────
    embedding_params = [p for p in raw_aligner.language_model.parameters() if p.requires_grad]
    predictor_params = list(raw_aligner.predictor.parameters())
    trainable = embedding_params + predictor_params
    parameter_groups = [
        {"params": embedding_params, "lr": cfg["learning_rate"], "weight_decay": 0.0}
    ]
    if predictor_params:
        parameter_groups.append(
            {
                "params": predictor_params,
                "lr": cfg["learning_rate"],
                "weight_decay": cfg["weight_decay"],
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    if is_main:
        logger.info(
            "Trainable: SID embedding rows=%d predictor params=%.2fM",
            len(tokenizer) - orig_vocab_size,
            sum(p.numel() for p in predictor_params) / 1e6,
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset = ISGViewDataset(
        data_dir=cfg["data_dir"],
        dataset=cfg["dataset"],
        tokenizer=tokenizer,
        item_marker=ITEM_MARKER,
        max_text_len=cfg["max_text_len"],
        max_sid_len=cfg["max_sid_len"],
    )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        if world_size > 1
        else None
    )
    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"] // max(world_size, 1),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=lambda b: jepa_collate_fn(b, pad_id=pad_id),
        drop_last=True,
    )
    if is_main:
        logger.info("Items: %d", len(train_dataset))

    total_steps = len(train_loader) * cfg["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    aligner.train()

    for epoch in range(cfg["num_epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"ISG epoch {epoch + 1}/{cfg['num_epochs']}",
            disable=not is_main,
            dynamic_ncols=True,
        )
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logs = aligner(
                sid_input_ids=batch["sid_input_ids"],
                sid_attention_mask=batch["sid_attention_mask"],
                item_pos=batch["item_pos"],
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                eos_pos=batch["eos_pos"],
            )
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Non-finite ISG loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            global_step += 1
            lr_now = scheduler.get_last_lr()[0]
            progress.set_postfix(isg=f"{logs['isg_loss'].item():.4f}", lr=f"{lr_now:.2e}")
            if is_main and global_step % cfg["log_steps"] == 0:
                logger.info(
                    "epoch=%d step=%d isg_loss=%.4f lr=%.2e",
                    epoch + 1,
                    global_step,
                    logs["isg_loss"].item(),
                    lr_now,
                )
        progress.close()

        if is_main:
            logger.info("Epoch %d done | isg_loss=%.4f", epoch + 1, epoch_loss / len(train_loader))
            if cfg["save_each_epoch"]:
                _save_checkpoint(raw_aligner, tokenizer, output_dir / f"epoch_{epoch + 1}", cfg)

    # ── Grounded LLM used to initialize recommendation fine-tuning ───────────
    if is_main:
        _save_checkpoint(raw_aligner, tokenizer, output_dir / "grounded_model", cfg)
        logger.info("ISG complete -> %s", output_dir / "grounded_model")

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _save_checkpoint(aligner: ISGAligner, tokenizer, save_dir: Path, cfg: dict) -> None:
    with staged_checkpoint(save_dir, prefix="sage_isg_") as tmp_dir:
        aligner.language_model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        torch.save(
            {"predictor": aligner.predictor.state_dict(), "cfg": cfg},
            tmp_dir / "isg_heads.pt",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE item-level semantic grounding")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    train_isg(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
