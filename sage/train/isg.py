"""Train SAGE item-level semantic grounding (ISG).

Trains:
  - SID embedding rows (new tokens only)
  - optional context projection
  - representation predictor

Keeps frozen:
  - All original LLM backbone weights

EMA updates the target encoder after every optimizer step.

Output: grounded_model/ checkpoint directory containing:
  - model weights (LLM + grounded SID embeddings)
  - tokenizer
  - isg_heads.pt (projection and predictor states)

Usage:
  python -m sage.train.isg --config configs/beauty_qwen3_0.6b_isg.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import yaml

from sage.models.isg import (
    ISGAligner,
    add_sid_tokens_to_model,
    build_sid_tokens,
    freeze_backbone,
    ITEM_MARKER,
)
from sage.data.views import ISGViewDataset, jepa_collate_fn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "model_name": "Qwen/Qwen3-0.6B",
    "dataset": "Beauty",
    "data_dir": "data/Beauty",
    "output_dir": "outputs/qwen3_0.6b_beauty/isg",
    "d_proj": 1024,
    "projection_type": "identity",  # linear | identity (raw hidden space)
    "ema_tau": 0.999,
    "predictor_type": "linear",  # identity | linear | mlp2 | mlp3
    "jepa_loss_type": "cosine",  # mse | cosine | norm_mse
    "sid_init_strategy": "mean",  # normal | mean
    "batch_size": 256,
    "learning_rate": 1.5e-3,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_ratio": 0.05,
    "max_text_len": 128,
    "max_sid_len": 16,
    "use_var_reg": False,
    "var_reg_weight": 0.1,
    "log_steps": 50,
    "bf16": True,
    "grad_clip": 1.0,
    "dataloader_num_workers": 4,
    "seed": 42,
    # Full-model checkpoints are large. Keep the legacy per-epoch behavior by
    # default, but allow large sweeps to retain only aligned_base_model.
    "save_each_epoch": False,
}


def _coerce_numeric(cfg: dict) -> dict:
    """Coerce numeric config values that YAML may parse as strings.

    PyYAML parses ``1e-4`` (no decimal point) as a *string*, so cast the known
    numeric keys explicitly to be robust to how the config was written.
    """
    float_keys = [
        "learning_rate",
        "weight_decay",
        "ema_tau",
        "warmup_ratio",
        "grad_clip",
        "var_reg_weight",
        "lambda_jepa",
    ]
    int_keys = [
        "d_proj",
        "batch_size",
        "num_epochs",
        "max_text_len",
        "max_sid_len",
        "max_seq_len",
        "max_history",
        "log_steps",
        "dataloader_num_workers",
        "seed",
    ]
    for k in float_keys:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    for k in int_keys:
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])
    return cfg


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    merged = {**DEFAULT_CFG, **(cfg or {})}
    cfg = _coerce_numeric(merged)
    if cfg["projection_type"] not in {"linear", "identity"}:
        raise ValueError("projection_type must be 'linear' or 'identity'")
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────


def train_isg(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])

    # ── Device setup ─────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        logger.info("SAGE ISG config: %s", cfg)

    # ── Load base model + tokenizer ───────────────────────────────────────────
    if is_main:
        logger.info("Loading base model: %s", cfg["model_name"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
        trust_remote_code=True,
        use_cache=False,
    )

    # ── Add SID tokens + [ITEM] marker ───────────────────────────────────────
    sid_tokens = build_sid_tokens()  # 4 SID levels + dedup tokens + [ITEM]
    orig_vocab_size = len(tokenizer)
    model, tokenizer, orig_vocab_size = add_sid_tokens_to_model(
        model,
        tokenizer,
        sid_tokens,
        init_strategy=cfg.get("sid_init_strategy", "normal"),
    )

    if is_main:
        logger.info(
            "Vocabulary extended: %d → %d (+%d SID tokens)",
            orig_vocab_size,
            len(tokenizer),
            len(tokenizer) - orig_vocab_size,
        )

    # ── Freeze backbone (only SID embedding rows are trainable) ──────────────
    freeze_backbone(model, orig_vocab_size)

    d_model = model.config.hidden_size
    if is_main:
        logger.info("LLM d_model = %d, d_proj = %d", d_model, cfg["d_proj"])

    # ── Build the ISG online and EMA target branches ──────────────────────────
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32
    jepa = ISGAligner(
        context_model=model,
        d_model=d_model,
        d_proj=cfg["d_proj"],
        ema_tau=cfg["ema_tau"],
        predictor_type=cfg.get("predictor_type", "mlp3"),
        jepa_loss_type=cfg.get("jepa_loss_type", "mse"),
        projection_type=cfg.get("projection_type", "linear"),
    )
    jepa = jepa.to(device=device, dtype=dtype)

    if world_size > 1:
        jepa = nn.parallel.DistributedDataParallel(
            jepa, device_ids=[local_rank], find_unused_parameters=False
        )

    # ── Optimizer: only trainable params ─────────────────────────────────────
    def get_trainable_params(module):
        return [p for p in module.parameters() if p.requires_grad]

    raw_jepa = jepa.module if world_size > 1 else jepa
    embedding_params = get_trainable_params(raw_jepa.context_encoder)
    head_params = list(raw_jepa.context_proj.parameters()) + list(raw_jepa.predictor.parameters())
    trainable = embedding_params + head_params

    if is_main:
        total = sum(p.numel() for p in trainable)
        logger.info(
            "Trainable params: %.2fM | SID embedding group weight_decay=0 | "
            "ISG head group weight_decay=%s",
            total / 1e6,
            cfg["weight_decay"],
        )

    parameter_groups = [
        {
            "params": embedding_params,
            "lr": cfg["learning_rate"],
            "weight_decay": 0.0,
        }
    ]
    if head_params:
        parameter_groups.append(
            {
                "params": head_params,
                "lr": cfg["learning_rate"],
                "weight_decay": cfg["weight_decay"],
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset = ISGViewDataset(
        data_dir=cfg["data_dir"],
        dataset=cfg["dataset"],
        tokenizer=tokenizer,
        item_marker=ITEM_MARKER,
        max_text_len=cfg["max_text_len"],
        max_sid_len=cfg["max_sid_len"],
    )

    if is_main:
        logger.info("Dataset size: %d items", len(train_dataset))

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

    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    jepa.train()

    for epoch in range(cfg["num_epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg['num_epochs']}",
            disable=not is_main,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, log_dict = jepa(
                sid_input_ids=batch["sid_input_ids"],
                sid_attention_mask=batch["sid_attention_mask"],
                item_pos=batch["item_pos"],
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                eos_pos=batch["eos_pos"],
                use_var_reg=cfg["use_var_reg"],
                var_reg_weight=cfg["var_reg_weight"],
            )

            optimizer.zero_grad()
            loss.backward()

            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])

            optimizer.step()
            scheduler.step()

            # EMA update (on the raw module regardless of DDP wrapper)
            raw_jepa.update_ema()

            epoch_loss += loss.item()
            global_step += 1

            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()} | {"lr": f"{lr_now:.2e}"})

            if is_main and global_step % cfg["log_steps"] == 0:
                log_str = " | ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
                logger.info(
                    "Epoch %d step %d | %s | lr=%.2e",
                    epoch + 1,
                    global_step,
                    log_str,
                    lr_now,
                )

        pbar.close()

        if is_main:
            avg_loss = epoch_loss / len(train_loader)
            logger.info("Epoch %d done — avg_loss=%.4f", epoch + 1, avg_loss)
            if cfg.get("save_each_epoch", True):
                _save_checkpoint(raw_jepa, tokenizer, output_dir / f"epoch_{epoch + 1}", cfg)
                logger.info("Checkpoint saved → epoch_%d/", epoch + 1)

    # ── Save the grounded LLM used to initialize GSA ─────────────────────────
    if is_main:
        _save_checkpoint(raw_jepa, tokenizer, output_dir / "grounded_model", cfg)
        logger.info("ISG complete. grounded_model saved to %s", output_dir)

    if world_size > 1:
        torch.distributed.destroy_process_group()


def _save_checkpoint(
    jepa: ISGAligner,
    tokenizer,
    save_dir: Path,
    cfg: dict,
) -> None:
    """Save the grounded model, tokenizer, and ISG heads.

    Writes to a local /tmp staging area first, then copies to the final
    (possibly network) path to avoid EIO on NFS/ossfs under concurrent load.
    """
    save_dir = Path(save_dir)

    with tempfile.TemporaryDirectory(prefix="sage_isg_", dir="/tmp") as tmp_root:
        tmp_dir = Path(tmp_root) / "ckpt"
        tmp_dir.mkdir()

        jepa.context_encoder.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        torch.save(
            {
                "context_proj": jepa.context_proj.state_dict(),
                "target_proj": jepa.target_proj.state_dict(),
                "predictor": jepa.predictor.state_dict(),
                "cfg": cfg,
            },
            tmp_dir / "isg_heads.pt",
        )

        _publish_checkpoint(tmp_dir, save_dir)


def _publish_checkpoint(tmp_dir: Path, save_dir: Path) -> None:
    """Publish a complete checkpoint without exposing a partial final path."""
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = save_dir.with_name(f".{save_dir.name}.partial-{os.getpid()}")
    shutil.rmtree(partial_dir, ignore_errors=True)
    try:
        shutil.copytree(tmp_dir, partial_dir)
        if save_dir.exists():
            shutil.rmtree(save_dir)
        partial_dir.replace(save_dir)
    except BaseException:
        # Quota/NFS failures used to leave multi-GB directories that looked
        # checkpoint-like. Removing the staging path keeps retries reliable.
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE item-level semantic grounding")
    parser.add_argument("--config", type=str, default="configs/beauty_qwen3_0.6b_isg.yaml")
    # Allow overriding any config key from CLI
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--d_proj", type=int, default=None)
    parser.add_argument("--use_var_reg", action="store_true", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else dict(DEFAULT_CFG)

    # CLI overrides
    for key in [
        "model_name",
        "dataset",
        "data_dir",
        "output_dir",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "d_proj",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.use_var_reg:
        cfg["use_var_reg"] = True

    train_isg(cfg)


if __name__ == "__main__":
    main()
