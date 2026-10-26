"""Train SAGE predictive semantic anchoring (PSA).

The recommender is initialized from the ISG-grounded LLM and fine-tuned with
next-token cross-entropy plus two readouts of one shared linear predictor g.
g is applied to the four decoding states h_0..h_3, where h_s is the final hidden
state immediately before the target SID token of level s.

* Semantic readout: g(h_s) is aligned with the fixed ISG anchor of the target
  item, ``mean_s(1 - cos(g(h_s), a_target))``.
* Identity readout: g(h_s), normalized, is scored against the normalized
  output-embedding rows of the SID tokens (stop-gradient, temperature kappa)
  and predicts every future code, i.e. state s predicts levels s+1..3.

The two loss weights are not set by hand. ``semantic_grad_ratio`` (rho_sem) and
``identity_grad_ratio`` (rho_id) are target ratios between the gradient of each
readout and the cross-entropy gradient on the decoding states it supervises.
The weights lambda = rho * |grad CE| / |grad readout| are measured once on the
first optimization step and then kept fixed.

g, the anchor bank and the readouts are used only for training. Inference is
plain trie-constrained beam search with the fine-tuned LLM.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from sage.data.minionerec_protocol import MiniOneRecSIDSeqRecDataset
from sage.data.views import jepa_collate_fn
from sage.models.isg import SID_LEVELS, build_sid_tokens, unfreeze_backbone
from sage.models.predictor import PSAPredictor
from sage.utils.checkpoint import staged_checkpoint


logger = logging.getLogger(__name__)

NUM_CODES = 256
READOUTS = ("semantic", "identity")


DEFAULT_CFG = {
    "grounded_model": "outputs/qwen3_0.6b_beauty/isg/grounded_model",
    "anchor_cache": "outputs/qwen3_0.6b_beauty/anchors/item_anchors.pt",
    "dataset": "Beauty",
    "data_dir": "data/Beauty",
    "output_dir": "outputs/qwen3_0.6b_beauty/psa",
    # Gradient-ratio targets (rho), not additive loss coefficients.
    # Setting both to 0 gives plain cross-entropy fine-tuning.
    "semantic_grad_ratio": 0.010,
    "identity_grad_ratio": 0.030,
    "identity_temperature": 0.07,  # kappa
    "max_aux_weight": 10.0,
    "psa_lr_ratio": 1.0,
    "batch_size": 128,  # global batch
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "num_epochs": 3,
    "scheduler_num_epochs": 4,
    "save_epochs": [3],
    "warmup_ratio": 0.05,
    "max_seq_len": 512,
    "max_history": 20,
    "log_steps": 50,
    "bf16": True,
    "grad_clip": 1.0,
    "gradient_checkpointing": False,
    "dataloader_num_workers": 4,
    "seed": 42,
}

_FLOAT_KEYS = [
    "semantic_grad_ratio",
    "identity_grad_ratio",
    "identity_temperature",
    "max_aux_weight",
    "psa_lr_ratio",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "grad_clip",
]
_INT_KEYS = [
    "batch_size",
    "gradient_accumulation_steps",
    "num_epochs",
    "scheduler_num_epochs",
    "max_seq_len",
    "max_history",
    "log_steps",
    "dataloader_num_workers",
    "seed",
]


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    unknown = sorted(set(user_cfg).difference(DEFAULT_CFG))
    if unknown:
        raise ValueError(f"Unknown PSA config keys: {unknown}")
    cfg = {**DEFAULT_CFG, **user_cfg, **(overrides or {})}
    # PyYAML parses values such as ``2e-4`` as strings.
    for key in _FLOAT_KEYS:
        cfg[key] = float(cfg[key])
    for key in _INT_KEYS:
        cfg[key] = int(cfg[key])
    cfg["save_epochs"] = [int(epoch) for epoch in cfg["save_epochs"]]
    if cfg["semantic_grad_ratio"] < 0 or cfg["identity_grad_ratio"] < 0:
        raise ValueError("Gradient ratios must be non-negative")
    if cfg["identity_temperature"] <= 0:
        raise ValueError("identity_temperature must be positive")
    if cfg["scheduler_num_epochs"] < cfg["num_epochs"]:
        raise ValueError("scheduler_num_epochs must be >= num_epochs")
    if cfg["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────


class PSADataset(MiniOneRecSIDSeqRecDataset):
    """Recommendation samples with the anchor-bank row of their target item."""

    def __init__(self, *, item_to_anchor_index: dict[str, int], **kwargs):
        super().__init__(include_jepa_views=False, **kwargs)
        self.item_to_anchor_index = item_to_anchor_index

    def __getitem__(self, index: int) -> dict:
        result = super().__getitem__(index)
        target_id = str(self.samples[index]["target_id"])
        if target_id not in self.item_to_anchor_index:
            raise KeyError(f"Target item {target_id!r} is missing from the anchor bank")
        result["target_index"] = self.item_to_anchor_index[target_id]
        return result


def psa_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    result = jepa_collate_fn(batch, pad_id=pad_id)
    result["target_index"] = torch.tensor(
        [sample["target_index"] for sample in batch], dtype=torch.long
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


class SAGEPSA(nn.Module):
    """LLM with the shared PSA predictor and its two readouts."""

    def __init__(
        self,
        language_model: nn.Module,
        predictor: PSAPredictor | None,
        level_token_ids: torch.Tensor,
        *,
        identity_temperature: float,
    ):
        super().__init__()
        self.language_model = language_model
        self.predictor = predictor
        self.identity_temperature = float(identity_temperature)
        num_levels = len(SID_LEVELS)
        self.register_buffer("level_token_ids", level_token_ids.long())
        # future_terms[s, l]: the identity readout at state s predicts level l > s.
        levels = torch.arange(num_levels)
        self.register_buffer("future_terms", levels[None, :].gt(levels[:, None]), persistent=False)

    def _sid_states(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        first_target: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return h_0..h_3, the states immediately before the four SID tokens."""
        offsets = torch.arange(len(SID_LEVELS), device=hidden.device)
        token_positions = first_target[:, None] + offsets[None, :]
        if int(token_positions.max().item()) >= labels.size(1):
            raise ValueError("A target sequence ended before all four SID levels")

        observed = labels.gather(1, token_positions)
        expected = self.level_token_ids.gather(1, target_codes.t()).t()
        if not torch.equal(observed, expected):
            mismatches = int(observed.ne(expected).sum().item())
            raise ValueError(
                f"The first four supervised tokens must be the target SID; "
                f"found {mismatches} mismatches"
            )

        state_positions = token_positions.sub(1).clamp_min(0)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)[:, None]
        return hidden[batch_index, state_positions], state_positions

    def _code_embeddings(self) -> torch.Tensor:
        """Normalized output-embedding rows of the SID tokens, (levels, 256, D)."""
        output_head = self.language_model.get_output_embeddings()
        if output_head is None:
            output_head = self.language_model.get_input_embeddings()
        rows = output_head.weight.index_select(0, self.level_token_ids.flatten())
        rows = rows.view(len(SID_LEVELS), NUM_CODES, -1).detach().float()
        return F.normalize(rows, dim=-1)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        target_index: torch.Tensor,
        anchor_bank: torch.Tensor,
        sid_codes: torch.Tensor,
        calibrate: bool = False,
    ) -> dict:
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        result = {"ce_loss": output.loss, "metrics": {}}
        if self.predictor is None:
            return result

        # Final-layer states, i.e. the inputs of the LM head.
        hidden = output.hidden_states[-1]
        supervised = labels.ne(-100)
        if not bool(supervised.any(dim=1).all()):
            raise ValueError("Every sample must contain supervised target tokens")
        first_target = supervised.long().argmax(dim=1)
        target_codes = sid_codes.index_select(0, target_index)  # (B, 4), zero-based
        states, state_positions = self._sid_states(hidden, labels, first_target, target_codes)
        prediction = self.predictor(states).float()  # g(h_s), (B, 4, D)

        # Semantic readout: align g(h_s) with the ISG anchor of the target item.
        anchors = anchor_bank.index_select(0, target_index)  # unit norm, (B, D)
        semantic_cos = F.cosine_similarity(prediction, anchors[:, None, :], dim=-1)
        semantic_loss = (1.0 - semantic_cos).mean()

        # Identity readout: g(h_s) predicts the codes of all later levels.
        query = F.normalize(prediction, dim=-1)
        logits = torch.einsum("bsd,kvd->bskv", query, self._code_embeddings())
        logits = logits / self.identity_temperature  # (B, state, level, 256)
        code_labels = target_codes[:, None, :].expand(-1, len(SID_LEVELS), -1)
        per_code_loss = F.cross_entropy(
            logits.reshape(-1, NUM_CODES),
            code_labels.reshape(-1),
            reduction="none",
        ).view(code_labels.shape)
        future = self.future_terms.to(per_code_loss.dtype)
        identity_loss = ((per_code_loss * future).sum(dim=(1, 2)) / future.sum()).mean()

        correct = logits.argmax(dim=-1).eq(code_labels).float()
        result.update(
            {
                "semantic_loss": semantic_loss,
                "identity_loss": identity_loss,
                "metrics": {
                    "semantic_cos": semantic_cos.mean().detach(),
                    "identity_acc": (
                        (correct * future).sum() / (future.sum() * correct.size(0))
                    ).detach(),
                },
            }
        )

        if calibrate:
            # Gradient norms on the decoding states each readout supervises:
            # all four states for the semantic readout, states 0..2 for the
            # identity readout (state 3 has no future level).
            ce_grad = torch.autograd.grad(output.loss, hidden, retain_graph=True)[0].float()
            supervised_states = {
                "semantic": state_positions,
                "identity": state_positions[:, self.future_terms.any(dim=1)],
            }
            for name in READOUTS:
                aux_grad = torch.autograd.grad(result[f"{name}_loss"], hidden, retain_graph=True)[
                    0
                ].float()
                positions = supervised_states[name]
                owners = torch.arange(hidden.size(0), device=hidden.device)[:, None]
                owners = owners.expand_as(positions)
                ce_part = ce_grad[owners, positions]
                aux_part = aux_grad[owners, positions]
                ce_norm = ce_part.norm().clamp_min(1e-12)
                aux_norm = aux_part.norm().clamp_min(1e-12)
                result[f"{name}_ce_grad_norm"] = ce_norm.detach()
                result[f"{name}_grad_norm"] = aux_norm.detach()
                result[f"{name}_grad_cos"] = (
                    (ce_part * aux_part).sum() / (ce_norm * aux_norm)
                ).detach()
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _distributed_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().clone()
    if world_size > 1:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result.div_(world_size)
    return result


def _level_token_ids(tokenizer) -> torch.Tensor:
    rows: list[list[int]] = []
    unknown_id = tokenizer.unk_token_id
    for level in SID_LEVELS:
        ids = [
            tokenizer.convert_tokens_to_ids(f"<|{level}_{code}|>")
            for code in range(1, NUM_CODES + 1)
        ]
        if any(token_id is None or token_id == unknown_id for token_id in ids):
            raise ValueError(f"Tokenizer is missing at least one level-{level} SID token")
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _save_checkpoint(
    module: SAGEPSA,
    tokenizer,
    save_dir: Path,
    cfg: dict,
    loss_weights: dict[str, float],
) -> None:
    with staged_checkpoint(save_dir, prefix="sage_psa_") as tmp_dir:
        module.language_model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        torch.save(
            {
                "cfg": cfg,
                "loss_weights": loss_weights,
                "predictor": (
                    module.predictor.state_dict() if module.predictor is not None else None
                ),
            },
            tmp_dir / "psa_heads.pt",
        )
        with open(tmp_dir / "training_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump({**cfg, "loss_weights": loss_weights}, handle, sort_keys=False)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.info("SAGE PSA config: %s", cfg)

    # ── ISG-grounded LLM ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["grounded_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["grounded_model"],
        dtype=dtype,
        trust_remote_code=True,
        use_cache=False,
    )
    missing = [token for token in build_sid_tokens() if token not in tokenizer.get_vocab()]
    if missing:
        raise ValueError(f"ISG checkpoint is missing {len(missing)} SID tokens")
    unfreeze_backbone(model)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ratios = {
        "semantic": cfg["semantic_grad_ratio"],
        "identity": cfg["identity_grad_ratio"],
    }
    use_psa = any(ratio > 0 for ratio in ratios.values())
    predictor = PSAPredictor(int(model.config.hidden_size)) if use_psa else None
    module = SAGEPSA(
        model,
        predictor,
        _level_token_ids(tokenizer),
        identity_temperature=cfg["identity_temperature"],
    ).to(device=device, dtype=dtype)

    # ── Fixed ISG anchor bank ────────────────────────────────────────────────
    cache = torch.load(cfg["anchor_cache"], map_location="cpu")
    missing_keys = {"item_ids", "sid_codes", "sid_hidden"}.difference(cache)
    if missing_keys:
        raise KeyError(f"Anchor cache is missing keys: {sorted(missing_keys)}")
    item_ids = [str(item_id) for item_id in cache["item_ids"]]
    item_to_anchor_index = {item_id: index for index, item_id in enumerate(item_ids)}
    sid_codes = cache["sid_codes"].to(device=device, dtype=torch.long)
    anchor_bank = F.normalize(cache["sid_hidden"].float(), dim=-1).to(device)
    if anchor_bank.size(1) != int(model.config.hidden_size):
        raise ValueError(
            f"Anchor dimension {anchor_bank.size(1)} does not match model hidden "
            f"size {model.config.hidden_size}"
        )

    if world_size > 1:
        module = nn.parallel.DistributedDataParallel(
            module,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    raw_module = module.module if world_size > 1 else module

    # ── Optimizer ────────────────────────────────────────────────────────────
    backbone_params = [p for p in raw_module.language_model.parameters() if p.requires_grad]
    predictor_params = list(raw_module.predictor.parameters()) if use_psa else []
    parameter_groups = [{"params": backbone_params, "lr": cfg["learning_rate"]}]
    if predictor_params:
        parameter_groups.append(
            {"params": predictor_params, "lr": cfg["learning_rate"] * cfg["psa_lr_ratio"]}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg["weight_decay"])

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = PSADataset(
        item_to_anchor_index=item_to_anchor_index,
        data_dir=cfg["data_dir"],
        dataset=cfg["dataset"],
        split="train",
        tokenizer=tokenizer,
        max_seq_len=cfg["max_seq_len"],
        max_history=cfg["max_history"],
        eval_prompt_style_for_sft=False,
    )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=cfg["seed"])
        if world_size > 1
        else None
    )
    accum = cfg["gradient_accumulation_steps"]
    if cfg["batch_size"] % (world_size * accum):
        raise ValueError("batch_size must be divisible by world_size * gradient_accumulation_steps")
    per_device_batch = cfg["batch_size"] // (world_size * accum)
    pad_id = tokenizer.pad_token_id or 0
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=lambda samples: psa_collate(samples, pad_id=pad_id),
        drop_last=True,
    )
    usable_microsteps = (len(loader) // accum) * accum
    if not usable_microsteps:
        raise ValueError("Dataset too small for one global batch")
    # The cosine schedule spans scheduler_num_epochs; training stops after num_epochs.
    total_steps = (usable_microsteps // accum) * cfg["scheduler_num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    if is_main:
        logger.info(
            "Samples=%d batch=%d per device x world=%d x accumulation=%d "
            "train_epochs=%d scheduler_epochs=%d",
            len(dataset),
            per_device_batch,
            world_size,
            accum,
            cfg["num_epochs"],
            cfg["scheduler_num_epochs"],
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_weights: dict[str, float] = {name: 0.0 for name in READOUTS}
    calibrated = not use_psa
    global_step = 0
    module.train()

    for epoch in range(cfg["num_epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        totals: dict[str, float] = {}
        progress = tqdm(
            loader,
            desc=f"SAGE PSA epoch {epoch + 1}/{cfg['num_epochs']}",
            disable=not is_main,
            dynamic_ncols=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(progress):
            if micro_step >= usable_microsteps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            boundary = (micro_step + 1) % accum == 0
            sync = nullcontext() if world_size == 1 or boundary else module.no_sync()
            with sync:
                logs = module(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    target_index=batch["target_index"],
                    anchor_bank=anchor_bank,
                    sid_codes=sid_codes,
                    calibrate=not calibrated,
                )

                if not calibrated:
                    for name in READOUTS:
                        if ratios[name] <= 0:
                            continue
                        ce_norm = _distributed_mean(logs[f"{name}_ce_grad_norm"], world_size)
                        aux_norm = _distributed_mean(logs[f"{name}_grad_norm"], world_size)
                        weight = ratios[name] * ce_norm / aux_norm.clamp_min(1e-12)
                        loss_weights[name] = float(weight.clamp(max=cfg["max_aux_weight"]).item())
                        if is_main:
                            logger.info(
                                "Calibrated %s readout: rho=%.4f lambda=%.6f "
                                "grad_cos=%.4f raw_grad_ratio=%.4f",
                                name,
                                ratios[name],
                                loss_weights[name],
                                logs[f"{name}_grad_cos"].item(),
                                (aux_norm / ce_norm.clamp_min(1e-12)).item(),
                            )
                    calibrated = True

                loss = logs["ce_loss"]
                if use_psa:
                    loss = (
                        loss
                        + loss_weights["semantic"] * logs["semantic_loss"]
                        + loss_weights["identity"] * logs["identity_loss"]
                    )
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("Non-finite training loss")
                (loss / accum).backward()

            if boundary:
                if cfg["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(backbone_params + predictor_params, cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            values = {"ce": logs["ce_loss"]}
            if use_psa:
                values.update(
                    semantic=logs["semantic_loss"],
                    identity=logs["identity_loss"],
                    **logs["metrics"],
                )
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.item())
            progress.set_postfix(
                {
                    name: f"{values[name].item():.4f}"
                    for name in ("ce", "semantic", "identity")
                    if name in values
                }
            )
            if is_main and boundary and global_step % cfg["log_steps"] == 0:
                logger.info(
                    "epoch=%d step=%d %s lr=%.2e",
                    epoch + 1,
                    global_step,
                    " ".join(f"{name}={value.item():.4f}" for name, value in values.items()),
                    scheduler.get_last_lr()[0],
                )
        progress.close()

        if is_main:
            count = max(usable_microsteps, 1)
            logger.info(
                "Epoch %d done | %s | lambda_sem=%.6f lambda_id=%.6f",
                epoch + 1,
                " ".join(f"{name}={total / count:.4f}" for name, total in totals.items()),
                loss_weights["semantic"],
                loss_weights["identity"],
            )
            epoch_number = epoch + 1
            if epoch_number in cfg["save_epochs"]:
                _save_checkpoint(
                    raw_module,
                    tokenizer,
                    output_dir / f"epoch_{epoch_number}",
                    cfg,
                    loss_weights,
                )
        if world_size > 1:
            torch.distributed.barrier()

    if is_main:
        logger.info("SAGE PSA complete -> %s", output_dir)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE predictive semantic anchoring")
    parser.add_argument("--config", required=True)
    parser.add_argument("--grounded_model")
    parser.add_argument("--anchor_cache")
    parser.add_argument("--dataset")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    overrides = {
        key: value for key, value in vars(args).items() if key != "config" and value is not None
    }
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
