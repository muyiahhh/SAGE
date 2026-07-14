"""URR Stage 1b: generative recommendation with point and plan objectives.

The language-model cross-entropy objective is augmented by two signals:

* point: the history state predicts the fixed Stage 1a SID representation of
  the next item with cosine distance;
* plan: each state before a target SID token predicts the current and remaining
  RQ-SID levels in the model's output-embedding space.

The user-facing point/plan values are desired hidden-gradient ratios relative
to cross entropy.  Their concrete loss weights are calibrated on the first
optimization step, which keeps the setting comparable across model scales.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from urr.data.minionerec_protocol import MiniOneRecSIDSeqRecDataset
from urr.data.views import jepa_collate_fn
from urr.models.jepa import SID_LEVELS, build_sid_tokens, unfreeze_backbone


logger = logging.getLogger(__name__)


DEFAULT_CFG = {
    "aligned_base_model": "outputs/qwen3_0.6b_beauty/stage1a/aligned_base_model",
    "target_cache": "outputs/qwen3_0.6b_beauty/targets/stage1b_targets.pt",
    "dataset": "Beauty",
    "data_dir": "data/Beauty",
    "output_dir": "outputs/qwen3_0.6b_beauty/stage1b",
    "point_grad_ratio": 0.015,
    "plan_grad_ratio": 0.03,
    "plan_temperature": 0.07,
    "plan_scope": "remaining",
    "plan_level_weights": [1.0, 1.0, 1.0, 1.0],
    "plan_state_weights": [1.0, 1.0, 1.0, 1.0],
    "max_aux_weight": 10.0,
    "plan_lr_ratio": 1.0,
    "batch_size": 128,
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
    "diagnose_first_step": True,
}


def _coerce_config(cfg: dict) -> dict:
    for key in [
        "point_grad_ratio",
        "plan_grad_ratio",
        "plan_temperature",
        "max_aux_weight",
        "plan_lr_ratio",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "grad_clip",
    ]:
        cfg[key] = float(cfg[key])
    for key in [
        "batch_size",
        "num_epochs",
        "scheduler_num_epochs",
        "max_seq_len",
        "max_history",
        "log_steps",
        "dataloader_num_workers",
        "seed",
    ]:
        cfg[key] = int(cfg[key])
    cfg["save_epochs"] = [int(epoch) for epoch in cfg.get("save_epochs", [3])]
    cfg["plan_level_weights"] = [float(value) for value in cfg["plan_level_weights"]]
    cfg["plan_state_weights"] = [float(value) for value in cfg["plan_state_weights"]]
    return cfg


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    cfg = {**DEFAULT_CFG, **user_cfg, **(overrides or {})}
    cfg = _coerce_config(cfg)
    if cfg["plan_scope"] != "remaining":
        raise ValueError("The released URR method uses plan_scope='remaining'")
    if cfg["point_grad_ratio"] <= 0 or cfg["plan_grad_ratio"] <= 0:
        raise ValueError("Both point_grad_ratio and plan_grad_ratio must be positive")
    if cfg["scheduler_num_epochs"] < cfg["num_epochs"]:
        raise ValueError("scheduler_num_epochs must be >= num_epochs")
    if len(cfg["plan_level_weights"]) != len(SID_LEVELS):
        raise ValueError("plan_level_weights must contain four values")
    if len(cfg["plan_state_weights"]) != len(SID_LEVELS):
        raise ValueError("plan_state_weights must contain four values")
    if any(value <= 0 for value in cfg["plan_level_weights"] + cfg["plan_state_weights"]):
        raise ValueError("All plan weights must be positive")
    return cfg


class Stage1bDataset(MiniOneRecSIDSeqRecDataset):
    def __init__(self, *, item_to_cache_index: dict[str, int], **kwargs):
        super().__init__(include_jepa_views=False, **kwargs)
        self.item_to_cache_index = item_to_cache_index

    def __getitem__(self, index: int) -> dict:
        result = super().__getitem__(index)
        target_id = str(self.samples[index]["target_id"])
        if target_id not in self.item_to_cache_index:
            raise KeyError(f"Target item {target_id!r} is missing from the target cache")
        result["target_index"] = self.item_to_cache_index[target_id]
        return result


def stage1b_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    result = jepa_collate_fn(batch, pad_id=pad_id)
    result["target_index"] = torch.tensor(
        [sample["target_index"] for sample in batch], dtype=torch.long
    )
    return result


class SIDLevelPlanner(nn.Module):
    """A near-identity linear predictor for each of the four SID levels."""

    def __init__(self, hidden_size: int, num_levels: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_levels)]
        )
        with torch.no_grad():
            identity = torch.eye(hidden_size)
            for layer in self.layers:
                layer.weight.copy_(identity)
                layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(hidden, (hidden.size(-1),))
        return torch.stack([layer(normalized) for layer in self.layers], dim=1)


class URRStage1b(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        planner: SIDLevelPlanner,
        level_token_ids: torch.Tensor,
        *,
        plan_temperature: float,
        plan_level_weights: torch.Tensor,
        plan_state_weights: torch.Tensor,
    ):
        super().__init__()
        self.language_model = language_model
        self.planner = planner
        self.plan_temperature = float(plan_temperature)
        self.register_buffer("level_token_ids", level_token_ids.long())
        self.register_buffer("plan_level_weights", plan_level_weights.float())
        self.register_buffer("plan_state_weights", plan_state_weights.float())

    def _target_level_states(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        first_target: torch.Tensor,
        plan_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(len(SID_LEVELS), device=hidden.device)
        token_positions = first_target[:, None] + offsets[None, :]
        if int(token_positions.max().item()) >= labels.size(1):
            raise ValueError("A target sequence ended before all four SID levels")

        observed = labels.gather(1, token_positions)
        expected = self.level_token_ids.gather(1, plan_targets.t()).t()
        if not torch.equal(observed, expected):
            mismatches = int(observed.ne(expected).sum().item())
            raise ValueError(
                f"The first four supervised tokens must be RQ-SIDs; found {mismatches} mismatches"
            )

        state_positions = token_positions.sub(1).clamp_min(0)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)[:, None]
        return hidden[batch_index, state_positions], state_positions

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        target_index: torch.Tensor,
        target_bank: torch.Tensor,
        sid_codes: torch.Tensor,
        diagnose: bool,
    ) -> dict:
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = output.hidden_states[-1]
        supervised = labels.ne(-100)
        if not bool(supervised.any(dim=1).all()):
            raise ValueError("Every sample must contain supervised target tokens")

        first_target = supervised.long().argmax(dim=1)
        history_positions = first_target.sub(1).clamp_min(0)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        history_hidden = hidden[batch_index, history_positions]

        positive = target_bank.index_select(0, target_index)
        point_cosine = F.cosine_similarity(history_hidden.float(), positive.float(), dim=-1)
        point_loss = (1.0 - point_cosine).mean()

        plan_targets = sid_codes.index_select(0, target_index)
        plan_states, plan_state_positions = self._target_level_states(
            hidden, labels, first_target, plan_targets
        )
        flat_states = plan_states.reshape(-1, plan_states.size(-1))
        level_prediction = (
            self.planner(flat_states)
            .float()
            .view(hidden.size(0), len(SID_LEVELS), len(SID_LEVELS), -1)
        )

        output_head = self.language_model.get_output_embeddings()
        if output_head is None:
            output_head = self.language_model.get_input_embeddings()
        level_bank = (
            output_head.weight.index_select(0, self.level_token_ids.flatten())
            .view(len(SID_LEVELS), 256, -1)
            .detach()
            .float()
        )
        level_prediction = F.normalize(level_prediction, dim=-1)
        level_bank = F.normalize(level_bank, dim=-1)
        plan_logits = torch.einsum("bskd,kvd->bskv", level_prediction, level_bank)
        plan_logits = plan_logits / max(self.plan_temperature, 1e-6)

        expanded_targets = plan_targets[:, None, :].expand(-1, len(SID_LEVELS), -1)
        per_code_loss = F.cross_entropy(
            plan_logits.reshape(-1, 256),
            expanded_targets.reshape(-1),
            reduction="none",
        ).view_as(expanded_targets)

        state_level = torch.arange(len(SID_LEVELS), device=hidden.device)[:, None]
        target_level = torch.arange(len(SID_LEVELS), device=hidden.device)[None, :]
        remaining_mask = target_level.ge(state_level)
        term_weights = (
            remaining_mask.to(per_code_loss.dtype)
            * self.plan_state_weights[:, None]
            * self.plan_level_weights[None, :]
        )
        plan_loss = (
            (per_code_loss * term_weights[None, :, :]).sum(dim=(1, 2))
            / term_weights.sum().clamp_min(1e-12)
        ).mean()

        plan_correct = plan_logits.argmax(dim=-1).eq(expanded_targets)
        plan_accuracy = (plan_correct.float() * remaining_mask[None, :, :]).sum() / (
            remaining_mask.sum() * hidden.size(0)
        ).clamp_min(1)

        result = {
            "ce_loss": output.loss,
            "point_loss": point_loss,
            "plan_loss": plan_loss,
            "point_cosine": point_cosine.mean().detach(),
            "plan_accuracy": plan_accuracy.detach(),
        }
        if diagnose:
            ce_gradient = torch.autograd.grad(
                output.loss, hidden, retain_graph=True, allow_unused=False
            )[0].float()
            point_gradient = torch.autograd.grad(
                point_loss, hidden, retain_graph=True, allow_unused=False
            )[0].float()
            plan_gradient = torch.autograd.grad(
                plan_loss, hidden, retain_graph=True, allow_unused=False
            )[0].float()

            point_ce = ce_gradient[batch_index, history_positions]
            point_aux = point_gradient[batch_index, history_positions]
            plan_owners = batch_index[:, None].expand_as(plan_state_positions)
            plan_ce = ce_gradient[plan_owners, plan_state_positions]
            plan_aux = plan_gradient[plan_owners, plan_state_positions]

            result.update(
                {
                    "point_ce_grad_norm": point_ce.norm().detach(),
                    "point_grad_norm": point_aux.norm().detach(),
                    "point_grad_cos": F.cosine_similarity(
                        point_ce.flatten()[None], point_aux.flatten()[None]
                    )
                    .mean()
                    .detach(),
                    "plan_ce_grad_norm": plan_ce.norm().detach(),
                    "plan_grad_norm": plan_aux.norm().detach(),
                    "plan_grad_cos": F.cosine_similarity(
                        plan_ce.flatten()[None], plan_aux.flatten()[None]
                    )
                    .mean()
                    .detach(),
                }
            )
        return result


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
        ids = [tokenizer.convert_tokens_to_ids(f"<|{level}_{code}|>") for code in range(1, 257)]
        if any(token_id is None or token_id == unknown_id for token_id in ids):
            raise ValueError(f"Tokenizer is missing at least one level-{level} SID token")
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _save_checkpoint(
    module: URRStage1b,
    tokenizer,
    save_dir: Path,
    cfg: dict,
    effective_weights: dict[str, float],
) -> None:
    with tempfile.TemporaryDirectory(prefix="urr_stage1b_", dir="/tmp") as temp_root:
        temp_dir = Path(temp_root) / "checkpoint"
        temp_dir.mkdir()
        module.language_model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        torch.save(
            {
                "cfg": cfg,
                "effective_weights": effective_weights,
                "planner": module.planner.state_dict(),
            },
            temp_dir / "urr_stage1b_heads.pt",
        )
        with open(temp_dir / "training_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {**cfg, "effective_weights": effective_weights},
                handle,
                sort_keys=False,
            )

        save_dir.parent.mkdir(parents=True, exist_ok=True)
        partial_dir = save_dir.with_name(f".{save_dir.name}.partial-{os.getpid()}")
        shutil.rmtree(partial_dir, ignore_errors=True)
        try:
            shutil.copytree(temp_dir, partial_dir)
            if save_dir.exists():
                shutil.rmtree(save_dir)
            partial_dir.replace(save_dir)
        except BaseException:
            shutil.rmtree(partial_dir, ignore_errors=True)
            raise


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
        logger.info("URR Stage 1b config: %s", cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["aligned_base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["aligned_base_model"],
        dtype=dtype,
        trust_remote_code=True,
        use_cache=False,
    )
    missing = [token for token in build_sid_tokens() if token not in tokenizer.get_vocab()]
    if missing:
        raise ValueError(f"Stage 1a checkpoint is missing {len(missing)} SID tokens")
    unfreeze_backbone(model)
    if cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    planner = SIDLevelPlanner(int(model.config.hidden_size))
    module = URRStage1b(
        model,
        planner,
        _level_token_ids(tokenizer),
        plan_temperature=cfg["plan_temperature"],
        plan_level_weights=torch.tensor(cfg["plan_level_weights"]),
        plan_state_weights=torch.tensor(cfg["plan_state_weights"]),
    ).to(device=device, dtype=dtype)

    cache = torch.load(cfg["target_cache"], map_location="cpu")
    required_keys = {"item_ids", "sid_codes", "sid_hidden"}
    missing_keys = required_keys.difference(cache)
    if missing_keys:
        raise KeyError(f"Target cache is missing keys: {sorted(missing_keys)}")
    item_ids = [str(item_id) for item_id in cache["item_ids"]]
    item_to_cache_index = {item_id: index for index, item_id in enumerate(item_ids)}
    sid_codes = cache["sid_codes"].to(device=device, dtype=torch.long)
    target_bank = F.normalize(cache["sid_hidden"].float(), dim=-1).to(device)
    if target_bank.size(1) != int(model.config.hidden_size):
        raise ValueError(
            f"Target dimension {target_bank.size(1)} does not match model hidden "
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

    backbone_parameters = [
        parameter for parameter in raw_module.language_model.parameters() if parameter.requires_grad
    ]
    planner_parameters = list(raw_module.planner.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": cfg["learning_rate"]},
            {
                "params": planner_parameters,
                "lr": cfg["learning_rate"] * cfg["plan_lr_ratio"],
            },
        ],
        weight_decay=cfg["weight_decay"],
    )

    dataset = Stage1bDataset(
        item_to_cache_index=item_to_cache_index,
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
    per_device_batch = max(1, cfg["batch_size"] // max(world_size, 1))
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=lambda samples: stage1b_collate(samples, pad_id=tokenizer.pad_token_id or 0),
        drop_last=True,
    )
    total_steps = len(loader) * cfg["scheduler_num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    if is_main:
        logger.info(
            "Samples=%d batch=%d x world=%d train_epochs=%d scheduler_epochs=%d",
            len(dataset),
            per_device_batch,
            world_size,
            cfg["num_epochs"],
            cfg["scheduler_num_epochs"],
        )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_weights: dict[str, float] = {}
    global_step = 0
    module.train()

    for epoch in range(cfg["num_epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        totals = {"ce": 0.0, "point": 0.0, "plan": 0.0}
        metric_totals = {"point_cosine": 0.0, "plan_accuracy": 0.0}
        progress = tqdm(
            loader,
            desc=f"URR Stage 1b epoch {epoch + 1}/{cfg['num_epochs']}",
            disable=not is_main,
            dynamic_ncols=True,
        )
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            diagnose = bool(cfg["diagnose_first_step"]) and global_step == 0
            logs = module(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                target_index=batch["target_index"],
                target_bank=target_bank,
                sid_codes=sid_codes,
                diagnose=diagnose,
            )

            if diagnose:
                for name, desired_ratio in [
                    ("point", cfg["point_grad_ratio"]),
                    ("plan", cfg["plan_grad_ratio"]),
                ]:
                    ce_norm = _distributed_mean(logs[f"{name}_ce_grad_norm"], world_size)
                    aux_norm = _distributed_mean(logs[f"{name}_grad_norm"], world_size)
                    weight = desired_ratio * ce_norm / aux_norm.clamp_min(1e-12)
                    effective_weights[name] = float(weight.clamp(max=cfg["max_aux_weight"]).item())
                    if is_main:
                        logger.info(
                            "Calibrated %s: target_grad_ratio=%.4f weight=%.6f "
                            "hidden_grad_cos=%.4f raw_grad_ratio=%.4f",
                            name,
                            desired_ratio,
                            effective_weights[name],
                            logs[f"{name}_grad_cos"].item(),
                            (aux_norm / ce_norm.clamp_min(1e-12)).item(),
                        )

            if not effective_weights:
                raise RuntimeError("Auxiliary weights were not calibrated")
            loss = (
                logs["ce_loss"]
                + effective_weights["point"] * logs["point_loss"]
                + effective_weights["plan"] * logs["plan_loss"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(backbone_parameters + planner_parameters, cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            totals["ce"] += float(logs["ce_loss"].item())
            totals["point"] += float(logs["point_loss"].item())
            totals["plan"] += float(logs["plan_loss"].item())
            for name in metric_totals:
                metric_totals[name] += float(logs[name].item())
            global_step += 1
            progress.set_postfix(
                ce=f"{logs['ce_loss'].item():.4f}",
                point=f"{logs['point_loss'].item():.3f}",
                plan=f"{logs['plan_loss'].item():.3f}",
            )
            if is_main and global_step % cfg["log_steps"] == 0:
                logger.info(
                    "epoch=%d step=%d ce=%.4f point=%.4f*%.6f plan=%.4f*%.6f lr=%.2e",
                    epoch + 1,
                    global_step,
                    logs["ce_loss"].item(),
                    logs["point_loss"].item(),
                    effective_weights["point"],
                    logs["plan_loss"].item(),
                    effective_weights["plan"],
                    scheduler.get_last_lr()[0],
                )
        progress.close()

        if is_main:
            count = max(len(loader), 1)
            logger.info(
                "Epoch %d done | ce=%.4f point=%.4f plan=%.4f point_cos=%.4f plan_acc=%.4f",
                epoch + 1,
                totals["ce"] / count,
                totals["point"] / count,
                totals["plan"] / count,
                metric_totals["point_cosine"] / count,
                metric_totals["plan_accuracy"] / count,
            )
            epoch_number = epoch + 1
            if epoch_number in cfg["save_epochs"]:
                _save_checkpoint(
                    raw_module,
                    tokenizer,
                    output_dir / f"epoch_{epoch_number}",
                    cfg,
                    effective_weights,
                )
        if world_size > 1:
            torch.distributed.barrier()

    if is_main:
        logger.info("URR Stage 1b complete -> %s", output_dir)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train URR Stage 1b")
    parser.add_argument("--config", required=True)
    parser.add_argument("--aligned_base_model")
    parser.add_argument("--target_cache")
    parser.add_argument("--dataset")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    overrides = {
        key: value for key, value in vars(args).items() if key != "config" and value is not None
    }
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
