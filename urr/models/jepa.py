"""JEPA alignment module used by URR Stage 1a.

Architecture:
  - context_encoder : LLM backbone + SID embedding (frozen backbone, trainable SID rows)
  - context_proj    : projection or identity mapping (trainable if projected)
  - predictor       : representation predictor (linear in the released setting)
  - target_encoder  : EMA copy of (LLM backbone + SID embedding)
  - target_proj     : EMA copy of context_proj

During Stage 1a the original LLM parameters are frozen and only newly added SID
embedding rows plus the representation heads are optimized. Stage 1b loads the
aligned language model and then unfreezes it for recommendation training.

The released Qwen3-0.6B setting works directly in model hidden space and uses
cosine distance between the predicted SID view and stop-gradient text view.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from urr.models.projection import ProjectionHead
from urr.models.predictor import build_predictor
from urr.utils.ema import build_ema_model, ema_update


def jepa_distance(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """JEPA prediction distance. target must already be stop-gradient'd.

    - "mse"     : F.mse_loss(pred, target)
    - "cosine"  : mean(1 - cos(pred, target))
    - "norm_mse": F.mse_loss(normalize(pred), normalize(target))
    """
    t = (loss_type or "mse").lower()
    if t == "mse":
        return F.mse_loss(pred, target)
    if t == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    if t == "norm_mse":
        return F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))
    raise ValueError(f"Unknown jepa_loss_type: {loss_type!r}")


# SID token naming convention used by the data pipeline
SID_LEVELS = ["a", "b", "c", "d"]  # 4 quantization levels, 256 codes each
SID_DEDUP_TOKENS = [f"<|e_{i}|>" for i in range(1, 257)]  # collision dedup tokens
ITEM_MARKER = "[ITEM]"


def build_sid_tokens(levels=SID_LEVELS, n_codes: int = 256) -> list[str]:
    """Return SID tokens matching the RQ-KMeans index format.

    The RQ pipeline writes codes as ``codes_all + 1``, so valid code ids are
    1..n_codes.  Registering 0..255 leaves ``<|*_256|>`` as ordinary text,
    which makes SID targets tokenize into mixed regular/SID tokens.
    """
    tokens = []
    for lv in levels:
        tokens += [f"<|{lv}_{i}|>" for i in range(1, n_codes + 1)]
    tokens += SID_DEDUP_TOKENS
    tokens += [ITEM_MARKER]
    return tokens


def add_sid_tokens_to_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sid_tokens: list[str],
    init_strategy: str = "normal",
) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:
    """Add SID tokens to the tokenizer and resize the model's embedding table.

    ``init_strategy`` controls only newly-added rows:
      - ``normal`` preserves the original JEPA initialization (N(0, 0.02)).
      - ``mean`` initializes input and untied output rows from the pretrained
        vocabulary mean.  This is useful for controlled GTI/JEPA comparisons.

    Returns (model, tokenizer, orig_vocab_size).
    """
    init_strategy = str(init_strategy).lower()
    if init_strategy not in {"normal", "mean"}:
        raise ValueError(
            f"Unknown SID init strategy {init_strategy!r}; expected 'normal' or 'mean'."
        )

    orig_vocab_size = len(tokenizer)
    new_tokens = [t for t in sid_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        # Qwen tokenizers reserve the <|...|> pattern for special tokens.  Adding
        # SID strings as special added tokens makes adjacent SID strings tokenize
        # atomically in Stage 1b targets instead of leaking byte/digit pieces.
        tokenizer.add_tokens(new_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        emb = model.get_input_embeddings()
        with torch.no_grad():
            if init_strategy == "normal":
                nn.init.normal_(emb.weight[orig_vocab_size:], mean=0.0, std=0.02)
            else:
                mean_row = emb.weight[:orig_vocab_size].float().mean(dim=0)
                emb.weight[orig_vocab_size:] = mean_row.to(
                    dtype=emb.weight.dtype, device=emb.weight.device
                )

                # Qwen variants may use an untied LM head.  Initialize those
                # new output rows identically so the control does not depend on
                # Transformers' resize default.
                out_emb = model.get_output_embeddings()
                if out_emb is not None and out_emb.weight.data_ptr() != emb.weight.data_ptr():
                    out_emb.weight[orig_vocab_size:] = mean_row.to(
                        dtype=out_emb.weight.dtype, device=out_emb.weight.device
                    )

    return model, tokenizer, orig_vocab_size


def freeze_backbone(model: AutoModelForCausalLM, orig_vocab_size: int) -> None:
    """Freeze all LLM backbone params; leave newly-added SID embedding rows trainable.

    We freeze by setting requires_grad=False (NOT by no_grad), so the context-side
    forward graph still reaches the trainable SID embedding rows.
    """
    emb = model.get_input_embeddings()

    # Freeze the entire embedding weight tensor, then we will hook to allow grad
    # only through the new token rows via a gradient mask applied in the optimizer.
    # Simpler alternative: replace the embedding with a SplitEmbedding wrapper.
    # We use requires_grad on all backbone params except the embedding, and handle
    # the embedding specially with a gradient hook.
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "embed" in name.lower():
            # Keep gradient flowing — we'll zero old rows via hook
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Register a hook that zeroes gradients for the original vocabulary rows
    # so only the new SID rows actually receive gradient updates.
    def _zero_old_rows(grad):
        if grad is None:
            return grad
        grad = grad.clone()
        grad[:orig_vocab_size] = 0.0
        return grad

    emb.weight.register_hook(_zero_old_rows)


def unfreeze_backbone(model: AutoModelForCausalLM) -> None:
    """Unfreeze all LLM backbone params for Stage 1b."""
    for param in model.parameters():
        param.requires_grad_(True)


class JEPAAligner(nn.Module):
    """Full JEPA alignment module.

    Parameters
    ----------
    context_model : LLM backbone (Qwen3-0.6B with SID tokens added)
    d_model       : hidden dim of the LLM (read from config, never hardcoded)
    d_proj        : projection dimension, equal to d_model for identity mode
    ema_tau       : EMA momentum for target encoder (default 0.999)
    """

    def __init__(
        self,
        context_model: nn.Module,
        d_model: int,
        d_proj: int = 256,
        ema_tau: float = 0.999,
        predictor_type: str = "mlp3",
        jepa_loss_type: str = "mse",
        projection_type: str = "linear",
    ):
        super().__init__()
        self.ema_tau = ema_tau
        self.jepa_loss_type = jepa_loss_type
        self.projection_type = str(projection_type).lower()
        if self.projection_type not in {"linear", "identity"}:
            raise ValueError("projection_type must be 'linear' or 'identity'")
        if self.projection_type == "identity" and d_proj != d_model:
            raise ValueError(
                f"identity projection requires d_proj == d_model ({d_model}), got {d_proj}"
            )

        # Context branch (gradients flow through SID embeddings + projection + predictor)
        self.context_encoder = context_model
        self.context_proj = (
            nn.Identity() if self.projection_type == "identity" else ProjectionHead(d_model, d_proj)
        )
        self.predictor = build_predictor(predictor_type, d_proj)

        # Target branch (EMA copy; no gradient)
        self.target_encoder = build_ema_model(context_model)
        self.target_proj = build_ema_model(self.context_proj)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        item_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Run context encoder; extract hidden state at [ITEM] position.

        input_ids     : (B, L)
        attention_mask: (B, L)
        item_pos      : (B,)  — index of [ITEM] token in each sequence
        Returns z_SID : (B, d_proj)
        """
        out = self.context_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]  # (B, L, d_model)
        # Gather the [ITEM] position for each sample
        idx = item_pos.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        idx = idx.expand(-1, 1, hidden.size(-1))  # (B, 1, d_model)
        h_item = hidden.gather(1, idx).squeeze(1)  # (B, d_model)
        z_SID = self.context_proj(h_item)  # (B, d_proj)
        return z_SID

    @torch.no_grad()
    def _encode_target(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Run EMA target encoder; pool at EOS position.

        Returns z_text: (B, d_proj)
        """
        out = self.target_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]  # (B, L, d_model)
        idx = eos_pos.unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, 1, hidden.size(-1))
        h_eos = hidden.gather(1, idx).squeeze(1)  # (B, d_model)
        z_text = self.target_proj(h_eos)  # (B, d_proj)
        return z_text

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def forward(
        self,
        # View A: SID sequence
        sid_input_ids: torch.Tensor,
        sid_attention_mask: torch.Tensor,
        item_pos: torch.Tensor,
        # View B: text description
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        eos_pos: torch.Tensor,
        # Optional variance regularization
        use_var_reg: bool = False,
        var_reg_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:

        z_SID = self._encode_context(sid_input_ids, sid_attention_mask, item_pos)
        pred = self.predictor(z_SID)  # (B, d_proj)

        z_text = self._encode_target(text_input_ids, text_attention_mask, eos_pos)
        # stop-gradient on target — NEVER remove this
        z_text = z_text.detach()

        jepa_loss = jepa_distance(pred, z_text, self.jepa_loss_type)

        loss = jepa_loss
        log = {"jepa_loss": jepa_loss.detach()}

        if use_var_reg:
            # Penalize collapsed predictor output: variance < 1 per dimension
            std = pred.std(dim=0)  # (d_proj,)
            var_loss = F.relu(1.0 - std).mean()
            loss = loss + var_reg_weight * var_loss
            log["var_loss"] = var_loss.detach()

        log["total_jepa"] = loss.detach()
        return loss, log

    # ------------------------------------------------------------------
    # EMA step — call after every optimizer step during Stage 1a
    # ------------------------------------------------------------------

    def update_ema(self) -> None:
        ema_update(self.context_encoder, self.target_encoder, tau=self.ema_tau)
        ema_update(self.context_proj, self.target_proj, tau=self.ema_tau)
