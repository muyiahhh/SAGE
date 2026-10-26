"""Item-level Semantic Grounding (ISG).

ISG treats the SID sequence and the text of an item as two views of the same
item. The LLM backbone stays frozen and only the newly added SID embedding rows
and a linear predictor are trained.

* SID view:  ``<|a_i|> <|b_j|> <|c_k|> <|d_l|> [<|e_n|>] [ITEM]``. Its
  representation is the final hidden state at ``[ITEM]``.
* Text view: ``title: ..., categories: ..., brand: ..., description: ...``. Its
  representation is the final hidden state at the last token, computed by the
  same frozen backbone without gradients.

The loss is the cosine distance between the predicted SID view and the text
view, ``mean(1 - cos(g(z_sid), z_text))``. The text view contains no SID tokens,
so its representation is exactly that of the frozen pretrained LLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sage.models.predictor import build_predictor


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
    init_strategy: str = "mean",
) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:
    """Add SID tokens to the tokenizer and resize the model's embedding table.

    ``init_strategy`` controls only newly-added rows:
      - ``mean`` initializes input and untied output rows from the pretrained
        vocabulary mean (the setting used in the paper).
      - ``normal`` samples them from N(0, 0.02).

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
        # atomically in recommendation targets instead of leaking byte/digit pieces.
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

                # Some backbones use an untied LM head. Initialize those new
                # output rows identically so the result does not depend on the
                # Transformers resize default.
                out_emb = model.get_output_embeddings()
                if out_emb is not None and out_emb.weight.data_ptr() != emb.weight.data_ptr():
                    out_emb.weight[orig_vocab_size:] = mean_row.to(
                        dtype=out_emb.weight.dtype, device=out_emb.weight.device
                    )

    return model, tokenizer, orig_vocab_size


def freeze_backbone(model: AutoModelForCausalLM, orig_vocab_size: int) -> None:
    """Freeze all LLM parameters except the newly added SID embedding rows.

    The embedding matrix keeps ``requires_grad=True`` so that the SID view can
    back-propagate into it. A gradient hook zeroes the rows of the pretrained
    vocabulary, which therefore never change (their weight decay is zero).
    """
    emb = model.get_input_embeddings()

    for param in model.parameters():
        param.requires_grad_(False)
    emb.weight.requires_grad_(True)

    def _zero_old_rows(grad):
        if grad is None:
            return grad
        grad = grad.clone()
        grad[:orig_vocab_size] = 0.0
        return grad

    emb.weight.register_hook(_zero_old_rows)


def unfreeze_backbone(model: AutoModelForCausalLM) -> None:
    """Unfreeze all LLM parameters for recommendation fine-tuning."""
    for param in model.parameters():
        param.requires_grad_(True)


def gather_hidden(
    language_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Final-layer hidden state of the backbone at one position per sequence.

    ``base_model`` returns the normalized last hidden state, which is the input
    of the LM head. The LM-head logits are not computed.
    """
    hidden = language_model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state
    batch_index = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_index, positions]


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``mean(1 - cos(prediction, target))`` computed in float32."""
    return (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()


class ISGAligner(nn.Module):
    """SID-to-text predictive alignment on a frozen LLM.

    Parameters
    ----------
    language_model : causal LM with SID tokens added and the backbone frozen
    hidden_size    : hidden dimension of the LLM
    predictor_type : predictor g from the SID view to the text view
                     (``linear`` in the paper)
    """

    def __init__(
        self,
        language_model: nn.Module,
        hidden_size: int,
        predictor_type: str = "linear",
    ):
        super().__init__()
        self.language_model = language_model
        self.predictor = build_predictor(predictor_type, hidden_size)

    def forward(
        self,
        sid_input_ids: torch.Tensor,
        sid_attention_mask: torch.Tensor,
        item_pos: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        eos_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # SID view: gradients reach only the SID embedding rows and g.
        z_sid = gather_hidden(self.language_model, sid_input_ids, sid_attention_mask, item_pos)
        prediction = self.predictor(z_sid)

        # Text view: the frozen backbone, stop-gradient target.
        with torch.no_grad():
            z_text = gather_hidden(
                self.language_model, text_input_ids, text_attention_mask, eos_pos
            )

        loss = cosine_distance(prediction, z_text.detach())
        logs = {"isg_loss": loss.detach()}
        return loss, logs
