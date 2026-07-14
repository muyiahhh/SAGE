from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from urr.train.stage1b import SIDLevelPlanner, URRStage1b


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 1024, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        labels,
        output_hidden_states,
        use_cache,
    ):
        del attention_mask, output_hidden_states, use_cache
        hidden = self.embedding(input_ids)
        logits = self.output(hidden)
        supervised = labels.ne(-100)
        loss = F.cross_entropy(logits[supervised], labels[supervised])
        return SimpleNamespace(loss=loss, hidden_states=(hidden,))

    def get_output_embeddings(self):
        return self.output

    def get_input_embeddings(self):
        return self.embedding


def test_point_plan_forward_and_gradient_diagnostics():
    torch.manual_seed(0)
    hidden_size = 8
    level_token_ids = torch.arange(4 * 256).view(4, 256)
    sid_codes = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    target_index = torch.tensor([0, 1])

    labels = torch.full((2, 7), -100, dtype=torch.long)
    input_ids = torch.zeros((2, 7), dtype=torch.long)
    for row in range(2):
        target_tokens = level_token_ids[torch.arange(4), sid_codes[row]]
        labels[row, 2:6] = target_tokens
        input_ids[row, 2:6] = target_tokens

    model = TinyLanguageModel(hidden_size=hidden_size)
    module = URRStage1b(
        model,
        SIDLevelPlanner(hidden_size),
        level_token_ids,
        plan_temperature=0.07,
        plan_level_weights=torch.ones(4),
        plan_state_weights=torch.ones(4),
    )
    logs = module(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        target_index=target_index,
        target_bank=F.normalize(torch.randn(2, hidden_size), dim=-1),
        sid_codes=sid_codes,
        diagnose=True,
    )

    for key in [
        "ce_loss",
        "point_loss",
        "plan_loss",
        "point_grad_norm",
        "plan_grad_norm",
    ]:
        assert torch.isfinite(logs[key])
    total = logs["ce_loss"] + 0.1 * logs["point_loss"] + 0.1 * logs["plan_loss"]
    total.backward()
    assert all(layer.weight.grad is not None for layer in module.planner.layers)
