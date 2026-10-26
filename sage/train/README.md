# Training

- `isg.py` grounds the SID embeddings on a frozen LLM (Stage 1).
- `build_anchor_bank.py` encodes each item once with the grounded model.
- `psa.py` fine-tunes recommendation with cross-entropy and the semantic and
  identity readouts of one shared predictor (Stage 2).

The anchor bank and the PSA predictor are training-only components.
