# Training

- `isg.py` grounds complete SID sequences against item-text representations.
- `build_anchor_bank.py` encodes each item once with the grounded model.
- `gsa.py` fine-tunes recommendation with cross-entropy, DSA, and HPP.

The DSA anchor bank and HPP planners are training-only components.
