from pathlib import Path

from urr.train.stage1a_jepa import load_config as load_stage1a_config
from urr.train.stage1b import load_config as load_stage1b_config


ROOT = Path(__file__).resolve().parents[1]


def test_released_stage1a_configuration():
    cfg = load_stage1a_config(ROOT / "configs/beauty_0.6b_stage1a.yaml")
    assert cfg["projection_type"] == "identity"
    assert cfg["d_proj"] == 1024
    assert cfg["predictor_type"] == "linear"
    assert cfg["jepa_loss_type"] == "cosine"
    assert cfg["learning_rate"] == 1.5e-3
    assert cfg["num_epochs"] == 10
    assert cfg["save_each_epoch"] is False


def test_released_stage1b_configuration():
    cfg = load_stage1b_config(ROOT / "configs/beauty_0.6b_stage1b.yaml")
    assert cfg["point_grad_ratio"] == 0.015
    assert cfg["plan_grad_ratio"] == 0.03
    assert cfg["plan_scope"] == "remaining"
    assert cfg["num_epochs"] == 3
    assert cfg["scheduler_num_epochs"] == 4
    assert cfg["save_epochs"] == [3]
