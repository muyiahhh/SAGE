from pathlib import Path

from sage.train.gsa import load_config as load_gsa_config
from sage.train.isg import load_config as load_isg_config


ROOT = Path(__file__).resolve().parents[1]


def test_released_isg_configuration():
    cfg = load_isg_config(ROOT / "configs/beauty_qwen3_0.6b_isg.yaml")
    assert cfg["projection_type"] == "identity"
    assert cfg["d_proj"] == 1024
    assert cfg["predictor_type"] == "linear"
    assert cfg["jepa_loss_type"] == "cosine"
    assert cfg["learning_rate"] == 1.5e-3
    assert cfg["num_epochs"] == 10
    assert cfg["save_each_epoch"] is False


def test_released_gsa_configuration():
    cfg = load_gsa_config(ROOT / "configs/beauty_qwen3_0.6b_gsa.yaml")
    assert cfg["dsa_grad_ratio"] == 0.01
    assert cfg["dsa_loss_type"] == "catalog_ce"
    assert cfg["hpp_grad_ratio"] == 0.03
    assert cfg["hpp_scope"] == "remaining"
    assert cfg["num_epochs"] == 3
    assert cfg["scheduler_num_epochs"] == 4
    assert cfg["save_epochs"] == [3]
