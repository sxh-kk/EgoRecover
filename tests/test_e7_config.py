from pathlib import Path
import pytest
from config.defaults import get_cfg_defaults, validate_e7


def test_default_config_is_dense_e7_with_repaint_off():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "config/e7.yaml"))
    validate_e7(cfg)
    assert cfg.FLOW.NUM_STEPS == 10
    assert cfg.FLOW.REPAINT_ENABLED is False
    assert cfg.DATA.WINDOW == 80
    assert cfg.FLOW.GLOBAL_WEIGHT == 8
    assert cfg.TRAIN.EMA_DECAY == 0.992028


def test_non_e7_model_is_rejected():
    cfg = get_cfg_defaults()
    cfg.MODEL.GENERATIVE_TYPE = "diffusion"
    with pytest.raises(ValueError, match="E7 requires"):
        validate_e7(cfg)
