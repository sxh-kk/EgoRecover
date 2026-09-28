import os
from yacs.config import CfgNode as CN

_C = CN()
_C.DATA = CN()
_C.DATA.DATA_DIR = os.environ.get("UEM_DATA_DIR", "data/ee4d_motion_uniegomotion")
_C.DATA.DATASET_NAME = "ee4d"
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 4
_C.DATA.PIN_MEMORY = False
_C.DATA.PERSISTENT_WORKERS = False
_C.DATA.PREFETCH_FACTOR = 2
_C.DATA.DROP_LAST = False
_C.DATA.WINDOW = 80  # Length of the motion sequence. At 10 fps, window of 80 is 8 seconds.
_C.DATA.REPRE_TYPE = "v4_beta"  # motion representation type
_C.DATA.COND_IMG_FEAT = True  # whether to condition on image features
_C.DATA.COND_TRAJ = True  # whether to condition on aria trajectory
_C.DATA.COND_BETAS = False
_C.DATA.IMG_FEAT_TYPE = "dinov2"  # image feature type if conditioning on image features
_C.MODEL = CN()
_C.MODEL.CKPT_PATH = None
_C.MODEL.GENERATIVE_TYPE = "flow"
_C.MODEL.MODEL_NAME = "uem"
_C.FLOW = CN()
_C.FLOW.NUM_STEPS = 10
_C.FLOW.SOLVER = "euler"
_C.FLOW.BETA_ALPHA = 1.5
_C.FLOW.BETA_BETA = 1.0
_C.FLOW.T_MIN = 0.001
_C.FLOW.PREDICTION_TYPE = "x0"
_C.FLOW.GLOBAL_WEIGHT = 8.0
_C.FLOW.GLOBAL_FEATURE_START = 198
_C.FLOW.GLOBAL_FEATURE_END = 207
_C.TRAIN = CN()
_C.TRAIN.LR = 3.0e-5
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.USE_CKPT_LR = False  # whether to use lr from the checkpoint rather than the config lr.
_C.TRAIN.EXP_PATH = None  # experiment log path to save logs and checkpoints
_C.TRAIN.NUM_EPOCHS = 200
_C.TRAIN.MAX_STEPS = -1  # positive values stop at an exact optimizer-step budget.
_C.TRAIN.LOG_EVERY_N_STEPS = 50
_C.TRAIN.CHECK_VAL_EVERY_N_EPOCHS = 1
_C.TRAIN.SAVE_EVERY_N_EPOCHS = 10
_C.TRAIN.ONLY_VALIDATE = False
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.PRECISION = "32-true"
_C.TRAIN.NUM_SANITY_VAL_STEPS = 2
_C.TRAIN.ACCUMULATE_GRAD_BATCHES = 1
_C.TRAIN.GRADIENT_CLIP_VAL = 1.0
_C.TRAIN.DDP_STATIC_GRAPH = False
_C.TRAIN.CUDNN_BENCHMARK = False
_C.TRAIN.FUSED_ADAMW = False
_C.TRAIN.EMA_DECAY = 0.999
_C.TRAIN.SCHEDULER = "step"
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.SCHEDULER_TOTAL_EPOCHS = 0  # 0 follows NUM_EPOCHS; set explicitly for stable extension runs.
_C.TRAIN.MIN_LR_RATIO = 0.1
_C.TRAIN.EARLY_STOP_PATIENCE = 0
_C.TRAIN.EARLY_STOP_MIN_DELTA = 1.0e-4
_C.TRAIN.PROGRESS_REFRESH_RATE = 1
_C.TRAIN.EVAL_SUFFIX = ""  # suffix to append to the evaluation and visualization results file
_C.TRAIN.EVAL_TASK = None  # task to evaluate or visualize. Should be one of ["recon", "gen", "fore"]
_C.TRAIN.COND_SCALE = None  # classifier free guidance scale. We do not use this for UniEgoMotion evaluation.
_C.EVAL = CN()
_C.EVAL.KEY_JOINTS_ONLY = False
_C.EVAL.KEY_JOINT_INDICES = [0, 4, 5, 7, 8, 10, 11, 15, 18, 19, 20, 21]
_C.EVAL.RUN_SEMANTIC = False
_C.EVAL.NUM_GPUS = 1
_C.EVAL.NUM_SAMPLES = 0  # 0 evaluates the full strided validation split.
_C.EVAL.BATCH_SIZE = 64  # Per-rank inference batch size.
_C.FLOW.REPAINT_ENABLED = False


def get_cfg_defaults():
    return _C.clone()


def validate_e7(cfg):
    required = {
        "MODEL.GENERATIVE_TYPE": "flow",
        "MODEL.MODEL_NAME": "uem",
        "DATA.REPRE_TYPE": "v4_beta",
        "DATA.IMG_FEAT_TYPE": "dinov2",
        "DATA.COND_BETAS": False,
        "DATA.COND_IMG_FEAT": True,
        "DATA.COND_TRAJ": True,
        "FLOW.PREDICTION_TYPE": "x0",
        "FLOW.SOLVER": "euler",
        "FLOW.GLOBAL_WEIGHT": 8.0,
        "FLOW.GLOBAL_FEATURE_START": 198,
        "FLOW.GLOBAL_FEATURE_END": 207,
    }
    for key, expected in required.items():
        node = cfg
        for part in key.split("."):
            node = node[part]
        if node != expected:
            raise ValueError(f"E7 requires {key}={expected!r}, got {node!r}.")


def get_cfg():
    import sys
    from pathlib import Path

    cfg = get_cfg_defaults()
    args = sys.argv[1:]
    if args and args[0] == "CONFIG":
        if len(args) < 2:
            raise ValueError("CONFIG requires a YAML path.")
        cfg.merge_from_file(args[1])
        args = args[2:]
    else:
        cfg.merge_from_file(str(Path(__file__).with_name("e7.yaml")))
    cfg.merge_from_list(args)
    validate_e7(cfg)
    cfg.freeze()
    print(cfg.dump())
    return cfg
