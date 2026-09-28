import sys
import copy
import subprocess


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    old_argv = copy.deepcopy(sys.argv)

    from config.defaults import get_cfg

    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert cfg.MODEL.CKPT_PATH is not None

    for task in ["recon", "gen", "fore"]:
        # for task in ["recon"]:

        prediction_cmd = [sys.executable, "-m", "eval.save_uem_preds"]
        if cfg.EVAL.NUM_GPUS > 1:
            prediction_cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(cfg.EVAL.NUM_GPUS),
                "--module",
                "eval.save_uem_preds",
            ]
        cmd = prediction_cmd + old_argv[1:] + ["TRAIN.EVAL_TASK", task]
        run_cmd(cmd)

        cmd = [sys.executable, "-m", "eval.compute_3d_metrics", "--EXP_PATH", cfg.TRAIN.EXP_PATH, "--EVAL_TASK", task]
        cmd += ["--DATA_DIR", cfg.DATA.DATA_DIR]
        if cfg.EVAL.KEY_JOINTS_ONLY:
            cmd.append("--KEY_JOINTS_ONLY")
            if cfg.EVAL.KEY_JOINT_INDICES:
                cmd += ["--KEY_JOINT_INDICES"] + [str(index) for index in cfg.EVAL.KEY_JOINT_INDICES]
        if cfg.TRAIN.EVAL_SUFFIX is not None and cfg.TRAIN.EVAL_SUFFIX != "":
            cmd += ["--EVAL_SUFFIX", cfg.TRAIN.EVAL_SUFFIX]
        run_cmd(cmd)

        if cfg.EVAL.RUN_SEMANTIC:
            cmd = [
                sys.executable,
                "-m",
                "eval.compute_semantic_metrics",
                "--EXP_PATH",
                cfg.TRAIN.EXP_PATH,
                "--EVAL_TASK",
                task,
            ]
            if cfg.TRAIN.EVAL_SUFFIX is not None and cfg.TRAIN.EVAL_SUFFIX != "":
                cmd += ["--EVAL_SUFFIX", cfg.TRAIN.EVAL_SUFFIX]
            run_cmd(cmd)


if __name__ == "__main__":
    main()
