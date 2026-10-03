# E7 frozen engineering smoke — 2026-09-28

Source: repository commit 8e135ad; ICCV2027.md; run/check_frozen_e7_smoke.py.

Uses the previously audited 2-take pilot (14 variants total); selects clean, freeze_3s, drift_0p03mps. No new/full dataset construction.

Each take runs a11 (both modalities), a10 (image only), a01 (trajectory only) for each selected variant: 9 independent closed loops, 18 total. Each is 200 frames at 10 FPS: shared 20-frame E7 EMA startup followed by 180 predicted frames. Same startup and sampling seed within each take; each action owns its subsequent history. Frame-index requests are checked for causality. No GT body, shape, or floor is provided to inference. Predictions, commits and references are saved before offline supervision is loaded.

Frozen G uses the E7 EMA backbone in HistoryUniEgoMotion, with new projections initialized to zero. There is no trained P/Q or optimizer, and Gaussian source sigma=1 is retained. This is an execution and robustness diagnostic, not the official E7 windowed protocol or a test of a trained EgoRecover method. The histories still enter the transformer as past motion tokens. Constant-action rollouts are separate trajectories, not same-history counterfactual utility labels.

Offline evaluation verifies the SMPL-X asset against source GT, then computes FK geometry, phase errors and clean-paired fault deltas. Motion-frame deltas use 10 FPS. No recovery tolerance is invented. Official val samples are engineering-only; these two takes do not provide separate train/dev/holdout groups.

launch.json records reproducible commands, PIDs and physical GPUs (3 and 1). The processes preload existing driver 535.161.08 libraries to match the kernel, without changing system libraries. take0.log and take1.log record progress. Each take/report.json records loading, rollout, offline_smplx, complete or failed; completed=true requires successful offline evaluation.
