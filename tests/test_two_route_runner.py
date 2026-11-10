from scripts.run_two_route_training import route_command


def test_routes_share_c_initialization_and_keep_training_controls_on_resume(tmp_path):
    plan = {"python": "python", "data_root": "data", "prior_checkpoint": "P.pt", "common_G_checkpoint": "C.pt"}
    job = {"port": 30311, "config": "two.json", "history": "mixed", "rewrite": True, "unroll_steps": 2}
    command = route_command(plan, job, tmp_path, preflight=True)
    assert command[command.index("--initialize") + 1] == "C.pt"
    assert command[command.index("--unroll-steps") + 1] == "2"
    assert command[command.index("--feedback-intervention") + 1] == "body_pose"
    assert "--isolate-forward-rng" in command
    resumed = route_command(plan, {**job, "unroll_steps": 1}, tmp_path, preflight=True, resume=True)
    assert "--resume" in resumed and "--initialize" not in resumed
    assert resumed[resumed.index("--unroll-steps") + 1] == "1"
