"""Complete the fixed stages 2–4 protocol on explicitly selected GPUs."""

import argparse
import csv
import fcntl
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote

from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from run.select_budget import select
from run.summarize_paired_development import interval, summarize


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path("exp/egorecover_stages_1_4")
SIGNAL = "data/EE4D_MISMATCH_TRAIN_72TAKE_READY.json"
SPLIT = "config/egorecover_train_72take_split_v1.json"
BOOTSTRAP = "exp/egorecover_train_72take_v1/bootstrap.pt"
CHECKPOINT = "exp/e7/last.ckpt"
MODES = ("gaussian", "history", "prior_only")
VARIANTS = ("clean", "freeze_3s", "drift_0p03mps")


def now():
    return datetime.now(timezone.utc).isoformat()


def command(module, **options):
    result = [sys.executable, "-u", "-m", module]
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        result.append(flag)
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, (list, tuple)) else [value]))
    return result


@dataclass
class Task:
    name: str
    argv: list
    output: Path
    kind: str
    expected: int = 0
    deps: list = field(default_factory=list)
    gpu: int | None = None

    @property
    def report(self):
        return self.output if self.kind == "diagnostic" else self.output / "report.json"


def verify_task(task):
    report = json.loads(task.report.read_text())
    if report.get("completed") is not True:
        raise ValueError(f"Incomplete report: {task.name}")
    if task.kind == "train":
        modes = requested_modes(task.argv, ("gaussian", "history"))
        if set(report["sources"]) != set(modes):
            raise ValueError("Training must complete all requested generators.")
        import torch
        for name in ("prior.pt", *(f"g_{mode}.pt" for mode in modes)):
            checkpoint = torch.load(task.output / name, map_location="cpu", weights_only=True, mmap=True)
            if any(checkpoint.get(key) != report[key] for key in
                   ("split_manifest_sha256", "stats_sha256", "bootstrap_cache_sha256", "reference_mode")):
                raise ValueError(f"Checkpoint identity differs: {task.output / name}")
            if not all(bool(torch.isfinite(tensor).all()) for tensor in checkpoint["state_dict"].values()):
                raise ValueError("Nonfinite model weights.")
            if name != "prior.pt":
                mode = name[2:-3]
                if file_sha256(task.output / name) != report["sources"][mode]["checkpoint_sha256"]:
                    raise ValueError("Generator checkpoint checksum differs.")
        if report.get("fixed_prior") and file_sha256(task.output / "prior.pt") != report["fixed_prior"]["checkpoint_sha256"]:
            raise ValueError("Fixed prior checkpoint checksum differs.")
    elif task.kind == "eval":
        rows = report["results"]
        expected = {(label, mode, take, variant) for label in report["experiments"]
                    for mode in requested_modes(task.argv, MODES) for take in report["takes"] for variant in VARIANTS}
        actual = [(row["experiment"], row["mode"], row["take"], row["variant"]) for row in rows]
        if len(rows) != task.expected or set(actual) != expected or len(actual) != len(set(actual)):
            raise ValueError("Incomplete evaluation coverage.")
        for row in rows:
            if file_sha256(Path(row["trace"])) != row["trace_sha256"]:
                raise ValueError("Evaluation trajectory checksum differs.")
    elif task.kind == "cache":
        if report["frames"] != task.expected or file_sha256(task.output / "frames.pt") != report["cache_sha256"]:
            raise ValueError("Incomplete or altered replay cache.")
    elif task.kind == "diagnostic":
        if len(report["rows"]) != task.expected or len(report["draw_averaged"]["states"]) * 3 != task.expected:
            raise ValueError("Incomplete four-action diagnostic.")
    elif task.kind == "prior_data":
        if report["sequences"] != task.expected or file_sha256(task.output / "sequences.pt") != report["cache_sha256"]:
            raise ValueError("Incomplete or altered P sequence cache.")
    elif task.kind == "prior":
        import torch
        checkpoint = torch.load(task.output / "prior.pt", map_location="cpu", weights_only=True)
        if file_sha256(task.output / "prior.pt") != report["prior_sha256"] or checkpoint["selected_step"] != report["selected_step"]:
            raise ValueError("P checkpoint differs from its report.")
        if not all(bool(torch.isfinite(value).all()) for value in checkpoint["state_dict"].values()):
            raise ValueError("Nonfinite P weights.")
        if file_sha256(task.output / "initial.pt") != report["initial_sha256"]:
            raise ValueError("Initial P weights differ from the report.")
        for key in ("data_sha256", "two_forward", "fk_weight", "training_seed", "code_sha256"):
            if checkpoint.get(key) != report[key]:
                raise ValueError(f"P checkpoint identity differs: {key}.")
        for result in report["results"].values():
            if file_sha256(result["artifact"]) != result["artifact_sha256"]:
                raise ValueError("P evaluation artifact differs from its report.")
    # Reject nonfinite numbers in all reports, including nested diagnostics.
    json.dumps(report, allow_nan=False)
    return report


def requested_modes(argv, default):
    if "--source-modes" not in argv:
        return tuple(default)
    result = []
    for value in argv[argv.index("--source-modes") + 1:]:
        if value.startswith("--"):
            break
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise ValueError("Expected unique source modes.")
    return tuple(result)


def idle_gpu_indices(gpus, env):
    def query(args):
        output = subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                         env=env, text=True, timeout=15)
        return list(csv.reader(io.StringIO(output)))
    cards = query(["--query-gpu=index,uuid,memory.used,utilization.gpu"])
    processes = query(["--query-compute-apps=gpu_uuid,pid"])
    occupied = {row[0].strip() for row in processes if len(row) == 2}
    return {int(index) for index, uuid, memory, utilization in cards
            if int(index) in gpus and uuid.strip() not in occupied
            and int(memory) < 1024 and int(utilization) < 5}


def shared_gpu_indices(gpus, env, minimum_free_gib):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,compute_mode", "--format=csv,noheader,nounits"],
        env=env, text=True, timeout=15,
    )
    return {int(index) for index, total, used, mode in csv.reader(io.StringIO(output))
            if int(index) in gpus and mode.strip() == "Default"
            and int(total) - int(used) >= minimum_free_gib * 1024}


class Queue:
    def __init__(self, output, gpus, poll_seconds, *, allow_shared=False, minimum_free_gib=32):
        self.output, self.gpus, self.poll_seconds = output, gpus, poll_seconds
        self.allow_shared, self.minimum_free_gib = allow_shared, minimum_free_gib
        output.mkdir(parents=True, exist_ok=True)
        (output / "jobs").mkdir(exist_ok=True)
        self.lock = (output / "queue.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = output / "queue.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"tasks": {}, "created_at": now()}
        previous_policy = {"gpus": self.state.get("gpus", gpus),
                           "allow_shared": self.state.get("allow_shared", False),
                           "minimum_free_gib": self.state.get("minimum_free_gib", 32)}
        policy = {"gpus": gpus, "allow_shared": allow_shared, "minimum_free_gib": minimum_free_gib}
        if previous_policy != policy:
            if any(entry["status"] == "running" for entry in self.state["tasks"].values()):
                raise ValueError("Cannot change GPU policy while experiment jobs are running.")
            self.state.setdefault("policy_changes", []).append({"at": now(), "from": previous_policy, "to": policy})
        self.state.update({**policy, "controller_pid": os.getpid(), "status": "running"})
        self.state.pop("error", None)
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(ROOT) + os.pathsep + self.env.get("PYTHONPATH", "")
        nvml = Path("/tmp/uniegomotion-nvml-host")
        if (nvml / "libnvidia-ml.so.1").exists():
            self.env["LD_LIBRARY_PATH"] = str(nvml) + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")
        self.idle_previous = set()
        self.children = {}
        self.save()

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.path, self.state)

    def log(self, message):
        print(f"{now()} {message}", flush=True)

    def register(self, task):
        definition = {"command": task.argv, "output": str(task.output), "kind": task.kind,
                      "expected": task.expected, "deps": task.deps}
        if task.gpu is not None:
            definition["assigned_gpu"] = task.gpu
        old = self.state["tasks"].get(task.name)
        if old and any(old.get(key) != value for key, value in definition.items()):
            raise ValueError(f"Task definition changed: {task.name}")
        if old is None:
            self.state["tasks"][task.name] = {**definition, "status": "pending", "attempt": 0}
        elif old["status"] == "complete":
            verify_task(task)

    def launch(self, task, gpu):
        if task.gpu is not None and gpu != task.gpu:
            raise ValueError("Task launched on a different GPU than assigned.")
        entry = self.state["tasks"][task.name]
        argv = list(task.argv)
        if task.output.exists():
            if task.kind == "eval" and (task.output / "progress.json").exists():
                argv.append("--resume")
            elif task.kind == "prior" and (task.output / "resume.pt").exists():
                argv.append("--resume")
            else:
                archive = task.output.with_name(task.output.name + f".interrupted_{time.time_ns()}")
                task.output.rename(archive)
                self.log(f"Preserved interrupted output: {archive}")
        task.output.parent.mkdir(parents=True, exist_ok=True)
        entry["attempt"] += 1
        spec = self.output / "jobs" / f"{task.name}.{entry['attempt']}.json"
        log = spec.with_suffix(".log")
        atomic_json(spec, {"command": argv, "gpu": gpu})
        env = {**self.env, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"}
        with log.open("a") as stream:
            child = subprocess.Popen(command("run.stage_job", spec=spec), cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[task.name] = child
        for key in ("error", "exit_code", "ended_at"):
            entry.pop(key, None)
        entry.update({"status": "running", "pid": child.pid, "gpu": gpu, "spec": str(spec),
                      "log": str(log), "started_at": now()})
        self.save()
        self.log(f"Started {task.name} on GPU{gpu}, PID {child.pid}, log {log}")

    def finish_running(self, task):
        entry = self.state["tasks"][task.name]
        spec = Path(entry["spec"])
        status_path = spec.with_suffix(".status.json")
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        child = self.children.get(task.name)
        code = child.poll() if child else status.get("exit_code")
        if status.get("exit_code") is not None:
            code = status["exit_code"]
        if code is None:
            try:
                cmdline = Path(f"/proc/{entry['pid']}/cmdline").read_bytes()
                if b"run.stage_job" in cmdline and str(spec).encode() in cmdline:
                    return
            except OSError:
                pass
            # A wrapper killed externally may leave its experiment alive.
            if status.get("child_pid") and Path(f"/proc/{status['child_pid']}").exists():
                return
            code = -1
        entry.update({"exit_code": code, "ended_at": now()})
        try:
            if code != 0:
                raise ValueError(f"Task exited with status {code}; inspect {entry['log']}")
            verify_task(task)
            entry["status"] = "complete"
        except Exception as error:
            entry.update({"status": "failed", "error": str(error)})
        self.log(f"{task.name}: {entry['status']}")
        self.save()

    def run(self, tasks):
        for task in tasks:
            self.register(task)
        self.save()
        while True:
            for task in tasks:
                if self.state["tasks"][task.name]["status"] == "running":
                    self.finish_running(task)
            statuses = {task.name: self.state["tasks"][task.name]["status"] for task in tasks}
            if all(value == "complete" for value in statuses.values()):
                return
            if "failed" in statuses.values():
                if "running" not in statuses.values():
                    self.state["status"] = "failed"
                    self.save()
                    raise RuntimeError("Experiment failed; dependent stages were not started.")
                time.sleep(self.poll_seconds)
                continue
            try:
                idle = (shared_gpu_indices(self.gpus, self.env, self.minimum_free_gib) if self.allow_shared
                        else idle_gpu_indices(self.gpus, self.env))
                available = (idle & self.idle_previous) - {
                    entry["gpu"] for entry in self.state["tasks"].values() if entry["status"] == "running"
                }
                self.idle_previous = idle
                self.state.pop("gpu_query_error", None)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.state["gpu_query_error"] = str(error)
                self.idle_previous, available = set(), set()
            for task in tasks:
                if available and statuses[task.name] == "pending" and all(
                    self.state["tasks"][dep]["status"] == "complete" for dep in task.deps
                ):
                    if task.gpu is not None and task.gpu not in available:
                        continue
                    gpu = task.gpu if task.gpu is not None else min(available)
                    available.remove(gpu)
                    self.launch(task, gpu)
            running = any(entry["status"] == "running" for entry in self.state["tasks"].values())
            self.state["status"] = "running" if running else (
                "waiting_for_gpu_memory" if self.allow_shared else "waiting_for_idle_gpu"
            )
            self.save()
            time.sleep(self.poll_seconds)


def training(output, *, pilot=False, steps=400, resume=None, initial=None, cache=None, probability=0.5):
    return command("run.engineering_pilot", output=output, device="cuda", seed=62, sampling_seed=1062,
                   signal="data/EE4D_MISMATCH_TRAIN_PILOT_READY.json" if pilot else SIGNAL,
                   split_manifest="config/egorecover_train_pilot_split_v1.json" if pilot else SPLIT,
                   bootstrap_cache="exp/egorecover_train_pilot_v1/bootstrap.pt" if pilot else BOOTSTRAP,
                   selection_signal=SIGNAL, selection_split_manifest=SPLIT, selection_bootstrap_cache=BOOTSTRAP,
                   checkpoint=CHECKPOINT, weight_source="ema", prior_steps=steps, flow_steps=steps,
                   batch_size=32, geometry_weight=1., fk_weight=0., prior_selection="dense",
                   flow_selection="closed_loop_fk", eval_every=200, resume_incomplete=resume,
                   init_experiment=initial, history_cache=cache, replay_probability=probability)


def evaluation(label, experiment, output, deps=None):
    return Task(f"eval_{label}", command("run.evaluate_paired_development", signal=SIGNAL,
                split_manifest=SPLIT, bootstrap_cache=BOOTSTRAP, checkpoint=CHECKPOINT,
                experiment=f"{label}={experiment}", group="dev", seed=62, output=output),
                output, "eval", 108, deps or [])


def stage2_tasks(output):
    experiments = {
        "12take_400": EXPERIMENTS / "stage2_budget/12take_400",
        "12take_1200": EXPERIMENTS / "stage2_budget/12take_1200",
        "72take_400": Path("exp/egorecover_train_72take_v1/pass1"),
        "72take_1200": EXPERIMENTS / "stage2_budget/72take_1200",
    }
    tasks = []
    for group in ("12take", "72take"):
        label = f"{group}_2400"
        path = output / "stage2_budget" / label
        experiments[label] = path
        tasks.append(Task(f"train_{label}", training(path, pilot=group == "12take", steps=2400,
                          resume=EXPERIMENTS / "stage2_budget/72take_2400_interrupted" if group == "72take" else None),
                          path, "train"))
    for label, experiment in experiments.items():
        deps = [f"train_{label}"] if label.endswith("2400") else []
        tasks.append(evaluation(label, experiment, output / "stage2_eval" / label, deps))
    return tasks, experiments


def replay_selection(summary):
    scores = summary["scores"]
    per_take = {
        label: {take: sum(scores[f"{label}/{mode}"]["per_take_macro_smpl22_mm"][take]
                         for mode in ("gaussian", "history")) / 2 for take in summary["takes"]}
        for label in ("before_replay", "replay_0", "replay_0p25", "replay_0p5")
    }
    means = {label: sum(values.values()) / len(values) for label, values in per_take.items()}
    labels = ("replay_0", "replay_0p25", "replay_0p5")
    best = min(labels, key=lambda label: (means[label], labels.index(label)))
    best_replay = min(labels[1:], key=lambda label: (means[label], labels.index(label)))
    pairs = {}
    for label in labels:
        for baseline in ("before_replay", "replay_0"):
            if label == baseline:
                continue
            differences = [per_take[label][take] - per_take[baseline][take] for take in summary["takes"]]
            pairs[f"{label} minus {baseline}"] = {"mean_mm": sum(differences) / len(differences),
                                                   "take_bootstrap_95ci_mm": interval(differences)}
    return {"two_source_mean_mm": means, "best_probability_label": best, "paired": pairs,
            "replay_improvement_supported": pairs[f"{best_replay} minus replay_0"]["take_bootstrap_95ci_mm"][1] < 0,
            "scope": "single-seed development comparison; no holdout or Q training", "holdout_used": False}


def append_log(title, lines):
    # Full per-checkpoint curves live in each experiment's selection.json.
    if title == "独立 P 选点":
        return
    # Append only, preserving all earlier user edits and historical runs.
    def linked_path(match):
        target = match.group(1)
        if not (ROOT / target).exists():
            return match.group(0)
        return f"[{target}]({quote(target, safe='/')})"
    lines = [re.sub(r"`((?:exp|verification|docs)/[^`\n]+)`", linked_path, line) for line in lines]
    with (ROOT / "LOG.md").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d--%H：%M")
        stream.write(f"\n## {timestamp}：{title}\n\n" + "\n".join(f"- {line}" for line in lines) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def write_results(output, budget_summary, choice, replay_summary, replay_choice, diagnostic_tasks):
    lines = ["# 阶段 2–4 开发实验结果", "",
             "共同 12-take dev；clean/freeze_3s/drift_0p03mps；SMPL22 FK MPJPE，mm，越低越好。",
             "单训练种子，不使用 holdout；四动作结果属于同状态诊断。", "",
             "## 阶段 2：预算网格", "", "| 预算 | Gaussian | History | P-only | 双来源均值 |",
             "|---|---:|---:|---:|---:|"]
    for label, case in sorted(choice["cases"].items()):
        prior = budget_summary["scores"][f"{label}/prior_only"]["macro_smpl22_mm"]
        lines.append(f"| {label} | {case['gaussian_mm']:.3f} | {case['history_mm']:.3f} | {prior:.3f} | {case['two_source_mean_mm']:.3f} |")
    lines += ["", f"入选预算：12-take/{choice['winners']['12take']['steps']}；72-take/{choice['winners']['72take']['steps']}。",
              "", "## 阶段 3：额外 400 步 replay 对照", "",
              "| 配置 | Gaussian | History | P-only | 双来源均值 |", "|---|---:|---:|---:|---:|"]
    for label, mean in replay_choice["two_source_mean_mm"].items():
        scores = [replay_summary["scores"][f"{label}/{mode}"]["macro_smpl22_mm"] for mode in MODES]
        lines.append(f"| {label} | {scores[0]:.3f} | {scores[1]:.3f} | {scores[2]:.3f} | {mean:.3f} |")
    lines += ["", f"三组最低分：{replay_choice['best_probability_label']}。",
              "相对 p=0 有明确配对改善证据。" if replay_choice["replay_improvement_supported"] else "相对 p=0 的 replay 改善证据不足。",
              "", "| 配对比较 | 均值差 | take 配对 95% 区间 |", "|---|---:|---|"]
    for label, pair in replay_choice["paired"].items():
        lo, hi = pair["take_bootstrap_95ci_mm"]
        lines.append(f"| {label} | {pair['mean_mm']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "## 阶段 4：四动作上限", "",
              "动作顺序为 a11/a10/a01/a00，视觉位在前。正 oracle 增益表示低于 a11 误差。", "",
              "| 划分/来源/变体 | 四动作宏平均误差 | 逐噪声 oracle 增益 | 先平均噪声 oracle 增益 |",
              "|---|---|---:|---:|"]
    for task in diagnostic_tasks:
        report = json.loads(task.report.read_text())
        for label, group in report["draw_averaged"]["groups"].items():
            if label.count("/") != 1:
                continue
            errors = ", ".join(f"{value:.3f}" for value in group["macro_action_error_mm"])
            lines.append(f"| {report['group']}/{label} | {errors} | {group['macro_per_draw_oracle_gain_mm']:.3f} | {group['macro_oracle_gain_after_draw_mean_mm']:.3f} |")
    lines += ["", "逐 take、故障阶段、动作频率和故障差值见同目录各阶段 JSON 报告。", ""]
    path = output / "results.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)


def complete(queue):
    output = queue.output
    tasks, experiments = stage2_tasks(output)
    for label, experiment in experiments.items():
        if not label.endswith("2400"):
            verify_task(Task(label, [], experiment, "train"))
    queue.run(tasks)
    reports = [task.report for task in tasks if task.kind == "eval"]
    budget_summary = summarize(reports)
    atomic_json(output / "stage2_summary.json", budget_summary)
    choice = select(budget_summary, [f"{label}={label}" for label in experiments])
    atomic_json(output / "stage2_selection.json", choice)
    selected_label = choice["winners"]["72take"]["label"]
    selected = experiments[selected_label]
    append_log("阶段 2 六格预算完成", [f"运行目录：`{output}`；汇总与选择：`stage2_summary.json`、`stage2_selection.json`。",
                "共同 dev 三变体双来源均值（mm）：" + "；".join(
                    f"{label}={case['two_source_mean_mm']:.3f}" for label, case in sorted(choice['cases'].items())) + "。",
                f"分别入选：12-take/{choice['winners']['12take']['steps']}、72-take/{choice['winners']['72take']['steps']}；holdout 未使用。"])

    cache_tasks = []
    for group, count in (("train", 48), ("dev", 12)):
        for shard in range(4):
            path = output / "histories" / f"{group}_shard{shard}"
            cache_tasks.append(Task(f"cache_{group}_{shard}", command("run.collect_predicted_histories",
                              experiment=selected, output=path, signal=SIGNAL, split_manifest=SPLIT,
                              bootstrap_cache=BOOTSTRAP, group=group, shard_count=4, shard_index=shard, seed=62),
                              path, "cache", count // 4 * 3 * 2 * 180))
    queue.run(cache_tasks)
    from run.merge_predicted_histories import merge
    for group in ("train", "dev"):
        path = output / "histories" / group
        if not (path / "report.json").exists():
            if path.exists():
                path.rename(path.with_name(path.name + f".interrupted_{time.time_ns()}"))
            merge([output / "histories" / f"{group}_shard{i}" for i in range(4)], path)
        verify_task(Task(f"merged_{group}", [], path, "cache", 51840 if group == "train" else 12960))

    replay_tasks = []
    for label, probability in (("replay_0", 0.), ("replay_0p25", .25), ("replay_0p5", .5)):
        path = output / "stage3_train" / label
        replay_tasks.append(Task(f"train_{label}", training(path, initial=selected,
                                 cache=output / "histories/train/frames.pt", probability=probability), path, "train"))
        replay_tasks.append(evaluation(label, path, output / "stage3_eval" / label, [f"train_{label}"]))
    queue.run(replay_tasks)
    # Reuse the completed pre-replay evaluation by changing only its presentation label.
    before = json.loads((output / "stage2_eval" / selected_label / "report.json").read_text())
    before["experiments"] = {"before_replay": str(selected)}
    for row in before["results"]:
        row["experiment"] = "before_replay"
    before_path = output / "stage3_before_replay.json"
    atomic_json(before_path, before)
    replay_summary = summarize([before_path] + [task.report for task in replay_tasks if task.kind == "eval"])
    atomic_json(output / "stage3_summary.json", replay_summary)
    replay_choice = replay_selection(replay_summary)
    atomic_json(output / "stage3_comparison.json", replay_choice)
    append_log("阶段 3 replay 对照完成", [f"运行目录：`{output}`；训练缓存 51,840 帧；三组 P/G 各额外训练 400 步。",
                "共同 dev 双来源均值（mm）：" + "；".join(
                    f"{label}={mean:.3f}" for label, mean in replay_choice['two_source_mean_mm'].items()) + "。",
                f"共同 dev 配对汇总：`stage3_summary.json`；综合比较：`stage3_comparison.json`。",
                f"三组最低分：{replay_choice['best_probability_label']}；相对 p=0 的 replay 改善证据：{replay_choice['replay_improvement_supported']}。"])

    diagnostic_tasks = []
    for group, count in (("train", 48), ("dev", 12)):
        path = output / "stage4" / f"{group}.json"
        diagnostic_tasks.append(Task(f"diagnose_{group}", command("run.diagnose_four_actions",
                                signal=SIGNAL, split_manifest=SPLIT, checkpoint=CHECKPOINT, experiment=selected,
                                cache=output / "histories" / group / "frames.pt", group=group,
                                max_takes=count, draws=3, seed=62, output=path), path, "diagnostic", count * 3 * 2 * 3 * 3))
    queue.run(diagnostic_tasks)
    write_results(output, budget_summary, choice, replay_summary, replay_choice, diagnostic_tasks)
    append_log("阶段 4 四动作诊断完成", [f"报告：`{output}/stage4/train.json`、`{output}/stage4/dev.json`。",
                f"完整结果表：`{output}/results.md`。",
                "固定阶段 2 入选 P/G，train/dev 分别覆盖 48/12 takes，2,592/648 条状态×噪声记录；包含逐噪声和先平均噪声的 oracle。",
                "阶段 2–4 已完成；这些是单训练种子开发实验及同状态诊断，未训练 Q，未使用 holdout。"])
    queue.state["status"] = "complete"
    queue.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--allow-shared", action="store_true", help="Allow selected GPUs already occupied by other jobs")
    parser.add_argument("--minimum-free-gib", type=float, default=32,
                        help="Minimum free GPU memory before launching a shared job")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed jobs after addressing their logged cause")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    os.chdir(ROOT)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not set(args.gpus).issubset(set(range(8))):
        parser.error("Use unique GPU indices within 0–7.")
    if not 0 < args.minimum_free_gib < 96:
        parser.error("Minimum free memory must be positive and below 96 GiB.")
    if not 10 <= args.poll_seconds <= 60:
        parser.error("Poll interval must be between 10 and 60 seconds.")
    if args.dry_run:
        for task in stage2_tasks(args.output)[0]:
            print(task.name, shlex.join(task.argv), "dependencies:", task.deps)
        return
    queue = Queue(args.output, args.gpus, args.poll_seconds,
                  allow_shared=args.allow_shared, minimum_free_gib=args.minimum_free_gib)
    if args.retry_failed:
        for entry in queue.state["tasks"].values():
            if entry["status"] == "failed":
                entry["status"] = "pending"
        queue.save()
    policy = f"shared, at least {args.minimum_free_gib:g} GiB free" if args.allow_shared else "idle only"
    queue.log(f"Queue started; GPUs {args.gpus}, policy {policy}. Status: {queue.path}")
    try:
        complete(queue)
    except Exception as error:
        queue.state.update({"status": "failed", "error": repr(error)})
        queue.save()
        append_log("阶段 2–4 调度停止", [f"运行目录：`{args.output}`；原因：`{error}`；未完成项不得计入结果。"])
        raise


if __name__ == "__main__":
    main()
