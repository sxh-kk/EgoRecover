"""Build and fully audit every official EE4D validation sequence."""

from datetime import datetime
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import torch

from .ee4d_mismatch.audit import audit_dataset
from .ee4d_mismatch.build import build_dataset
from .ee4d_mismatch.source import EE4DSource, file_sha256, write_json
from .wait_for_dataset import ready


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUTPUT = DATA / "ee4d_mismatch_val_full_v1"
STATUS = DATA / ".ee4d_full_build" / "status.json"


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def update(phase, **fields):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
    previous.update(phase=phase, updated_at=timestamp(), **fields)
    temporary = STATUS.with_suffix(".tmp")
    write_json(temporary, previous)
    temporary.replace(STATUS)
    print(f"{timestamp()} {phase}: {fields}", flush=True)


def publish(signal, path):
    if path.exists():
        raise FileExistsError(f"Completion signal already exists: {path}")
    temporary = path.with_suffix(".tmp")
    write_json(temporary, signal)
    ready(temporary)
    temporary.replace(path)


def main():
    torch.set_num_threads(2)
    update("loading_source", started_at=timestamp(), output=str(OUTPUT))
    source = EE4DSource(DATA / "ee4d_motion_uniegomotion")
    download = json.loads((DATA / ".ee4d_validation_download" / "status.json").read_text())
    if download["phase"] != "complete" or len(download["verified_files"]) != 5:
        raise ValueError("The five source assets have not passed download verification")
    if OUTPUT.exists():
        raise FileExistsError(f"Full dataset version is already present: {OUTPUT}")

    update("building", source_sequences=len(source.motion), source_takes=len({name.rsplit("___", 2)[0]
                                                                             for name in source.motion}))
    build_dataset(source, OUTPUT, all_sequences=True, profile="pilot", purpose="engineering")
    update("full_audit", output=str(OUTPUT))
    report = audit_dataset(OUTPUT, source=source, verify_source=True)
    expected = dict(num_takes=560, num_episodes=5236, num_variants=36652,
                    source_sequences_covered=5236, physical_frames_covered=932525)
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f"Full validation coverage mismatch: {key}={report[key]}, expected {value}")
    if report["status"] != "passed" or not report["source_hashes_verified"]:
        raise ValueError("Full validation audit did not pass")
    write_json(OUTPUT / "audit" / "validation.json", report)

    spec = json.loads((OUTPUT / "spec.json").read_text())
    if spec["episode_scope"] != "all_source_sequences" or spec["frames"] is not None:
        raise ValueError("The artifact has the wrong full-corpus schema")
    original = {str(Path(item["path"]).relative_to(source.root)): item for item in download["verified_files"]}
    for name, item in spec["source_files"].items():
        if item["sha256"] != original[name]["sha256"] or item["bytes"] != original[name]["bytes"]:
            raise ValueError(f"Official source differs from downloaded data: {name}")
    junit = PROJECT / "data_pipeline" / "test_results_full.xml"
    suites = ET.parse(junit).getroot().findall("testsuite")
    passed = sum(int(s.attrib["tests"]) for s in suites)
    if passed < 24 or any(int(s.attrib[key]) for s in suites for key in ("errors", "failures", "skipped")):
        raise ValueError("Full-corpus processing tests have not passed")

    episodes = json.loads((OUTPUT / "audit" / "construction_summary.json").read_text())["episodes"]
    short = sum(episode["num_frames"] < 200 for episode in episodes)
    clipped = sum(episode["available_fault_frames"] < 30 for episode in episodes)
    shortened_recovery = sum(episode["minimum_recovery_frames"] < 80 for episode in episodes)
    handoff = OUTPUT / "HANDOFF.md"
    handoff.write_text(
        "# 完整验证集错配数据交接\n\n"
        f"构建完成：{timestamp()}。原始 560 个 take、5,236 个序列全部覆盖，"
        "每个序列完整保留原始长度；7 个配对变体，共 36,652 条、6,527,675 个变体帧。\n\n"
        f"短于 200 帧的序列 {short} 条；30 帧故障被截短的序列 {clipped} 条；"
        f"不足 80 帧恢复期的序列 {shortened_recovery} 条。每条实际参数记录在 manifest 中。\n\n"
        "目录内的 `engineering.jsonl` 是全部样本索引，`spec.json` 是格式定义，"
        "`audit/validation.json` 记录逐样本校验结果。输入读取：\n\n"
        "```python\n"
        "from data_pipeline.ee4d_mismatch.dataset import MismatchDataset\n"
        f"dataset = MismatchDataset({str(OUTPUT)!r})\n"
        "observations = dataset.observations(0, as_of=20, history_frames=20)\n"
        "labels = dataset.supervision(0)\n"
        "```\n\n"
        "`observations` 只提供当前及历史的视觉特征、9D 绝对头部轨迹与可用性；"
        "故障 mask、源索引和标签必须使用离线接口读取。"
        "全部数据源为官方 val；如果用这些 take 调参，最终独立评估需要另外保留数据。\n"
    )
    manifest = OUTPUT / "engineering.jsonl"
    full_entry = dict(path=str(OUTPUT), profile="pilot", scope="all_source_sequences",
                      num_takes=report["num_takes"], num_sequences=report["source_sequences_covered"],
                      num_variants=report["num_variants"], physical_frames=report["physical_frames_covered"],
                      manifest=str(manifest), audit_report=str(OUTPUT / "audit" / "validation.json"),
                      spec_sha256=file_sha256(OUTPUT / "spec.json"), manifest_sha256=file_sha256(manifest))
    full_signal = dict(status="complete", message="完整验证集错配数据集构建完成",
                       completed_at=timestamp(), source_root=str(source.root),
                       datasets=[full_entry], handoff=str(handoff),
                       tests=dict(passed=passed, failed=0, report=str(junit), sha256=file_sha256(junit)),
                       usage="官方 val 完整覆盖；用于工程检查和方法开发时，应另留独立最终评估数据。")
    update("publishing", report=report)
    publish(full_signal, DATA / "EE4D_MISMATCH_FULL_READY.json")
    previous_path = DATA / "EE4D_MISMATCH_READY.json"
    previous = json.loads(previous_path.read_text())
    previous["datasets"] = [item for item in previous["datasets"] if item["path"] != str(OUTPUT)] + [full_entry]
    previous.update(message=full_signal["message"], completed_at=full_signal["completed_at"],
                    next_step=f"读取完整验证集交接文档 {handoff}")
    temporary = previous_path.with_suffix(".tmp")
    write_json(temporary, previous)
    ready(temporary)
    temporary.replace(previous_path)
    update("complete", output=str(OUTPUT), report=report,
           signal=str(DATA / "EE4D_MISMATCH_FULL_READY.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update("error", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
