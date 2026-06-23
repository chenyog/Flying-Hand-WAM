import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks(robotwin_root: Path, cfg: DictConfig) -> list[str]:
    dataset_dirs = list(cfg.data.train.get("dataset_dirs", []))
    if len(dataset_dirs) > 1:
        data_root = robotwin_root / "data"
        tasks = []
        for dataset_dir in dataset_dirs:
            name = Path(str(dataset_dir)).name
            matches = sorted(path for path in data_root.glob(f"*/{name}") if path.is_dir())
            tasks.append(matches[0].relative_to(data_root).as_posix() if matches else name)
        return tasks

    eval_step_limit_file = robotwin_root / "task_config" / "_eval_step_limit.yml"
    if not eval_step_limit_file.exists():
        raise FileNotFoundError(f"Task list file not found: {eval_step_limit_file}")
    with eval_step_limit_file.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {eval_step_limit_file}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = output_dir / "manager.log"
    failed_tasks_file = output_dir / "failed_tasks.txt"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks(robotwin_root, cfg)
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()
    config_name = HydraConfig.get().job.config_name

    task_rates: dict[str, float | None] = {task: None for task in tasks}
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int) -> list[str]:
        return [
            sys.executable,
            str(SINGLE_ENTRY),
            "--config-name",
            str(config_name),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={str(cfg.EVALUATION.task_config)}",
            f"EVALUATION.output_dir={str(output_dir)}",
            *extra_overrides,
        ]

    def launch_task(task_name: str, gpu_id: int) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id)
        log(f"launch task={task_name} gpu={gpu_id} cmd={' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            running_states.append(launch_task(task_name=pending_tasks.popleft(), gpu_id=gpu_id))

    def write_outputs() -> None:
        mean = _mean_or_none([task_rates[t] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "success_rate"])
            for task in tasks:
                writer.writerow([task, task_rates[task]])
            writer.writerow(["__overall__", mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "success_rate": _to_jsonable(task_rates[task]),
                }
                for task in tasks
            ],
            "overall": {
                "mean_success_rate": _to_jsonable(mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = output_dir / state.task_name / "_result.txt"
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name] = success_rate
            log(f"done task={state.task_name} gpu={gpu_id} success_rate={success_rate:.4f}")
            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
