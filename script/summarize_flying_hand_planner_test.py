#!/usr/bin/env python3
"""Strictly validate and summarize Flying-Hand planner test JSONL results."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


STATUS_NAMES = (
    "stable",
    "review",
    "unstable",
    "success",
    "task_failed",
    "error",
)
Z_95 = 1.959963984540054


class CoverageError(RuntimeError):
    """Raised when worker output does not match the requested task/seed grid."""


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _wilson_95(successes, total):
    if total == 0:
        return {"low": None, "high": None}
    proportion = successes / total
    denominator = 1.0 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2.0 * total)) / denominator
    margin = Z_95 * math.sqrt(
        proportion * (1.0 - proportion) / total + Z_95**2 / (4.0 * total**2)
    ) / denominator
    return {"low": center - margin, "high": center + margin}


def _read_rows(input_dir):
    worker_paths = sorted((input_dir / "workers").glob("*.jsonl"))
    if not worker_paths:
        raise CoverageError(f"no worker JSONL files found under {input_dir / 'workers'}")
    rows = []
    for path in worker_paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CoverageError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise CoverageError(f"expected a JSON object in {path}:{line_number}")
                if "task" not in row or "seed" not in row:
                    raise CoverageError(f"missing task or seed in {path}:{line_number}")
                try:
                    pair = (str(row["task"]), int(row["seed"]))
                except (TypeError, ValueError) as exc:
                    raise CoverageError(f"invalid task/seed in {path}:{line_number}") from exc
                rows.append((pair, row, path, line_number))
    return rows


def _validate_coverage(indexed_rows, tasks, seeds):
    expected = {(task, seed) for task in tasks for seed in seeds}
    by_pair = defaultdict(list)
    for pair, row, path, line_number in indexed_rows:
        by_pair[pair].append((row, path, line_number))
    observed = set(by_pair)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = {
        pair: [(str(path), line_number) for _, path, line_number in records]
        for pair, records in by_pair.items()
        if len(records) > 1
    }
    if missing or unexpected or duplicates:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unexpected:
            parts.append(f"unexpected={unexpected}")
        if duplicates:
            parts.append(f"duplicates={duplicates}")
        raise CoverageError("coverage validation failed: " + "; ".join(parts))
    return [by_pair[(task, seed)][0][0] for task in tasks for seed in seeds]


def _planner_values(row):
    summary = row.get("planner_summary") or {}
    plans = row.get("minco_plans") or []
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(plans, list):
        plans = []
    initial_cost = 0.0
    final_cost = 0.0
    failed_phases = 0
    reach_checked_phases = 0
    reach_failed_phases = 0
    reach_wait_time_s = 0.0
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        initial_cost += _number(plan.get("initial_cost"))
        final_cost += _number(plan.get("final_cost"))
        failed_phases += int(not bool(plan.get("success", False)))
        if "reach_succeeded" in plan:
            reach_checked_phases += 1
            reach_failed_phases += int(not bool(plan.get("reach_succeeded", False)))
            reach_wait_time_s += _number(plan.get("reach_wait_time_s"))
    return {
        "phase_count": int(_number(summary.get("phase_count"), len(plans))),
        "optimized_flight_time_s": _number(summary.get("total_optimized_flight_time_s")),
        "iterations": int(_number(summary.get("total_iterations"))),
        "function_evaluations": int(_number(summary.get("total_function_evaluations"))),
        "planner_failed": bool(summary.get("any_failed", False)),
        "failed_phase_count": int(_number(summary.get("failed_phase_count"), failed_phases)),
        "reach_checked_phase_count": int(_number(summary.get("reach_checked_phase_count"), reach_checked_phases)),
        "reach_failed_phase_count": int(_number(summary.get("reach_failed_phase_count"), reach_failed_phases)),
        "reach_wait_time_s": _number(summary.get("total_reach_wait_time_s"), reach_wait_time_s),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
    }


def _failure_reasons(rows):
    reasons = Counter()
    for row in rows:
        row_reasons = set()
        status = str(row.get("status", "error"))
        if status == "error":
            row_reasons.add(f"error:{row.get('error_type', 'unknown')}")
        if not bool(row.get("task_success", False)):
            row_reasons.add("task_check_success_false")
        if bool(row.get("task_failed_flag", False)):
            row_reasons.add("task_failed_flag")
        for key in ("unstable_reasons", "review_reasons"):
            for reason in row.get(key, []) or []:
                row_reasons.add(str(reason))
        for plan in row.get("minco_plans", []) or []:
            if isinstance(plan, dict) and not bool(plan.get("success", False)):
                phase = plan.get("phase", "unknown_phase")
                message = plan.get("message", "unknown")
                row_reasons.add(f"minco:{phase}:{message}")
            if isinstance(plan, dict) and plan.get("reach_succeeded") is False:
                phase = plan.get("phase", "unknown_phase")
                row_reasons.add(f"reach_timeout:{phase}")
        reasons.update(row_reasons)
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    ]


def _aggregate(rows):
    total = len(rows)
    successes = sum(bool(row.get("task_success", False)) for row in rows)
    status_counts = Counter(str(row.get("status", "error")) for row in rows)
    planner = Counter()
    for row in rows:
        values = _planner_values(row)
        planner["phase_count"] += values["phase_count"]
        planner["optimized_flight_time_s"] += values["optimized_flight_time_s"]
        planner["iterations"] += values["iterations"]
        planner["function_evaluations"] += values["function_evaluations"]
        planner["failed_job_count"] += int(values["planner_failed"])
        planner["failed_phase_count"] += values["failed_phase_count"]
        planner["reach_checked_phase_count"] += values["reach_checked_phase_count"]
        planner["reach_failed_phase_count"] += values["reach_failed_phase_count"]
        planner["reach_wait_time_s"] += values["reach_wait_time_s"]
        planner["initial_cost"] += values["initial_cost"]
        planner["final_cost"] += values["final_cost"]
    result = {
        "jobs": total,
        "task_successes": successes,
        "task_success_rate": successes / total if total else None,
        "task_success_wilson_95": _wilson_95(successes, total),
        "status_counts": {name: int(status_counts[name]) for name in STATUS_NAMES},
        "other_status_counts": {
            name: int(count) for name, count in sorted(status_counts.items()) if name not in STATUS_NAMES
        },
        "planner": dict(planner),
    }
    result["planner"]["cost_reduction"] = (
        result["planner"]["initial_cost"] - result["planner"]["final_cost"]
    )
    return result


def _csv_row(task, aggregate):
    interval = aggregate["task_success_wilson_95"]
    planner = aggregate["planner"]
    statuses = aggregate["status_counts"]
    return {
        "task": task,
        "jobs": aggregate["jobs"],
        "task_successes": aggregate["task_successes"],
        "task_success_rate": aggregate["task_success_rate"],
        "wilson_95_low": interval["low"],
        "wilson_95_high": interval["high"],
        **{status: statuses[status] for status in STATUS_NAMES},
        "planner_phase_count": planner["phase_count"],
        "planner_optimized_flight_time_s": planner["optimized_flight_time_s"],
        "planner_iterations": planner["iterations"],
        "planner_function_evaluations": planner["function_evaluations"],
        "planner_failed_job_count": planner["failed_job_count"],
        "planner_failed_phase_count": planner["failed_phase_count"],
        "planner_reach_checked_phase_count": planner["reach_checked_phase_count"],
        "planner_reach_failed_phase_count": planner["reach_failed_phase_count"],
        "planner_reach_wait_time_s": planner["reach_wait_time_s"],
        "planner_initial_cost": planner["initial_cost"],
        "planner_final_cost": planner["final_cost"],
        "planner_cost_reduction": planner["cost_reduction"],
    }


def _write_report(path, total, task_rows, failure_reasons):
    total_row = _csv_row("total", total)
    fields = list(total_row)
    lines = ["# Flying-Hand planner test summary", "", "## Overall", ""]
    for field in fields[1:]:
        lines.append(f"- `{field}`: {total_row[field]}")
    lines.extend(["", "## Per task", "", "| " + " | ".join(fields) + " |"])
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in task_rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines.extend(["", "## Failure reasons", ""])
    if failure_reasons:
        for item in failure_reasons:
            lines.append(f"- `{item['reason']}`: {item['count']}")
    else:
        lines.append("- None recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(input_dir, output_dir, tasks, seeds):
    tasks = list(dict.fromkeys(tasks))
    seeds = list(dict.fromkeys(seeds))
    if not tasks or not seeds:
        raise CoverageError("--tasks and --seeds must both be non-empty")
    rows = _validate_coverage(_read_rows(input_dir), tasks, seeds)
    rows_by_task = {task: [row for row in rows if row["task"] == task] for task in tasks}
    total = _aggregate(rows)
    per_task = {task: _aggregate(rows_by_task[task]) for task in tasks}
    failure_reasons = _failure_reasons(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "coverage": {"expected_jobs": len(tasks) * len(seeds), "observed_jobs": len(rows), "tasks": tasks, "seeds": seeds},
        "total": total,
        "tasks": per_task,
        "failure_reasons": failure_reasons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_rows = [_csv_row("total", total)] + [_csv_row(task, per_task[task]) for task in tasks]
    with (output_dir / "task_success.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _write_report(output_dir / "report.md", total, csv_rows[1:], failure_reasons)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Stability collector output root containing workers/*.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    try:
        summary = summarize(args.input_dir.resolve(), args.output_dir.resolve(), args.tasks, args.seeds)
    except (CoverageError, OSError) as exc:
        raise SystemExit(f"summary failed: {exc}") from exc
    print(json.dumps(summary["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
