#!/usr/bin/env python3
"""Summarize FastWAM Flying-Hand evaluation and safety diagnostics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(input_dirs):
    rows = []
    for input_dir in input_dirs:
        for path in sorted(input_dir.rglob("policy_diagnostics.jsonl")):
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                    row["_source"] = str(path)
                    rows.append(row)
    if not rows:
        raise ValueError(f"no policy_diagnostics.jsonl found under {input_dirs}")
    return rows


def _aggregate(rows):
    pitches = [float(row["flight"]["max_abs_pitch_deg"]) for row in rows]
    grasp_events = [event for row in rows for event in row.get("grasp_events", [])]
    command_latency_values = [
        float(event["command_latency_seconds"])
        for event in grasp_events
        if event.get("command_latency_seconds") is not None
    ]
    close_events = [event for event in grasp_events if event.get("event") == "close"]
    completed_close_events = [
        event
        for event in close_events
        if event.get("completed", True)
        and (
            event.get("gripper_close_commanded", False)
            or event.get("gripper_closed", False)
        )
    ]
    attachments = [event for event in close_events if event.get("attachment") == "attached"]
    rejected_captures = [
        event
        for event in close_events
        if event.get("attachment") == "rejected_box_center_outside_grasp_region"
    ]
    cancelled_transitions = [
        event for event in grasp_events if event.get("cancelled", False)
    ]
    pending_transitions = [
        event
        for event in grasp_events
        if not event.get("completed", True) and not event.get("cancelled", False)
    ]
    invalid_attachments = []
    for event in attachments:
        attached_name = event.get("attached_actor")
        matching = [actor for actor in event.get("actors", []) if actor.get("actor") == attached_name]
        if (
            len(matching) != 1
            or not matching[0].get("center_inside", False)
        ):
            invalid_attachments.append(event)
    waypoint_tracks = [
        track
        for row in rows
        for track in row.get("waypoint_tracking", [])
    ]
    max_bodyrate = [
        max(
            float(row["flight"].get("max_abs_bodyrate_rad_s", [0.0, 0.0, 0.0])[axis])
            for row in rows
        )
        for axis in range(3)
    ]
    episodes_with_close_intent = sum(
        int(row.get("actions", {}).get("above_close_threshold", 0)) > 0
        for row in rows
    )
    episodes_with_physical_close = 0
    episodes_with_attachment = 0
    failed_after_attachment = 0
    for row in rows:
        episode_closes = [
            event for event in row.get("grasp_events", [])
            if event.get("event") == "close"
        ]
        episode_completed_closes = [
            event
            for event in episode_closes
            if event.get("completed", True)
            and (
                event.get("gripper_close_commanded", False)
                or event.get("gripper_closed", False)
            )
        ]
        episode_attached = any(
            event.get("attachment") == "attached" for event in episode_closes
        )
        episodes_with_physical_close += int(bool(episode_completed_closes))
        episodes_with_attachment += int(episode_attached)
        failed_after_attachment += int(episode_attached and not row.get("success", False))
    return {
        "episodes": len(rows),
        "successes": sum(bool(row.get("success", False)) for row in rows),
        "success_rate": sum(bool(row.get("success", False)) for row in rows) / len(rows),
        "mean_episode_max_abs_pitch_deg": sum(pitches) / len(pitches),
        "max_abs_pitch_deg": max(pitches),
        "episodes_with_large_pitch": sum(
            int(row["flight"].get("large_pitch_excursions", 0)) > 0 for row in rows
        ),
        "large_pitch_excursions": sum(
            int(row["flight"].get("large_pitch_excursions", 0)) for row in rows
        ),
        "max_abs_pitch_rate_rad_s": max(
            float(row["flight"].get("max_abs_pitch_rate_rad_s", 0.0)) for row in rows
        ),
        "max_abs_bodyrate_rad_s": max_bodyrate,
        "rotor_saturation_samples": sum(
            int(row["flight"].get("rotor_saturation_samples", 0)) for row in rows
        ),
        "max_position_error_m": max(
            float(row["flight"].get("max_position_error_m", 0.0)) for row in rows
        ),
        "close_events": len(close_events),
        "completed_close_events": len(completed_close_events),
        "grasp_edge_events": len(grasp_events),
        "min_grasp_command_latency_seconds": min(command_latency_values, default=0.0),
        "max_grasp_command_latency_seconds": max(command_latency_values, default=0.0),
        "cancelled_gripper_transitions": len(cancelled_transitions),
        "pending_gripper_transitions": len(pending_transitions),
        "episodes_with_close_intent": episodes_with_close_intent,
        "episodes_with_physical_close": episodes_with_physical_close,
        "episodes_with_attachment": episodes_with_attachment,
        "failed_after_attachment": failed_after_attachment,
        "attachments": len(attachments),
        "rejected_captures": len(rejected_captures),
        "invalid_attachments": len(invalid_attachments),
        "waypoint_tracks": len(waypoint_tracks),
        "waypoint_segments": sum(
            int(track.get("segments", 0)) for track in waypoint_tracks
        ),
        "max_waypoint_spacing_m": max(
            (float(track.get("max_waypoint_spacing_m", 0.0)) for track in waypoint_tracks),
            default=0.0,
        ),
        "max_reference_velocity_mps": max(
            (float(track.get("max_reference_velocity_mps", 0.0)) for track in waypoint_tracks),
            default=0.0,
        ),
        "max_reference_acceleration_mps2": max(
            (float(track.get("max_reference_acceleration_mps2", 0.0)) for track in waypoint_tracks),
            default=0.0,
        ),
        "max_reference_lag_m": max(
            (float(track.get("max_reference_lag_m", 0.0)) for track in waypoint_tracks),
            default=0.0,
        ),
        "total_waypoint_trajectory_seconds": sum(
            float(track.get("duration_seconds", 0.0)) for track in waypoint_tracks
        ),
    }


def summarize(input_dirs, output_dir: Path, expected_episodes: int | None):
    rows = _load_rows(input_dirs)
    by_task = defaultdict(list)
    pairs = set()
    duplicates = []
    for row in rows:
        task = str(row["task"])
        episode = int(row["episode"])
        pair = (task, episode)
        if pair in pairs:
            duplicates.append(pair)
        pairs.add(pair)
        by_task[task].append(row)
    if duplicates:
        raise ValueError(f"duplicate task/episode rows: {sorted(set(duplicates))}")
    if expected_episodes is not None:
        wrong = {
            task: len(task_rows)
            for task, task_rows in by_task.items()
            if len(task_rows) != expected_episodes
        }
        if wrong:
            raise ValueError(
                f"episode coverage mismatch; expected {expected_episodes} per task, got {wrong}"
            )

    per_task = {task: _aggregate(task_rows) for task, task_rows in sorted(by_task.items())}
    total = _aggregate(rows)
    summary = {
        "coverage": {
            "sources": [str(path) for path in input_dirs],
            "tasks": len(by_task),
            "episodes": len(rows),
            "episodes_per_task": {
                task: len(task_rows) for task, task_rows in sorted(by_task.items())
            },
        },
        "total": total,
        "tasks": per_task,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = ["task", *next(iter(per_task.values())).keys()]
    with (output_dir / "task_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"task": "__overall__", **total})
        for task, metrics_for_task in per_task.items():
            writer.writerow({"task": task, **metrics_for_task})

    report = [
        "# Flying-Hand policy evaluation",
        "",
        f"- Coverage: {len(by_task)} tasks, {len(rows)} episodes",
        f"- Success: {total['successes']}/{total['episodes']} ({total['success_rate']:.1%})",
        f"- Maximum absolute pitch: {total['max_abs_pitch_deg']:.3f} deg",
        f"- Maximum absolute body rate [x, y, z]: "
        f"{[round(value, 3) for value in total['max_abs_bodyrate_rad_s']]} rad/s",
        f"- Episodes with >=15 deg pitch excursion: {total['episodes_with_large_pitch']}",
        f"- Rotor saturation samples: {total['rotor_saturation_samples']}",
        f"- Maximum raw-waypoint/reference lag: {total['max_reference_lag_m']:.3f} m",
        f"- Maximum controller position error: {total['max_position_error_m']:.3f} m",
        f"- Invalid actor attachments: {total['invalid_attachments']}",
        f"- Valid actor attachment events: {total['attachments']}",
        f"- Rejected box-center captures: {total['rejected_captures']}",
        f"- Grasp edge command latency range: "
        f"{total['min_grasp_command_latency_seconds']:.3f}-"
        f"{total['max_grasp_command_latency_seconds']:.3f} s",
        f"- Cancelled / pending gripper transitions: "
        f"{total['cancelled_gripper_transitions']} / {total['pending_gripper_transitions']}",
        f"- Episodes with close intent / close command / attachment: "
        f"{total['episodes_with_close_intent']} / {total['episodes_with_physical_close']} / "
        f"{total['episodes_with_attachment']}",
        f"- Failed episodes after a valid attachment: {total['failed_after_attachment']}",
        f"- Waypoint segments tracked directly: {total['waypoint_segments']}",
        f"- Total waypoint trajectory time: {total['total_waypoint_trajectory_seconds']:.3f} s",
        "",
        "## Per task",
        "",
        "| Task | Success | Max pitch (deg) | Large-pitch episodes | Close-intent episodes | Attached episodes | Invalid attachments |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, task_metrics in per_task.items():
        report.append(
            f"| {task} | {task_metrics['successes']}/{task_metrics['episodes']} "
            f"({task_metrics['success_rate']:.1%}) | {task_metrics['max_abs_pitch_deg']:.3f} | "
            f"{task_metrics['episodes_with_large_pitch']} | "
            f"{task_metrics['episodes_with_close_intent']} | "
            f"{task_metrics['episodes_with_attachment']} | "
            f"{task_metrics['invalid_attachments']} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    input_dirs = [path.resolve() for path in args.input_dir]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dirs[0] / "summary"
    )
    summary = summarize(input_dirs, output_dir, args.expected_episodes)
    print(json.dumps(summary["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
