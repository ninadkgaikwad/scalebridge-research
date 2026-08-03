# -*- coding: utf-8 -*-
"""Validate a C9 Phase C MLflow parent run and its nested children."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import argparse
import json
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage-runs", type=int, default=8)
    parser.add_argument("--expected-training-task-runs", type=int, default=None)
    parser.add_argument("--expected-evaluation-task-runs", type=int, default=None)
    parser.add_argument("--expected-inference-task-runs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registration = json.loads(
        args.registration_manifest.read_text(encoding="utf-8")
    )
    tracking_uri = registration["tracking_uri"]
    experiment_id = registration["experiment_id"]
    parent_run_id = registration["parent_run_id"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    parent = client.get_run(parent_run_id)
    children = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=10000,
    )

    stage_children = [
        run for run in children
        if str(run.data.tags.get("phase_c_stage", "")) in {
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"
        }
    ]
    stage_by_name = {
        str(run.data.tags.get("phase_c_stage", "")): run
        for run in stage_children
    }

    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        max_results=10000,
    )
    task_runs = [
        run for run in all_runs
        if str(run.data.tags.get("run_kind", "")) in {
            "model_training", "model_evaluation", "zone_inference"
        }
        and str(run.data.tags.get("phase_c_run_id", ""))
        == registration["phase_c_run_id"]
    ]

    training = [
        run for run in task_runs
        if run.data.tags.get("run_kind") == "model_training"
    ]
    evaluation = [
        run for run in task_runs
        if run.data.tags.get("run_kind") == "model_evaluation"
    ]
    inference = [
        run for run in task_runs
        if run.data.tags.get("run_kind") == "zone_inference"
    ]

    expected_task_parents = {
        "model_training": stage_by_name.get("C6"),
        "model_evaluation": stage_by_name.get("C7"),
        "zone_inference": stage_by_name.get("C8"),
    }
    misplaced_tasks = []
    for run in task_runs:
        kind = str(run.data.tags.get("run_kind", ""))
        expected_parent = expected_task_parents.get(kind)
        actual_parent = str(run.data.tags.get("mlflow.parentRunId", ""))
        if expected_parent is None or actual_parent != expected_parent.info.run_id:
            misplaced_tasks.append(run.info.run_id)

    availability_summary = registration.get("availability_summary", {})
    if not isinstance(availability_summary, dict):
        availability_summary = {}

    expected_training_task_runs = (
        args.expected_training_task_runs
        if args.expected_training_task_runs is not None
        else _optional_int(availability_summary.get("trained_model_count"))
    )
    expected_evaluation_task_runs = (
        args.expected_evaluation_task_runs
        if args.expected_evaluation_task_runs is not None
        else _optional_int(availability_summary.get("evaluated_model_count"))
    )
    expected_inference_task_runs = (
        args.expected_inference_task_runs
        if args.expected_inference_task_runs is not None
        else _optional_int(availability_summary.get("inference_zone_count"))
    )

    checks = {
        "parent_run_exists": parent.info.run_id == parent_run_id,
        "parent_status_finished": parent.info.status == "FINISHED",
        "stage_run_count": len(stage_children) == args.expected_stage_runs,
        "training_task_run_count": (
            expected_training_task_runs is None
            or len(training) == expected_training_task_runs
        ),
        "evaluation_task_run_count": (
            expected_evaluation_task_runs is None
            or len(evaluation) == expected_evaluation_task_runs
        ),
        "inference_task_run_count": (
            expected_inference_task_runs is None
            or len(inference) == expected_inference_task_runs
        ),
        "machine_id_tag_present": bool(
            parent.data.tags.get("machine_id", "")
        ),
        "all_stage_runs_unique": len(stage_by_name) == args.expected_stage_runs,
        "task_runs_nested_under_correct_stage": len(misplaced_tasks) == 0,
        "availability_summary_present": isinstance(
            registration.get("availability_summary"), dict
        ),
    }

    print("=" * 100)
    print("C9 PHASE C MLFLOW VALIDATION")
    print("=" * 100)
    print(f"tracking_uri: {tracking_uri}")
    print(f"experiment_id: {experiment_id}")
    print(f"parent_run_id: {parent_run_id}")
    print(f"stage_run_count: {len(stage_children)}")
    print(f"training_task_run_count: {len(training)}")
    print(f"evaluation_task_run_count: {len(evaluation)}")
    print(f"inference_task_run_count: {len(inference)}")
    print(f"misplaced_task_run_count: {len(misplaced_tasks)}")
    print(f"expected_training_task_run_count: {expected_training_task_runs}")
    print(f"expected_evaluation_task_run_count: {expected_evaluation_task_runs}")
    print(f"expected_inference_task_run_count: {expected_inference_task_runs}")
    print(f"availability_summary: {registration.get('availability_summary', {})}")
    print("")
    for key, passed in checks.items():
        print(f"{key}: {'passed' if passed else 'failed'}")

    failed = [key for key, value in checks.items() if not value]
    print("")
    print(f"validation_status: {'passed' if not failed else 'failed'}")
    if failed:
        print(f"failed_checks: {failed}")
        return 1
    return 0



def _optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
