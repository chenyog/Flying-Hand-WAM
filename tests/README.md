# Flying-Hand tests

Run commands from the repository root. Test and diagnostic outputs default to
`tests/results/`; the directory contents are intentionally ignored by Git.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_flying_hand_minco_planner.py -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_flying_hand_grasp_validation.py -v
.venv/bin/python tests/test_render.py
.venv/bin/python tests/build_flying_hand_minco_cpp.py
.venv/bin/python tests/check_flying_hand_grasp_stability.py
.venv/bin/python tests/collect_flying_hand_l1_debug.py
.venv/bin/python tests/summarize_flying_hand_planner_test.py \
  --input-dir tests/results/flying_hand_grasp_stability \
  --tasks move_bottle --seeds 0
.venv/bin/python tests/summarize_flying_hand_policy_evaluation.py \
  --input-dir /path/to/policy/evaluation
```

Explicit `--output-dir` arguments remain supported when a separate result root
is needed for a specific experiment.
