# Flying-Hand one-shot evaluation perturbations

The feature is disabled unless `eval_perturbation.enabled` is true. Pass a YAML
mapping through the shared evaluator's `--overrides` (all policy adapters use
the same execution hook):

```sh
--eval_perturbation '{enabled: true, position_m: 0.05, force_n: 2.0, duration_s: 0.2, start_fraction: 0.2, end_fraction: 0.4}'
```

The selected task YAML still owns `enable_dynamics`; this option does not switch
the simulation mode. The horizontal world-frame direction and scheduled time
are determined by task name and episode seed, independently of policy RNG.
Time is measured in policy reference simulation steps, excluding expert seed
validation and any auxiliary gripper holds. The window is a fraction of
`step_lim * save_freq * timestep`.

- Dynamics: a single finite-duration external force pulse acts on the hand's
  center of mass in the translational dynamics integrator. It does not alter
  controller reference commands, inertia or the allocation matrix.
- Direct: one horizontal displacement of the hand. The displacement is retained
  for the remainder of the current open-loop action chunk so the next simulator
  tick cannot erase it. The next chunk starts from the actual displaced pose;
  no persistent actuator bias and no forced replan are introduced. Thus different
  native policy replan intervals remain part of the recovery experiment.
- No task object is directly repositioned by this feature. Objects already
  attached through the existing grasp implementation continue to follow it.

`perturbations.jsonl` records each episode's scheduled event, actual triggering,
direction, magnitude, force duration in ticks, and final success. Episodes that
end before the event must be reported separately, not counted as successful
recovery. End-of-episode task success is a recovery outcome, not a measurement
of settling time. When comparing methods, also match their actual episode seed
lists. Use a new output directory; do not mix perturbed and baseline results.

This evaluation uses no video. Deferred replay does not currently reconstruct
external disturbance events, so do not use unmodified action-only replay to
illustrate these perturbed trajectories.
