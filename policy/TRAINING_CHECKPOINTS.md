# Inference checkpoints and latest-only training state

This describes the Flying-Hand training entrypoints. It does not migrate or
delete historical checkpoints. Inference exports retain the policy-specific
model format and normalization assets; they do not contain optimizer state.

| Policy | Original checkpoint cadence (unchanged) | State cadence | State location |
| --- | --- | --- | --- |
| pi0 / pi05 | `save_interval` steps; `keep_period` retains periodic inference exports plus latest | Same save event | `<run>/state/<step>/` (Orbax, latest only) |
| FastWAM | `save_every` steps | `state_save_every=100` steps | `<run>/checkpoints/state/step_XXXXXX/` |
| VLA-JEPA | `trainer.save_interval` steps | Same save event | `<run>/checkpoints/state/step_XXXXXX/` |
| OpenVLA-OFT | `save_freq` steps | Same save event | `<run>/state/latest.pt` |
| DP | `training.checkpoint_every` epochs, plus metric Top-K | Same save event | `<run>/checkpoints/latest.ckpt` |
| ACT | `save_freq` epochs, plus final/best model | Same save event | `<run>/state/latest.pt` |

Zero disables periodic saves for the corresponding cadence. Normal completion
still saves final model/state. DP/ACT save at epoch boundaries; their intervals
are **not optimizer steps**. Latest-only state includes model parameters so it
is self-contained. No additional state cadence is introduced for these six
methods. FastWAM retains its previously explicitly requested independent
`state_save_every=100` setting.

## Resume interfaces

- pi0/pi05 wrapper: `--resume-run EXACT_EXISTING_RUN_NAME`. Use the original
  config name, dataset arguments, assets, batch size and device mesh. Resume
  prefers `<run>/state`; legacy checkpoints containing `train_state` remain
  readable. A directory containing only inference weights is rejected as a
  training resume source. `--save-interval N` controls the original save cadence.
- DP wrapper: `--resume-state /absolute/path/checkpoints/latest.ckpt`.
  `training.checkpoint_every` controls epochs between saves. Model/EMA,
  optimizer, scheduler, EMA counter, next epoch, batch counter, sampler RNG,
  rank RNG and any pending gradients are saved. Historical full checkpoints
  remain readable, but lack some of the new resume metadata.
- ACT wrapper: `--resume-state /absolute/path/state/latest.pt`.
  State includes model, optimizer, normalization stats,
  next epoch, best-validation weights/score and rank RNG. Old model-only files
  require `--init-checkpoint` / `--init-stats` and are a warm start, not resume.
- OpenVLA-OFT wrapper: append `--resume_state /absolute/path/state/latest.pt`.
  Continue to provide the original base `VLA_PATH` and architecture options;
  the full state then restores the trained VLA, auxiliary modules, optimizer,
  scheduler, RNG and completed step. Dataset normalization is checked against
  the saved statistics. Do not combine with the legacy `--resume True` mode.
  Old complete model+`training_state.pt` checkpoints still use the legacy mode.
- VLA-JEPA wrapper: `RESUME_FROM=/absolute/path/checkpoints/state/step_XXXXXX`.
  See `vla-jepa/scripts/README_flying_hand.md` for its full-state contract.

Training step/epoch limits are **total targets**, not additional training length.
Use the original model, data, batch/accumulation settings and process count.
Weights-only files cannot reconstruct missing optimizer moments. Only load
trusted local state files: PyTorch optimizer/RNG state uses pickle serialization.

## Boundaries

- State publication uses temporary files/directories (or Orbax asynchronous
  checkpoint finalization). Retention applies after successful state writing.
- Fresh/resumed runs keep W&B's existing online default. Unless explicitly
  resuming the same run identity, W&B may create a new run with continued step
  numbers; this is distinct from optimizer-state recovery.
- The data loader and worker processes are rebuilt. These implementations do
  not promise bitwise-identical sample/augmentation replay across interruption.
  DP additionally restores its batch-sampler RNG; ACT resumes at the next epoch.
- Existing evaluation paths remain inference-compatible. Old files are not
  retroactively stripped of training state by this code change.
- CPU roundtrip tests cover retention and loading; production eight-GPU
  DeepSpeed/FSDP resume needs a separate live smoke test before relying on it.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest tests.test_training_state_retention
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=policy/pi0/src policy/pi0/.venv/bin/python tests/test_pi_split_checkpoints.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=policy/pi05/src policy/pi05/.venv/bin/python tests/test_pi_split_checkpoints.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=policy/vla-jepa policy/vla-jepa/.venv/bin/python -m unittest discover -s policy/vla-jepa/tests -p test_resume.py
```
