# Restored OpenVLA-OFT model sources

The repository copy was missing the entire `prismatic/models/` directory even
though its existing package entry point and training scripts import it.

All 32 files under `prismatic/models/` were restored without modification from:

- Repository: https://github.com/moojink/openvla-oft
- Commit: `e4287e94541f459edc4feabc4e181f537cd569a8`
- Tree: https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/models
- License: MIT, retained in the existing root `LICENSE`.

The restored files were compared byte-for-byte against that checkout. This adds
source code only, with no model weights or training data. Flying-Hand dimensions
are supplied through the existing `prismatic.vla.constants` imports and the
existing explicit `action_dim`/`proprio_dim` constructor arguments.

Compilation and a small CPU action-head/projector check do not establish full
backbone or checkpoint compatibility; that still requires the project's training
dependencies and pretrained weights.
