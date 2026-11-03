# Bundled EC66 runtime

The modules, scene configuration, robot meshes and particle snapshot in this directory are part of this repository. No source checkout, trajectory dataset or cached demonstration outside this repository is used at runtime.

`SOURCE_MANIFEST.json` records the original workspace-relative source names and SHA-256 hashes for provenance. The bundled copies have been adapted to local resource paths. The particle scene now loads the requested polynomial trajectory directly, and the GUI base and distribution helpers exclude unrelated planning and training entry points.

`my_scene.py` retains its original copyright notice: Copyright (c) 2025 The Newton Developers, SPDX-License-Identifier: Apache-2.0. The associated license text is included in `LICENSE-Newton.md`. Changes in this checkout include using the bundled scene configuration with demo-trajectory loading disabled. Newton and Warp are installed as Python dependencies and retain their own licenses.

Robot and shovel assets are copied from the existing EC66 workspace without changing their geometry. Their source hashes are recorded in the manifest; this notice does not assign a new license to those assets.
