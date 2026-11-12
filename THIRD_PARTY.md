# Dependencies and provenance

- Newton: https://github.com/newton-physics/newton — 1.7.0; `vendor/newton_twoway.py` retains its Apache-2.0 header.
- Warp: https://github.com/NVIDIA/warp — 1.17.0+cu12 in the experiment environment.
- BS-Cloth: https://simulation-intelligence.github.io/BS-Cloth/ — upstream checkout commit `9ae0eddbea82fbf8a8be5405f0d9623deb871714`; not vendored here, adapter patch included. Obtain source using the author project link and follow upstream licensing/build requirements.
- SPlisHSPlasH: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH — Python wheel 2.19.1.
- splashsurf: https://github.com/InteractiveComputerGraphics/splashsurf — 0.14.1.0.
- Blender: https://www.blender.org/ — Mantaflow baking used 3.0.1; final rendering used 4.5.3 LTS.
- MiniMax H3: https://huggingface.co/MiniMaxAI/MiniMax-H3 — locally supplied Ref2VA weights; consult model license. No model weights redistributed.
- UnitBox.obj is a simple unit-cube boundary mesh supplied with this harness.

Exact package availability and GPU compatibility must be checked on the target machine. Upstream dependency code is not covered by any implied repository-wide license.
