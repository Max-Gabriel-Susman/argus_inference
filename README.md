# Argus Inference

This repository contains the inference package for the Argus perception pipeline

The whole stack, and the one command that runs it on the board, is described in
[argus_bringup/README.md](https://github.com/Max-Gabriel-Susman/argus_bringup/blob/main/README.md).
Build and test this package from `~/Documents/argus_ws` with
`colcon build --packages-select argus_inference && colcon test --packages-select argus_inference`.

`inference_node` subscribes to `NeuralFrame` on `/argus/sensors/neural_telemetry`,
predicts one of four reach intents with StandardScaler → LDA, and publishes
`/cmd_vel`. It has two model paths and logs which one is active (`model path: ...`):

- **Saved model (the demo).** `ARGUS_MODEL_PATH` points at a pipeline from
  `argus_sim/tools/decode_test.py --save-model`. With counts plus power at
  3.5σ that model scores 53.5 % in 5-fold CV. The node builds
  `[counts..., power...]` from each frame. The launch sets the variable from
  `model:=`. On 2026-09-28 it decoded the fabric's features live, and all
  four intents occurred over a 90 s board run.
- **Train at startup.** Without `ARGUS_MODEL_PATH`, the node trains on the
  `.mat` given by `ARGUS_DATASET_PATH`, on counts only (44.0 %).

## Development

Run tests locally by navigating to the workspace directory for this package:
```
colcon test --packages-select argus_inference
```

Then output the results like so:
```
colcon test-result --verbose
```

Run the inference node with a saved model, or train it (and the
proof-of-concept decoding script) on the training dataset (see Data below
for where it lives):
```
ARGUS_MODEL_PATH=$HOME/argus_model.pkl ros2 run argus_inference inference_node

ARGUS_DATASET_PATH=$HOME/argus_data/indy_20161005_06.mat \
  ros2 run argus_inference inference_node

ARGUS_DATASET_PATH=$HOME/argus_data/indy_20161005_06.mat \
  python3 scripts/poc_decode_cmdvel.py
```

## Data

No large file lives in any Argus repository. Raw datasets go in `~/argus_data/`
and are referenced by path: `ARGUS_DATASET_PATH` for the decoder, pointing at
`indy_20161005_06.mat` (O'Doherty, Cardoso, Makin & Sabes, CC-BY-4.0,
doi:10.5281/zenodo.583331).

Provenance, download (`scripts/fetch.sh`), and the command that makes every
derived file (`scripts/derive.sh`) live in the
[argus_data](https://github.com/Max-Gabriel-Susman/argus_data) repository,
including the `.mat` layout this node reads and the shared session clock that
aligns it with the broadband recording.
