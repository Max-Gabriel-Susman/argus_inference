# Argus Inference

This repository contains the inference package for the Argus perception pipeline

## Development 

Run tests locally by navigating to the workspace directory for this package:
```
colcon test --packages-select argus_inference
```

Then output the results like so:
```
colcon test-result --verbose
```

Run proof of concept neural decoding script like so: 
```
ARGUS_DATASET_PATH=$HOME/Documents/datasets/indy_loco/indy_20161005_06.mat \
  python3 scripts/poc_decode_cmdvel.py
```

## Datasets

Two files, with distinct roles: one is the **training source**, the other is a
**replay artifact** derived from it. They are not interchangeable.

### `data/indy_20161005_06.mat` — training source

Session `indy_20161005_06` from *Nonhuman Primate Reaching with Multichannel
Sensorimotor Cortex Electrophysiology* (O'Doherty, Cardoso, Makin & Sabes,
UCSF). A macaque made self-paced reaches to targets on a grid while a
96-channel Utah array recorded M1. There are no trial boundaries — reaches are
continuous, with no inter-trial gaps or pre-movement delays.

Contents (MATLAB v7.3, i.e. HDF5, read here with `h5py`):

| Variable      | Shape   | Meaning                                     |
| ------------- | ------- | ------------------------------------------- |
| `t`           | k × 1   | Timestamps, seconds                         |
| `cursor_pos`  | k × 2   | Cursor position (x, y), mm, 250 Hz          |
| `target_pos`  | k × 2   | Target position (x, y), mm, 250 Hz          |
| `finger_pos`  | k × 3/6 | Fingertip position (z, -x, -y), cm          |
| `spikes`      | n × u   | Spike time vectors per channel per unit     |
| `wf`          | n × u   | Spike waveform snippets, µV                 |

**Role:** `inference_node.py` loads this at startup, bins spike times, derives
4-way intent labels from the cursor-to-target vector, and fits an
LDA-over-StandardScaler pipeline. This file never leaves the host — it is
training data, not pipeline input.

Source: doi:10.5281/zenodo.583331 (84.0 MB), CC-BY-4.0.

### `data/neural_96.csv` — replay artifact

Binned spike counts derived from the session above: 96 channels, 50 ms bins
(20 Hz), values typically 0–5. Columns are `sample,t,ch0..ch95`, with `t` in
session-relative seconds.

**Role:** stands in for live acquisition so the transport and ROS layers can be
exercised without hardware. Two consumers:

- `argus_sensors/neural_telemetry_replay` reads it directly and publishes
  `NeuralFrame` on `/argus/neural_interface_bridge/neural_data`.
- `generate_replay_header.py` converts the first N rows into `replay_data.h`
  for the embedded side, asserting every value fits `uint8_t`.

This file trains nothing. It exists to make the pipeline observable end to end.

### What these files cannot do

Binned spike counts are not voltages. Neither file can drive a DAC into an
analog frontend, so neither is usable for validating the RHD2132 acquisition
path. That requires raw broadband from the same session
(doi:10.5281/zenodo.1419774, ~24.4 kS/s, NWB 1.0.6), which is tracked
separately.

Separately: the decoder trained here will not transfer to dissociated cultures,
which have no behavioral correlates to label. The acquisition and transport
infrastructure transfers; the model does not.