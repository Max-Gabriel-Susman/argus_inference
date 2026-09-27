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

Run the inference node, or the proof-of-concept decoding script, against the
training dataset (see Data below for where it lives):
```
ARGUS_DATASET_PATH=$HOME/argus_data/indy_20161005_06.mat \
  ros2 run argus_inference inference_node

ARGUS_DATASET_PATH=$HOME/argus_data/indy_20161005_06.mat \
  python3 scripts/poc_decode_cmdvel.py
```

## Data

No large file lives in any Argus repository. Raw datasets go in `~/argus_data/`
and are referenced by path — `ARGUS_DATASET_PATH` for the decoder,
`dataset_path` for the relay. A fresh clone cannot contain them anyway, so the
honest arrangement is to say where they come from and where they go.

| File | Lives in | What it is | Consumed by |
| --- | --- | --- | --- |
| `indy_20161005_06.mat` | `~/argus_data/` | spike times + behaviour, 84 MB | `inference_node` (training) |
| `indy_20161005_06_broadband.nwb` | `~/argus_data/` | raw voltage, 24.4 kS/s, ~1.9 GB | `argus_sim/tools/nwb_to_replay.py` |
| `indy_20161005_06_*.bin` | `~/argus_data/` | RHD2132 codes, 30 kS/s, derived | `argus_sim dataset_relay_node` |
| `neural_96.csv` | `argus_sensors/data/` | binned spike counts, 20 Hz, derived | `argus_sensors neural_telemetry_replay` |

Only the CSV is small enough to belong in a repository, and it belongs in
`argus_sensors`, which installs it and reads it.

### Getting the raw files

Both are session `indy_20161005_06` from *Nonhuman Primate Reaching with
Multichannel Sensorimotor Cortex Electrophysiology* (O'Doherty, Cardoso, Makin
& Sabes, UCSF), CC-BY-4.0. A macaque made self-paced reaches to targets on a
grid while a 96-channel Utah array recorded M1. There are no trial boundaries —
reaches are continuous, with no inter-trial gaps or pre-movement delays.

```
mkdir -p ~/argus_data
wget -O ~/argus_data/indy_20161005_06.mat \
  "https://zenodo.org/records/583331/files/indy_20161005_06.mat?download=1"
```

The broadband supplement is a separate record, doi:10.5281/zenodo.1419774;
download from the record page into the same directory as
`indy_20161005_06_broadband.nwb`.

The two files share one session clock: the broadband starts at `t = 1278 s`,
the behavioural data at `1288 s`. Anything derived from one can be aligned to
the other by timestamp alone.

### `indy_20161005_06.mat` — training source

MATLAB v7.3, i.e. HDF5, read here with `h5py`:

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

Source: doi:10.5281/zenodo.583331.

### `indy_20161005_06_broadband.nwb` — acquisition-path source

NWB 1.0.6 (HDF5). `/acquisition/timeseries/broadband/data` is 9,619,237 × 96
`int16` codes with a `conversion` attribute of 3.05185e-07 V/code;
`/acquisition/timeseries/broadband/timestamps` gives per-sample seconds. 394 s
at 24,414 Hz, unfiltered below the 7.5 kHz anti-alias low-pass, so each
channel carries its electrode's DC offset.

**Role:** the only file here that is actual voltage, and therefore the only
one that can stand in for an electrode array. It is never read at runtime;
`argus_sim/tools/nwb_to_replay.py` converts a segment of it into what the
relay serves.

Source: doi:10.5281/zenodo.1419774.

### `indy_20161005_06_*.bin` — converted replay segments

What `dataset_relay_node` mmaps: headerless little-endian `uint16`,
sample-major, 96 columns, one row per sample. Values are RHD2132 ADC codes —
offset binary, `0x8000` = 0 V at the electrode, 0.195 µV per LSB — so the
simulated Intan chips in the codec return exactly what real silicon would
have returned for that electrode voltage.

The converter AC-couples each channel (mean removal, then a first-order
high-pass, standing in for the chip's analog coupling) and resamples from
24,414 Hz to the fabric's 30,012 Hz sweep rate so spike widths and filter
cutoffs are right in the fabric's time base. It reports per-channel RMS and
the clipped fraction; cortical broadband is 20–150 µV RMS, and a millivolt
reading means the conversion attribute was not what was assumed.

```
python3 ~/Documents/argus_ws/src/argus_sim/tools/nwb_to_replay.py \
  ~/argus_data/indy_20161005_06_broadband.nwb \
  --start 120 --seconds 10 \
  --out ~/argus_data/indy_20161005_06_s120_10s.bin

ros2 run argus_sim dataset_relay_node --ros-args \
  -p dataset_path:=$HOME/argus_data/indy_20161005_06_s120_10s.bin
```

Ten seconds is 58 MB and holds hundreds of spikes. These are regenerable from
the NWB and are not kept anywhere but `~/argus_data/`.

### `neural_96.csv` — replay artifact

Lives in `argus_sensors/data/`, not here. Binned spike counts derived from the
`.mat`: 96 channels, 50 ms bins (20 Hz), values typically 0–5. Columns are
`sample,t,ch0..ch95`, with `t` in session-relative seconds.

**Role:** stands in for the decoded feature stream so the ROS layer can be
exercised with no hardware and no relay. `argus_sensors/neural_telemetry_replay`
reads it and publishes `NeuralFrame` on
`/argus/neural_interface_bridge/neural_data`. It trains nothing.

### Roles, and what each file cannot do

The decoder trains on the `.mat` and nothing else. The hardware path is
validated with the `.nwb`, via the `.bin`, and nothing else. They meet only
through the shared session clock.

- **The `.mat` cannot drive the hardware path.** Its `wf` field holds real
  waveforms at real amplitudes, but only as snippets around detected threshold
  crossings; the continuous record between spikes, the noise floor and the
  LFP, is discarded. It is a reference for what a spike detector should find,
  not a signal to feed one.
- **The `.csv` cannot either.** Counts are not voltages; there is no waveform
  to reproduce.
- **The `.bin` cannot train the decoder.** It carries no labels. Labels come
  from `cursor_pos` and `target_pos` in the `.mat`, and can be attached to any
  segment of broadband by timestamp — but that alignment is done on the host,
  not in the replay file.
- **None of this transfers to dissociated cultures**, which have no behavioural
  correlate to label. The acquisition and transport infrastructure transfers;
  the model does not.
