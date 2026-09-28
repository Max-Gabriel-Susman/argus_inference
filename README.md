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
and are referenced by path: `ARGUS_DATASET_PATH` for the decoder, pointing at
`indy_20161005_06.mat` (O'Doherty, Cardoso, Makin & Sabes, CC-BY-4.0,
doi:10.5281/zenodo.583331).

Provenance, download (`scripts/fetch.sh`), and the command that makes every
derived file (`scripts/derive.sh`) live in the
[argus_data](https://github.com/Max-Gabriel-Susman/argus_data) repository,
including the `.mat` layout this node reads and the shared session clock that
aligns it with the broadband recording.
