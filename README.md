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
