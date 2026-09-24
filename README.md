# Deepfake Workbench

A uniform, plugin-driven workbench for deepfake detection research: raw dataset → verified
inventory → face clips → trained detector → score file → evaluation report, with every stage
pluggable and every result traceable to a protocol version, a processing profile and a config
fingerprint.

> **Status:** pre-release (`0.1.0a1`). Linux is the only supported and tested OS. Python ≥ 3.12.

## Run from a clone

```bash
git clone https://github.com/dfwb-research/deepfake-workbench && cd deepfake-workbench
uv sync
uv run dfwb doctor
```

## Data policy

DFWB never distributes media. Users obtain datasets from their owners; DFWB works with identifiers,
labels and splits only.

## Citing

If DFWB helps your work, please consider citing it (see `CITATION.cff`).

## Licence

MIT. See `LICENSE`.
