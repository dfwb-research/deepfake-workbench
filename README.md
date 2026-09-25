# Deepfake Workbench

A uniform, plugin-driven workbench for deepfake detection research: raw dataset → verified
inventory → face clips → trained detector → score file → evaluation report, with every stage
pluggable and every result traceable to a protocol version, a processing profile and a config
fingerprint.

> **Status:** pre-release (`0.1.0b1`). Linux is the only supported and tested OS. Python ≥ 3.12.

## Run from a clone

```bash
git clone https://github.com/dfwb-research/deepfake-workbench && cd deepfake-workbench
uv sync
uv run dfwb doctor
```

## Quickstart

dfwb ships a synthetic dataset, toyfake, generated entirely from a seed, so the whole
raw-dataset → inventory → protocol → training pipeline can be tried with nothing to download:

```bash
export DFWB_DATASETS_ROOT=~/datasets
export DFWB_WORK_ROOT=~/dfwb-work

uv run dfwb datasets synth toyfake --out "$DFWB_DATASETS_ROOT"
uv run dfwb inventory build toyfake
uv run dfwb protocols verify toyfake

uv run dfwb preprocess run toyfake --profile toy-64-center-8f
uv run dfwb preprocess status toyfake --profile toy-64-center-8f

cat > toy-cpu.yaml <<'YAML'
schema: dfwb.train/1
extends: [dfwb://templates/toy-cpu.yaml]
YAML
uv run --extra train dfwb train -c toy-cpu.yaml --device cpu

uv run dfwb runs list
uv run dfwb runs show toy-cpu
```

`datasets synth` writes a small set of synthetic real and blended-fake videos to
`$DFWB_DATASETS_ROOT/toyfake`. `inventory build` scans that folder and writes
`$DFWB_WORK_ROOT/toyfake/inventory.jsonl`. `protocols verify` joins that inventory against dfwb's
built-in toyfake protocol pack and writes a coverage report; with the defaults above it reports
full coverage. `preprocess run` crops and tracks a face through every video with the
`toy-64-center-8f` profile, which has no detector and no model to download (like every profile, it
needs the `preprocess` extra, which a clone's `uv sync` already installs), writing a lossless frame
store under `$DFWB_WORK_ROOT/toyfake/processed/`; `preprocess status` then counts it by outcome.
`train` needs the `train` extra (PyTorch, Lightning, timm) — `uv run --extra train` installs the
CPU build for this one invocation; `docs/install.md` covers installing a GPU build instead. The
`toy-cpu.yaml` file just extends the shipped `toy-cpu` template, which trains a tiny CNN on the
toyfake tree just built, for two epochs, writing its run under `./runs/toy-cpu/`; `runs list` shows
it, and `runs show` its full detail: data sources, validation metrics, and the config fingerprint
that identifies the experiment.

See `docs/concepts/protocols.md` for what `verify` checks and its exit codes,
`docs/concepts/processing-profiles.md` for the shipped face-processing profiles and the processed
store's layout, `docs/concepts/detectors.md` for the detector contract a trained run implements and
how to load one back, `docs/install.md` for installing the face-detection backends and a PyTorch
build, and `docs/guides/add-a-dataset.md` / `docs/guides/write-a-plugin.md` for wiring up a real
dataset or a model component of your own.

## Data policy

DFWB never distributes media. Users obtain datasets from their owners; DFWB works with identifiers,
labels and splits only.

## Data in several places, several machines

Datasets are often spread across more than one storage location, and that layout usually differs
from machine to machine. `DFWB_DATASETS_ROOT` accepts a `:`-separated list of roots, searched in
order; `dfwb doctor` lists every root it searched and where each dataset was found.

Per-machine settings belong in a `.env` file next to where you run `dfwb` (copy `.env.example`
and edit it; `.env` is git-ignored and never committed). `dfwb` loads it explicitly, before any
subcommand runs, never at import; a value already set in the real environment always wins:

```bash
# .env
DFWB_DATASETS_ROOT=/data/datasets:/nfs/datasets
DFWB_WORK_ROOT=/data/dfwb-work
```

`--env-file PATH` loads a specific file instead, and `--no-env-file` skips loading one entirely.

For settings that should be checked in (shared defaults plus per-host overrides), use
`dfwb.toml`'s `[hosts.<name>]` tables, keyed by short hostname (or `DFWB_HOST`):

```toml
# dfwb.toml
[roots]
datasets = "/shared/datasets"

[hosts.gpu-node-1.roots]
datasets = ["/fast/datasets", "/nfs/datasets"]

[hosts.gpu-node-1.datasets]
kodf = "/fast/KoDF-mirror"
```

## Citing

If DFWB helps your work, please consider citing it (see `CITATION.cff`).

## Licence

MIT. See `LICENSE`.
