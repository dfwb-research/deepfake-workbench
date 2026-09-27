#!/usr/bin/env bash
# One-command setup for a fresh clone of deepfake-workbench: installs the extras you need with
# uv, sets up .env and the in-clone data layout, and runs `dfwb doctor` to confirm it all works.
#
# Usage: scripts/setup.sh [--cpu] [--gpu] [--preprocess] [--train] [--dry-run] [-h|--help]
#
# Flags choose the extras `uv sync --locked` installs (they add up; give more than one to
# combine them). With no flag at all, this is the same as --cpu.
#
#   --cpu         The extras docs/quickstart.md's own recipe uses to run the toyfake benchmark
#                 on CPU: `train` and `preprocess`. The default.
#   --gpu         Add the `train` extra, then print the CUDA build steps from docs/install.md
#                 instead of running them: which CUDA build to install is specific to your
#                 driver, so it is never chosen for you.
#   --preprocess  Add the `preprocess` extra (media probing and the face pipeline).
#   --train       Add the `train` extra (the training stack: torch, lightning, timm, ...).
#   --dry-run     Print every command instead of running `uv sync` and `dfwb doctor` (the two
#                 steps that need a network connection and an installed environment). `.env` and
#                 the data directories are still set up for real: both are local, offline, and
#                 exactly what the idempotence and never-overwrite behaviour below need proving
#                 against, so a dry run still checks them.
#
# Idempotent: running this more than once is safe. An existing `.env` is never overwritten.

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
}

dry_run=0
want_preprocess=0
want_train=0
want_gpu=0
flag_given=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu)
      want_preprocess=1
      want_train=1
      flag_given=1
      ;;
    --gpu)
      want_train=1
      want_gpu=1
      flag_given=1
      ;;
    --preprocess)
      want_preprocess=1
      flag_given=1
      ;;
    --train)
      want_train=1
      flag_given=1
      ;;
    --dry-run)
      dry_run=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ "$flag_given" -eq 0 ]; then
  want_preprocess=1
  want_train=1
fi

# Needs the network or an installed environment: printed always, run only when not --dry-run.
run_network() {
  echo "+ $*"
  if [ "$dry_run" -eq 0 ]; then
    "$@"
  fi
}

# Local and offline: always actually done, even under --dry-run (see the --dry-run flag above).
run_local() {
  echo "+ $*"
  "$@"
}

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not installed" >&2
  echo "hint: install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

extra_args=()
[ "$want_train" -eq 1 ] && extra_args+=(--extra train)
[ "$want_preprocess" -eq 1 ] && extra_args+=(--extra preprocess)

run_network uv sync --locked "${extra_args[@]}"

if [ "$want_gpu" -eq 1 ]; then
  cat <<'RECIPE'

To train on a GPU, reinstall a matching CUDA build of torch over the CPU build `uv sync` just
installed, then run every later command with `--no-sync` so a later `uv sync` never puts the CPU
build back (docs/install.md, "Training: a CUDA build, with uv"):

  # Pick the index for your driver from the official selector at
  # https://pytorch.org/get-started/locally/ -- cu121 here is an example.
  uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121

  uv run --no-sync dfwb train -c your-config.yaml --device cuda:0

RECIPE
fi

if [ -f .env ]; then
  echo ".env already exists, leaving it alone"
else
  run_local cp .env.example .env
fi

for dir in data/datasets data/work data/runs data/cache; do
  run_local mkdir -p "$dir"
done

run_network uv run dfwb doctor
