"""Regenerate the built-in toyfake protocol pack, ``src/dfwb/_packs/toyfake``, byte for byte.

Run it after any change to what the pack holds: the toyfake generator or builder, the pack
writer, or the dfwb version, which ``pack.yaml`` and ``PROVENANCE.json`` record::

    uv run python scripts/regenerate_toyfake_pack.py

It takes the same path as a user's machine: ``synth(write_media=False)`` writes the tree made by
``dfwb datasets synth toyfake`` with its defaults (200 videos, seed 0) into a temporary folder,
the inventory builder scans it, and ``build_dataset`` writes the dataset's files. Nothing written
depends on the machine: rows are sorted, and no file holds a timestamp or an absolute path. The
test suite regenerates the pack the same way and compares every file with the committed one.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from dfwb import __version__
from dfwb.core.errors import ConfigError, DFWBError
from dfwb.core.paths import dataset_overrides, resolve_roots
from dfwb.preprocess.inventory.runner import build_inventory
from dfwb.preprocess.packbuild import PACK_YAML, build_dataset
from dfwb.preprocess.toyfake import DEFAULT_SEED, DEFAULT_VIDEOS, synth

DATASET = "toyfake"
PACK = Path(__file__).resolve().parents[1] / "src" / "dfwb" / "_packs" / DATASET


def pack_card() -> str:
    """``pack.yaml``: the pack is named after its one dataset and versioned with dfwb."""
    card = {"schema_version": 1, "name": DATASET, "version": __version__, "datasets": [DATASET]}
    return yaml.safe_dump(card, sort_keys=False)


def regenerate(out: Path) -> None:
    """Write the toyfake pack (``pack.yaml`` and ``toyfake/``) into ``out``.

    Raises:
        ConfigError: ``out`` exists and is not empty, or a folder override for toyfake is set
            (the official split would then be read from that folder, not from the fresh tree).
    """
    if out.exists() and any(out.iterdir()):
        raise ConfigError(f"{out} is not empty", hint="regenerate into an empty folder")
    override = dataset_overrides().get(DATASET)
    if override is not None:
        raise ConfigError(
            f"a folder override for {DATASET} is set ({override[1]})",
            hint="unset it: the pack is built from a freshly generated tree only",
        )
    with tempfile.TemporaryDirectory(prefix="dfwb-toyfake-") as scratch:
        datasets = Path(scratch) / "datasets"
        tree = synth(datasets, videos=DEFAULT_VIDEOS, seed=DEFAULT_SEED, write_media=False)
        roots = resolve_roots(flags={"datasets": datasets, "work": Path(scratch) / "work"})
        inventory = build_inventory(DATASET, root=tree.root, roots=roots)
        out.mkdir(parents=True, exist_ok=True)
        build_dataset(DATASET, out=out / DATASET, inventory=inventory.path, roots=roots)
    (out / PACK_YAML).write_text(pack_card(), encoding="utf-8", newline="\n")


def main() -> int:
    """Replace the committed pack with a fresh regeneration."""
    try:
        with tempfile.TemporaryDirectory(prefix="dfwb-toyfake-pack-") as scratch:
            fresh = Path(scratch) / "pack"
            regenerate(fresh)
            shutil.rmtree(PACK, ignore_errors=True)
            shutil.copytree(fresh, PACK)
    except DFWBError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return exc.exit_code
    print(f"regenerated {PACK.relative_to(PACK.parents[3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
