# Adding a dataset

Two independent pieces support each dataset dfwb understands:

1. an **inventory builder** — code that knows one dataset's raw folder layout, and turns it into
   a local inventory (a list of videos, their keys, labels and identities);
2. optionally, a **protocol pack** — data, published separately, describing exactly how that
   dataset is split into train/validation/test.

Both are plugins: they register themselves through the same `dfwb.plugins` entry point every
other plugin uses, so they never need to live inside dfwb itself. This guide writes both for a
small example dataset.

## 1. Write an inventory builder

Subclass `dfwb.preprocess.inventory.base.BaseBuilder`. It is table-driven: most of a builder is
class attributes, and the default discovery logic scans the folders the table describes.

```python
from pathlib import Path

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec


class MyDatasetBuilder(BaseBuilder):
    dataset_id = "my-dataset"  # the registry key
    expected_folder = "MyDataset"  # the folder name under a datasets root
    label_prefix = "MD"  # label keys become "MD-REAL", "MD-FAKE", ...

    tasks = (
        TaskSpec("REAL", "original", "real", "videos/real", "original"),
        TaskSpec("FAKE", "faceswap", "fake", "videos/fake", "faceswap"),
    )
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=0, family="real"),
        "FAKE": LabelSpec(binary=1, binary_av=1, multiclass=1, family="face-swap"),
    }
    schemes = {
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="no official split is published; every video counts as test",
        ),
    }
    default_scheme = "all-test"
    card_info = {
        "name": "My Dataset",
        "aliases": [],
        "release": "v1",
        "homepage": "https://example.org/my-dataset",
        "license": {"spdx": "CC-BY-4.0", "summary": "Free for research use.", "url": None},
        "access": "request access from the authors",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "the file stem",
    }
```

Each `TaskSpec` is one real source or one fake generation method: its abbreviation (the key
prefix, e.g. `"FAKE"`), a readable name, whether it is `"real"` or `"fake"`, where its videos sit
relative to the dataset folder, and the method name recorded on each of its records. A video
directory containing `{cX}` (e.g. `"videos/{cX}"`) is a placeholder for a compression level,
expanded over `known_compressions`.

The default `record_for_video` keys each video by its file stem and sets nothing else. Most
datasets need more — an identity parsed from the file name, or a pairing between a fake and the
real it was generated from — so override it:

```python
def record_for_video(self, task, path, relpath, compression):
    # "id003_id007.mp4" -> target id003 (a real identity), source id007
    if task.kind == "fake":
        target, source = path.stem.split("_")
        return self.record(
            task,
            path.stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            pair_key=target,
        )
    return self.record(task, path.stem, relpath, compression, identity=path.stem)
```

`self.record(...)` builds the `InventoryRecord`: it fills in the key (`<task abbr>/<local id>`),
the label key, and the builder/version stamp, and validates that the local id is non-empty and
that `relpath` is a relative path.

### Reading a metadata file

A dataset that ships identities, labels or an official split in a separate CSV or JSON file reads
it once, in `prepare`, and keeps what it needs on `self` for `discover` to use:

```python
class MyDatasetBuilder(BaseBuilder):
    metadata_files = ("identities.csv",)
    ...

    def prepare(self, root: Path) -> None:
        super().prepare(root)
        self._identity_of = _read_identities_csv(root / "identities.csv")
```

Declaring `metadata_files` matters when a dataset's folder shows up under more than one datasets
root (see below): it tells the runner which copy to read metadata from — the first copy that
holds every listed file — instead of an arbitrary one. Always call `super().prepare(root)` first;
it is what keeps `self.copies` correct when a builder is used directly, outside the runner.

### The other layout hooks

- **`official_splits(self, root, records)`** — read the publisher's own train/validation/test
  assignment from files under `root` (the metadata root), and return a mapping from record key to
  split. The default raises, meaning "this dataset has no official split"; only implement it when
  one exists, and only use an `official` scheme when you do.
- **`pair_candidates(self, fake)`** — given a fake `InventoryRecord`, return the real local id (or
  ids) it could pair with. The default returns `None` (no pairing). Used by pack building to
  record which real each fake was generated from.
- **`describe_layout(self)`** — human-readable text for `dfwb datasets info`. The default builds
  it from the task table; override it only if the layout is not "one folder per task".

### Datasets spread across more than one datasets root

`DFWB_DATASETS_ROOT` can list more than one location, and the same dataset's folder can exist
under more than one of them — for instance, one root holding a fresh `c23` copy and another still
holding an older `c40` copy left from a previous run. `BaseBuilder.discover` merges these
automatically: for each task and compression, it uses whichever root's copy holds a video there
first, in root order. Nothing needs to be written for this to work — it falls out of the default
`discover` once the runner has found every copy of the dataset's folder. `metadata_files` (above)
decides which one copy backs `prepare` and `official_splits`, independently of which copies back
which task's videos.

### Registering it

A builder does nothing until it is registered, through a `register(api)` function that a
`dfwb.plugins` entry point in your package's `pyproject.toml` points at:

```python
def register(api):
    api.inventory_builders.add(
        "my-dataset",
        target="my_package.builder:MyDatasetBuilder",
        summary="My Dataset",
        folder="MyDataset",
    )
```

```toml
# pyproject.toml
[project.entry-points."dfwb.plugins"]
my-package = "my_package:register"
```

Once installed in the same environment as dfwb, `dfwb datasets list` shows it, and:

```bash
dfwb inventory build my-dataset --root /path/to/raw/MyDataset
dfwb inventory show my-dataset
```

## 2. Publish a protocol pack

An inventory builder alone is enough to build a local inventory, but training and evaluation
compare against a *published* split, not just whatever is on your own disk. `dfwb protocols
new-pack` scaffolds a small, dependency-free package for that:

```bash
dfwb protocols new-pack my-dataset-protocols --name my-dataset-protocols --author "Your Name"
```

This writes `pyproject.toml` (registering itself through the same `dfwb.plugins` entry point,
with no dependencies of its own), an empty `pack.yaml`, and `README.md`/`NOTICE.md`/`LICENSE`
text to fill in.

With the inventory builder installed and a local inventory already built, generate the dataset's
protocol files from it:

```bash
dfwb protocols build my-dataset \
  --out my-dataset-protocols/src/my_dataset_protocols/packs/my-dataset \
  --update-pack-yaml
```

This writes every scheme's split file, the pairs, the labels, the dataset card, `NOTICE.md` and
`PROVENANCE.json` under `--out`, and lists `my-dataset` in the pack's `pack.yaml`. Building again
from the same inventory always produces the same bytes, so the result is safe to commit and diff.
Check it before committing:

```bash
dfwb protocols lint my-dataset-protocols/src/my_dataset_protocols/packs
```

Once the pack is installed in the same environment as dfwb (e.g. `uv pip install -e
my-dataset-protocols`), `dfwb protocols list` shows its schemes, and `dfwb protocols verify
my-dataset` checks any local inventory against it.

## See also

- `docs/concepts/protocols.md` — protocol references, split schemes, verify buckets and exit
  codes.
