"""WildDeepfake: real and fake face-crop sequences collected from the internet.

WildDeepfake has no videos. The release ships pre-cropped 224x224 face sequences, so every record
is a **frame directory**: one folder of PNG frames per sequence, and the record's ``relpath``
names that folder, not a file. Relative to the ``WildDeepfake`` folder (a single version, no
compression levels):

* reals: ``original_content/real/frames/224w_224h_wild_precropped/<key>/<frame>.png``;
* fakes: ``manipulated_content/fake/frames/224w_224h_wild_precropped/<key>/<frame>.png``.

The release packs the sequences as tar shards, one set per ``<label>_<split>`` category, each
shard holding ``<shard>/<label>/<sequence>/<frame>.png``. They are expected unpacked into one
folder per sequence, named ``<label>_<split>_<shard>_<sequence>`` (e.g. ``real_train_6_54``;
shard ids repeat across categories, so the category prefix keeps the names apart), with frame
names zero-padded to six digits so they sort in order.

A record is keyed by its folder's name. Its attributes are the ``split`` named in the key
(``train`` or ``test``, else None), the ``shard`` and ``sequence`` (the third and fourth parts,
when present) and ``n_frames``, the number of PNG frames in the folder. Identity, target, source
and ``pair_key`` are all None: a wild fake has no known source, and nothing pairs.

The official split is the one named in each key: WildDeepfake publishes train and test and no
val. The default scheme keeps that test and carves the train 80/20 into train and val by an md5
of each sequence's key, since no sequence carries an identity.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import (
    BaseBuilder,
    LabelSpec,
    SchemeSpec,
    TaskSpec,
    expand_compressions,
)
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key

__all__ = ["WildDeepfakeBuilder"]

# The folder, under each task's frames folder, that holds one sub-folder per sequence.
_SEQUENCE_GROUP: Final = "224w_224h_wild_precropped"
_FRAME_SUFFIX: Final = ".png"
# The splits a sequence's name can carry, as its second "_"-separated part.
_OFFICIAL_SPLITS: Final[dict[str, Split]] = {"train": "train", "test": "test"}


def _is_frame(path: Path) -> bool:
    """A frame is a visible ``.png`` file (the suffix matched ignoring case)."""
    return (
        not path.name.startswith(".")
        and path.suffix.lower() == _FRAME_SUFFIX
        and path.is_file()  # follows symlinks; a dangling link is not a frame
    )


def _sequence_dirs(folder: Path) -> list[Path]:
    """Every visible sub-folder of ``folder`` (symlinks followed), sorted; ``[]`` if missing."""
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if not path.name.startswith(".") and path.is_dir()
    )


def _holds_sequences(folder: Path) -> bool:
    """Whether ``folder`` has a sequence folder holding at least one frame; stops at the first."""
    return any(
        any(_is_frame(frame) for frame in sequence.iterdir()) for sequence in _sequence_dirs(folder)
    )


def _split_of(key: str) -> Split | None:
    """The split a ``<label>_<split>_<shard>_<sequence>`` name carries, if it is train or test."""
    parts = key.split("_")
    return _OFFICIAL_SPLITS.get(parts[1]) if len(parts) >= 2 else None


class WildDeepfakeBuilder(BaseBuilder):
    """WildDeepfake (``wilddeepfake``): face-crop sequences, one frame directory per record."""

    dataset_id = "wilddeepfake"
    expected_folder = "WildDeepfake"
    label_prefix = "WDF"
    tasks = (
        TaskSpec(
            "REAL", "real", "real", f"original_content/real/frames/{_SEQUENCE_GROUP}", "original"
        ),
        TaskSpec(
            "FAKE", "fake", "fake", f"manipulated_content/fake/frames/{_SEQUENCE_GROUP}", "fake"
        ),
    )
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        # The fakes were collected from the internet; how each was made is not recorded.
        "FAKE": LabelSpec(binary=1, binary_av=1, multiclass=2, family="unknown"),
    }
    schemes = {
        "official+ident-80-20": SchemeSpec(
            "official+ident-80-20",
            "derived",
            params={"policy": "official-train-test"},
            source="the train/test split named in each sequence (<label>_<split>_<shard>_"
            "<sequence>)",
            rationale="the publisher's split has train and test but no val: the official test "
            "is kept, and the official train is carved 80/20 into train and val by an md5 of "
            "each sequence's key, since the sequences carry no identity",
        ),
        "official": SchemeSpec(
            "official",
            "official",
            source="the train/test split named in each sequence (<label>_<split>_<shard>_"
            "<sequence>)",
            rationale="the publisher's split as released: train and test, no val",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every sequence is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: up to 100 fakes drawn at random from the "
            "official test, balanced with as many reals",
        ),
    }
    default_scheme = "official+ident-80-20"
    benchmark = BenchmarkSpec(k_fake=100)
    card_info = {
        "name": "WildDeepfake",
        "aliases": ["WDF"],
        "release": "7,314 face-crop sequences collected from the internet (3,805 real and 3,509 "
        "fake), released as 224x224 face crops rather than videos: 3,409 real and 3,099 fake "
        "train sequences, 396 real and 410 fake test sequences",
        "homepage": "https://huggingface.co/datasets/xingjunm/WildDeepfake",
        "license": {
            "spdx": None,
            "summary": "the WildDeepfake terms: research use only, after the authors approve an "
            "access request",
            "url": None,
        },
        "access": "request access from the authors (the download is gated); dfwb never "
        "distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "<task>/<label>_<split>_<shard>_<sequence>: the sequence folder's name, "
        "e.g. REAL/real_train_6_54",
    }
    layout_notes = (
        "Records are frame directories, not videos: one folder of 224x224 PNG face crops per "
        "sequence, and a record's relpath names the folder.\n"
        "The release's tar shards (<shard>/<label>/<sequence>/<frame>.png) are expected unpacked "
        "into one folder per sequence, named <label>_<split>_<shard>_<sequence>, with frame names "
        "zero-padded to six digits.\n"
        "The official split is the <split> in each name: train or test."
    )

    # ----------------------------------------------------------------------------- layout

    def layout_dirs(self) -> tuple[str, ...]:
        """Each task's sequence folder: where the frame directories are looked for."""
        return tuple(task.video_dir for task in self.tasks)

    def videos_present(self, folder: Path) -> tuple[str, ...]:
        """Which of :meth:`layout_dirs` hold a frame directory with at least one frame."""
        return tuple(dir_ for dir_ in self.layout_dirs() if _holds_sequences(folder / dir_))

    def layout_present(self, folder: Path) -> bool:
        """Whether a task's sequence folder under ``folder`` holds a frame directory of PNGs.

        The raw layout is the sequence folders themselves: a copy with only the folder names, or
        with only empty sequence folders, does not count.
        """
        return any(_holds_sequences(folder / dir_) for dir_ in self.layout_dirs())

    def video_copy_for(
        self, copies: Sequence[Path], task: TaskSpec, compression: str | None
    ) -> Path | None:
        """The first of ``copies`` whose sequence folder for ``task`` holds a frame directory."""
        for copy in copies:
            if _holds_sequences(copy / task.video_dir):
                return copy
        return None

    def copies_after(
        self, copies: Sequence[Path], chosen: Path, task: TaskSpec, compression: str | None
    ) -> list[Path]:
        """Every copy after ``chosen`` that also holds a frame directory for ``task``."""
        index = list(copies).index(chosen)
        return [copy for copy in copies[index + 1 :] if _holds_sequences(copy / task.video_dir)]

    # ----------------------------------------------------------------------------- discovery

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        """Yield a record for every sequence folder of every task.

        Across bound copies, each task is scanned in the first copy holding a frame directory
        for it (see :meth:`video_copy_for`); unbound, just ``root``. Every visible sub-folder is a
        record, even one without frames.

        Raises:
            ConfigError: ``compressions`` names a compression (this dataset has none).
        """
        copies = self._copies if self._copies is not None else (root,)
        chosen = self._chosen_copies
        for task in self.tasks:
            for compression in expand_compressions(
                task.video_dir, self.known_compressions, compressions
            ):
                key = (task.abbr, compression)
                copy = (
                    chosen[key]
                    if chosen is not None and key in chosen
                    else self.video_copy_for(copies, task, compression)
                )
                if copy is None:
                    continue
                for sequence in _sequence_dirs(copy / task.video_dir):
                    yield self._sequence_record(task, sequence, compression)

    def _sequence_record(
        self, task: TaskSpec, sequence: Path, compression: str | None
    ) -> InventoryRecord:
        key = sequence.name
        parts = key.split("_")
        return self.record(
            task,
            key,
            f"{task.video_dir}/{key}",
            compression,
            attrs={
                "split": _split_of(key),
                "shard": parts[2] if len(parts) >= 3 else None,
                "sequence": parts[3] if len(parts) >= 4 else None,
                "n_frames": sum(1 for path in sequence.iterdir() if _is_frame(path)),
            },
        )

    # ----------------------------------------------------------------------------- hooks

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split, named in its key: ``<label>_<split>_...``, train or test."""
        assignment: dict[str, Split] = {}
        for record in records:
            split = _split_of(local_key(record.key))
            if split is not None:
                assignment[record.key] = split
        return assignment

    def describe_layout(self) -> str:
        """The expected folder layout: one frame directory per sequence, per task."""
        lines = [
            f"{self.card_info['name']} ({self.dataset_id}): folder {self.expected_folder!r} "
            "under a datasets root.",
            "Frame directories, one per face-crop sequence, relative to that folder:",
        ]
        width = max(len(task.abbr) for task in self.tasks)
        for task in self.tasks:
            lines.append(
                f"  {task.abbr.ljust(width)}  {task.kind:<4}  "
                f"{task.video_dir}/<sequence>/<frame>{_FRAME_SUFFIX}  ({task.name})"
            )
        lines.append(f"Keys: {self.card_info['key_rule']}")
        lines.append(self.layout_notes)
        return "\n".join(lines)
