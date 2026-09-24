"""Record contract C3 (dataset and protocol records), with fast IO."""

from dfwb.core.records._base import assert_no_absolute_paths
from dfwb.core.records.io import (
    read_jsonl,
    read_split_tsv,
    split_sha256,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.local import (
    BuilderRef,
    InventoryRecord,
    Probe,
    ProcessedRecord,
    ProcessingProfile,
    TrackStats,
)
from dfwb.core.records.protocol import (
    DatasetCard,
    LabelVocab,
    PackCard,
    SchemeCard,
    SplitRow,
    VideoRecord,
)

__all__ = [
    "BuilderRef",
    "DatasetCard",
    "InventoryRecord",
    "LabelVocab",
    "PackCard",
    "Probe",
    "ProcessedRecord",
    "ProcessingProfile",
    "SchemeCard",
    "SplitRow",
    "TrackStats",
    "VideoRecord",
    "assert_no_absolute_paths",
    "read_jsonl",
    "read_split_tsv",
    "split_sha256",
    "write_jsonl",
    "write_split_tsv",
]
