"""Record contracts C3 (dataset and protocol records) and C5 (score files), with fast IO."""

from dfwb.core.records._base import assert_no_absolute_paths
from dfwb.core.records.io import (
    iter_jsonl_dicts,
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
    to_video_record,
)
from dfwb.core.records.protocol import (
    DatasetCard,
    LabelVocab,
    PackCard,
    PackProvenance,
    PairRecord,
    SchemeCard,
    SplitRow,
    VideoRecord,
)
from dfwb.core.records.scores import (
    CalibrationInfo,
    ScoreFile,
    ScoreMeta,
    ScoreRow,
    canonical_where,
    read_scores,
    write_scores,
)

__all__ = [
    "BuilderRef",
    "CalibrationInfo",
    "DatasetCard",
    "InventoryRecord",
    "LabelVocab",
    "PackCard",
    "PackProvenance",
    "PairRecord",
    "Probe",
    "ProcessedRecord",
    "ProcessingProfile",
    "SchemeCard",
    "ScoreFile",
    "ScoreMeta",
    "ScoreRow",
    "SplitRow",
    "TrackStats",
    "VideoRecord",
    "assert_no_absolute_paths",
    "canonical_where",
    "iter_jsonl_dicts",
    "read_jsonl",
    "read_scores",
    "read_split_tsv",
    "split_sha256",
    "to_video_record",
    "write_jsonl",
    "write_scores",
    "write_split_tsv",
]
