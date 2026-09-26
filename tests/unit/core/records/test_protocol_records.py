import gzip
import hashlib

import pytest
import yaml
from pydantic import ValidationError

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    PairRecord,
    SplitRow,
    VideoRecord,
    assert_no_absolute_paths,
    iter_jsonl_dicts,
    read_jsonl,
    read_split_tsv,
    records_sha256,
    split_sha256,
    write_jsonl,
    write_split_tsv,
)

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64
CARD = f"""\
id: ffpp
name: FaceForensics++
aliases: [FF++, FaceForensics++]
release: "v4 (2019)"
homepage: https://github.com/ondyari/FaceForensics
paper: {{title: "FaceForensics++", venue: ICCV, year: 2019, doi: "10.1109/ICCV.2019.00009"}}
license: {{spdx: null, summary: "Research-only; access by request form", url: null}}
access: "Request access via the upstream form; DFWB never distributes media."
distribution: undecided
terms: {{source: null, reviewed: null, notes: null}}
modalities: [video]
compressions: [raw, c23, c40]
key_rule: "real: <seq_id>; fake: <Method>/<tgt>_<src>"
schemes:
  official: {{kind: official, source: "FF++ official splits JSON", sha256: "{SHA_A}"}}
  ident-72-14-14:
    {{kind: derived, rule: md5-carve-key, ratios: [72, 14, 14], sha256: "{SHA_B}",
     rationale: "identity-disjoint"}}
  official+ident-80-20: {{kind: derived, rule: md5-carve-key, sha256: "{SHA_C}"}}
default_scheme: official
"""

LABELS = """\
vocab:
  FF-REAL:   {binary: 0, family: real,       modality: visual}
  FF-FS_DF:  {binary: 1, family: face-swap,  method: Deepfakes}
mappings:
  binary: {from: binary}
  visual-binary: {from: binary, override: {FF-FS_DF: exclude}}
"""


def test_dataset_card_from_contract_example():
    card = DatasetCard.model_validate(yaml.safe_load(CARD))
    assert card.schemes["ident-72-14-14"].ratios == [72, 14, 14]
    assert card.distribution == "undecided"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"default_scheme": "nope"}, "default_scheme 'nope' is not one of the schemes"),
        ({"id": "FF++"}, "String should match pattern"),
        ({"distribution": "maybe"}, "Input should be 'undecided', 'list' or 'recipe'"),
        ({"extra_field": 1}, "Extra inputs are not permitted"),
    ],
)
def test_dataset_card_rejects(change, message):
    data = yaml.safe_load(CARD) | change
    with pytest.raises(ValidationError, match=message):
        DatasetCard.model_validate(data)


def test_the_key_list_hashes_are_optional_and_checked():
    card = DatasetCard.model_validate(yaml.safe_load(CARD))
    assert (card.videos_sha256, card.pairs_sha256, card.pairing_rule) == (None, None, None)

    data = yaml.safe_load(CARD) | {
        "videos_sha256": SHA_A,
        "pairs_sha256": SHA_B,
        "pairing_rule": "target-id",
    }
    card = DatasetCard.model_validate(data)
    assert (card.videos_sha256, card.pairs_sha256, card.pairing_rule) == (SHA_A, SHA_B, "target-id")
    with pytest.raises(ValidationError, match="String should match pattern"):
        DatasetCard.model_validate(data | {"videos_sha256": "abc"})


def test_label_vocab_and_pack_card():
    vocab = LabelVocab.model_validate(yaml.safe_load(LABELS))
    assert vocab.mappings["visual-binary"].from_ == "binary"
    assert vocab.mappings["visual-binary"].override == {"FF-FS_DF": "exclude"}
    pack = PackCard.model_validate(
        {"schema_version": 1, "name": "dfwb-protocols", "version": "1.0.0", "datasets": ["ffpp"]}
    )
    assert pack.withheld == []
    with pytest.raises(ValidationError):
        PackCard.model_validate({"schema_version": 2, "name": "x", "version": "1", "datasets": []})


VIDEOS = [
    VideoRecord("000", "c23", "FF-REAL", "original", identity="000"),
    VideoRecord(
        "Deepfakes/000_003",
        "c23",
        "FF-FS_DF",
        "Deepfakes",
        identity="000",
        source_id="003",
        target_id="000",
        pair_key="000_003",
        attrs={"n": 1},
    ),
]


@pytest.mark.parametrize("name", ["videos.jsonl", "videos.jsonl.gz"])
def test_video_records_round_trip(tmp_path, name):
    path = tmp_path / name
    write_jsonl(path, VIDEOS)
    assert read_jsonl(path, VideoRecord) == VIDEOS
    assert read_jsonl(path, VideoRecord, strict=True) == VIDEOS
    assert not list(tmp_path.glob(".*"))  # no temp files left


def test_gz_output_is_byte_reproducible(tmp_path):
    write_jsonl(tmp_path / "a.jsonl.gz", VIDEOS)
    write_jsonl(tmp_path / "b.jsonl.gz", VIDEOS)
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()
    first = gzip.decompress((tmp_path / "a.jsonl.gz").read_bytes()).decode().splitlines()[0]
    assert first == (
        '{"attrs":{},"compression":"c23","identity":"000","key":"000","label_key":"FF-REAL",'
        '"method":"original","pair_key":null,"source_id":null,"target_id":null}'
    )


def test_reader_errors_are_contract_errors(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text(
        '{"key": "a", "compression": null, "label_key": "X", "method": "m", "bogus": 1}\n'
    )
    with pytest.raises(
        ContractError,
        match=r"v.jsonl:1: not a valid VideoRecord: unexpected field\(s\) \['bogus'\]",
    ):
        read_jsonl(path, VideoRecord)
    path.write_text('{"key": "a"}\n')
    with pytest.raises(ContractError, match="missing 3 required positional arguments"):
        read_jsonl(path, VideoRecord)
    path.write_text("not json\n")
    with pytest.raises(ContractError, match="invalid JSON"):
        read_jsonl(path, VideoRecord)
    path.write_text('{"key": 1, "compression": null, "label_key": "X", "method": "m"}\n')
    assert read_jsonl(path, VideoRecord)[0].key == 1  # fast path does not check types...
    with pytest.raises(ContractError, match="key: Input should be a valid string"):
        read_jsonl(path, VideoRecord, strict=True)  # ...strict mode does


def test_iter_jsonl_dicts_yields_lineno_and_raw_dict(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text('{"key": "a"}\n\n{"key": "b"}\n')  # a blank line in between is skipped
    assert list(iter_jsonl_dicts(path)) == [(1, {"key": "a"}), (3, {"key": "b"})]


def test_iter_jsonl_dicts_shares_iter_jsonls_error_mapping(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text('{"key": "a"}\nnot json\n')
    with pytest.raises(ContractError, match=r"v\.jsonl:2: invalid JSON"):
        list(iter_jsonl_dicts(path))
    path.write_text("[1, 2]\n")
    with pytest.raises(ContractError, match=r"v\.jsonl:1: expected a JSON object"):
        list(iter_jsonl_dicts(path))
    with pytest.raises(ContractError, match="file not found"):
        list(iter_jsonl_dicts(tmp_path / "missing.jsonl"))


def test_writer_refuses_absolute_paths(tmp_path):
    bad = VideoRecord("k", None, "X", "m", attrs={"source": "/data/raw/k.mp4"})
    with pytest.raises(ContractError, match="absolute path"):
        write_jsonl(tmp_path / "v.jsonl", [bad])
    assert not list(tmp_path.iterdir())


SPLITS = [SplitRow("b", "c23", "test"), SplitRow("a", None, "train"), SplitRow("a", "c23", "val")]


@pytest.mark.parametrize("name", ["official.tsv", "official.tsv.gz"])
def test_split_files_are_canonical_and_hashed(tmp_path, name):
    path = tmp_path / name
    digest = write_split_tsv(path, SPLITS)
    assert digest == split_sha256(reversed(SPLITS))
    assert read_split_tsv(path) == sorted(SPLITS, key=lambda r: (r.key, r.compression or ""))
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if name.endswith(".gz") else raw.decode()
    assert text == "a\t\ttrain\na\tc23\tval\nb\tc23\ttest\n"


@pytest.mark.parametrize("name", ["videos.jsonl", "videos.jsonl.gz"])
def test_records_sha256_is_the_hash_of_the_sorted_lines_a_jsonl_file_holds(tmp_path, name):
    path = tmp_path / name
    write_jsonl(path, VIDEOS)
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if name.endswith(".gz") else raw.decode()
    lines = sorted(text.splitlines(keepends=True))

    assert records_sha256(VIDEOS) == hashlib.sha256("".join(lines).encode()).hexdigest()
    assert records_sha256(reversed(VIDEOS)) == records_sha256(VIDEOS)
    assert records_sha256(read_jsonl(path, VideoRecord)) == records_sha256(VIDEOS)


def test_records_sha256_of_nothing_is_the_hash_of_empty_content():
    assert records_sha256([]) == hashlib.sha256(b"").hexdigest()
    pairs = [PairRecord("F/1", "R/1", "target-id")]
    assert records_sha256(pairs) != records_sha256([])
    assert records_sha256(pairs) != records_sha256([PairRecord("F/1", "R/1", "other")])


def test_split_validation(tmp_path):
    with pytest.raises(ContractError, match="tabs or newlines"):
        split_sha256([SplitRow("a\tb", None, "train")])
    with pytest.raises(ContractError, match="split 'dev'"):
        split_sha256([SplitRow("a", None, "dev")])  # type: ignore[arg-type]
    (tmp_path / "s.tsv").write_text("a\tc23\n")
    with pytest.raises(ContractError, match="expected key<TAB>compression<TAB>split"):
        read_split_tsv(tmp_path / "s.tsv")


@pytest.mark.parametrize(
    "value",
    ["/data/x", "~/x", "run:/x", "--out=/x", "x=~/y", "file:///data/x", "/"],
)
def test_absolute_path_guard_flags(value):
    with pytest.raises(ContractError, match="absolute path"):
        assert_no_absolute_paths({"attrs": {"v": value}})


@pytest.mark.parametrize(
    "value",
    [
        "a/b",
        "ffpp/official",
        "Deepfakes/000_003",
        "https://example.org/x",
        "s3://bucket/key",
        "StyleGAN2 / ADA",
        "real / fake",
        "16 / 9",
    ],
)
def test_absolute_path_guard_passes(value):
    assert_no_absolute_paths({"attrs": {"v": value}})


def test_unreadable_files_are_contract_errors(tmp_path):
    (tmp_path / "plain.jsonl.gz").write_text("not gzip\n")
    with pytest.raises(ContractError, match=r"plain.jsonl.gz: cannot read \(BadGzipFile"):
        read_jsonl(tmp_path / "plain.jsonl.gz", VideoRecord)
    write_jsonl(tmp_path / "good.jsonl.gz", VIDEOS)
    (tmp_path / "cut.jsonl.gz").write_bytes((tmp_path / "good.jsonl.gz").read_bytes()[:-8])
    with pytest.raises(ContractError, match=r"cut.jsonl.gz: cannot read \(EOFError"):
        read_jsonl(tmp_path / "cut.jsonl.gz", VideoRecord)
    (tmp_path / "zipped.jsonl").write_bytes((tmp_path / "good.jsonl.gz").read_bytes())
    with pytest.raises(ContractError, match=r"zipped.jsonl: cannot read \(UnicodeDecodeError"):
        read_jsonl(tmp_path / "zipped.jsonl", VideoRecord)
    with pytest.raises(ContractError, match="file not found"):
        read_jsonl(tmp_path / "missing.jsonl", VideoRecord)
    with pytest.raises(ContractError, match="file not found"):
        read_split_tsv(tmp_path / "missing.tsv")


@pytest.mark.parametrize(
    ("model", "document"),
    [
        (DatasetCard, CARD),
        (LabelVocab, LABELS),
        (PackCard, "schema_version: 1\nname: dfwb-protocols\nversion: 1.0.0\ndatasets: [ffpp]\n"),
    ],
)
def test_cards_round_trip_through_yaml(model, document):
    card = model.model_validate(yaml.safe_load(document))
    dumped = yaml.safe_dump(card.model_dump(mode="json", by_alias=True), sort_keys=False)
    assert model.model_validate(yaml.safe_load(dumped)) == card
