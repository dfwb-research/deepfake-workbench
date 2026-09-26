# Protocols

A **protocol** is one versioned way of splitting a dataset into train, validation and test
videos, published by an installed **protocol pack** (a small, separately-installable Python
distribution that carries no code of its own beyond registering itself). `dfwb protocols` lists,
inspects, verifies, builds and lints protocols; `dfwb inventory` turns a dataset folder on disk
into the local inventory a protocol is checked against.

## Referring to a protocol

Most `dfwb protocols` commands take a reference of the form:

```
[<pack>:]<dataset>[/<scheme>][@<version or hash>]
```

- `<dataset>` is the only required part, e.g. `ffpp`. It resolves to that dataset's default
  scheme, in whichever installed pack publishes it.
- `<pack>:` scopes the reference to one pack. It is only needed when more than one installed
  pack publishes the same dataset id and the plain id would be ambiguous.
- `/<scheme>` names one of the dataset's split schemes explicitly, e.g. `ffpp/official`, instead
  of taking the dataset's default.
- `@<pin>` pins the reference to an exact pack version (`@1.4.0`, or a PEP 440 pre-, post- or
  dev-release such as `@0.1.0a2`) or to a scheme's content hash, given as its first six or more
  hex characters (`@3f9a1c2`). Score files and run configs record
  a pinned reference, so exactly what a run trained or evaluated against can be reproduced later.

A few valid references:

```
ffpp
ffpp/official
dfwb-protocols:ffpp/official
ffpp/official@1.4.0
ffpp/official@3f9a1c2e
```

Dataset and pack ids are lower-kebab-case (`celebdf-v2`, not `CelebDF-v2`); scheme names are
similarly plain, allowing letters, digits, `.`, `+` and `-`. An invalid reference is a usage
error (exit code 2, see below), with a hint showing the grammar.

## Split schemes

A dataset can publish more than one split scheme; `dfwb protocols list` shows every scheme of
every dataset in every installed pack, marking the default with `*`, and `dfwb protocols info
REF` shows one scheme's counts by split, compression and label. Every scheme belongs to one of a
small number of general families:

- **official** — the publisher's own train/validation/test split, taken as published, with only
  the parts the publisher actually released (a dataset that only ships an official test set has
  no official train or validation split).
- **an official test plus a derived train/validation carve** — the publisher's official test set
  is kept as is, and an identity-disjoint train/validation split is carved, using a seeded,
  deterministic rule so no identity crosses between splits, either from every other video or,
  for a dataset whose publisher also lists a train split, from that train split alone.
- **a fully derived, identity-disjoint split** — used when a dataset has no official split at
  all: every video's identity decides its split, by the same seeded, deterministic rule.
- **all-test** — every video is test. Useful for evaluating a detector on a dataset it was never
  trained on.
- **benchmark** — a small, seeded subset of a dataset's test videos, for a fast comparison
  across datasets rather than an exhaustive evaluation. It holds as many reals as fakes, or every
  real when there are fewer, so it is not always class-balanced. A benchmark can be defined at
  given compressions, recorded in its scheme card's parameters, so it is the same subset
  whichever other compressions a local copy holds: the FaceForensics++ and DeepFakeDetection
  benchmarks are defined at c23. For a recipe dataset (below) that holds only once every
  compression the card lists is present, since materialising rebuilds the whole video list,
  which covers all of them.

## Lists and recipes

Whether a dataset's key lists may be redistributed depends on its terms, so every dataset card
records the outcome of a terms review as its `distribution`:

- **list**: the pack ships the key lists themselves: `videos.jsonl.gz` (each video's key, label
  and identities), `pairs.jsonl.gz` (the real each fake was made from) and one
  `splits/<scheme>.tsv.gz` per scheme. Installing the pack is all you need to do.
- **recipe**: the terms do not allow redistributing even file names, so the published pack ships
  no key list at all, only `dataset.yaml`, `labels.yaml`, `NOTICE.md` and `PROVENANCE.json`. The
  card (`dataset.yaml`) still records everything needed to rebuild the lists and check them: each
  scheme's rule, its parameters and the sha256 of the split it produces, the sha256 of the video
  list (`videos_sha256`) and of the pair list (`pairs_sha256`), and the rule the pairs are drawn
  by (`pairing_rule`).
- **undecided**: the terms have not been reviewed yet. A released pack leaves such a dataset out
  and lists it under `withheld` in its `pack.yaml`.

For a recipe dataset you rebuild the lists yourself, from your own copy of the dataset:

```bash
dfwb inventory build DATASET
dfwb protocols materialize DATASET
```

A recipe rebuilds the published lists exactly, so it needs an exact copy of the release:

- every video of the release, in each compression the card lists (`compressions`). Rows of a
  compression the card does not list are left out before anything is rebuilt, so a copy that
  holds more compressions is fine;
- an inventory built by the version of the dataset's inventory builder the pack was built with.
  The pack's `PROVENANCE.json` records that version and the dfwb version the pack was built by;
  build the inventory with that dfwb version, or a later one whose builder for the dataset is
  still the same version.

A partial copy, such as one without one of the listed compressions, cannot be materialised: the
published hashes cover the whole release, and there is no hash per compression. What describes
only your own copy, such as where an audio track sits after unpacking, stays in your inventory
and is never part of the lists, so how you unpacked the release makes no difference.

`materialize` turns your inventory into the video list, recomputes every scheme with the rules
the pack was built with, draws the pairs with the dataset's inventory builder, and checks each of
those lists against the hash the card publishes. Only when every hash matches does it write them
to `<work root>/<dataset>/materialized/`: `videos.jsonl.gz`, `pairs.jsonl.gz` (when the dataset
has pairs), `splits/<scheme>.tsv.gz` for every scheme, and `hashes.json`, which records the card
hashes they matched. They are byte for byte the files a list pack would have shipped, so a recipe
gives the same guarantee of comparability as a list. From then on every command that loads the
protocol (`protocols info` and `verify`, `preprocess run`, `train`, `score`, `eval`) reads that
copy without being asked. Until you materialise, those commands stop with an error whose hint is
the command to run, and they ask you to materialise again if an upgraded pack publishes other
hashes.

A mismatch exits with code 4 and writes nothing, leaving any earlier materialised copy as it was.
Its message names every list whose hash differs, gives each scheme's rebuilt and published split
counts, the number of videos and the compressions your inventory gives next to those the card
lists, and the inventory builder versions when they are not the pack's; its hint names the likely
cause: a missing compression, missing or extra videos, videos that differ in label or another
field (another upstream release than the card's `release`, say), or the dfwb version to rebuild
the inventory with.

A pack can also publish a single scheme as a recipe while still shipping its videos: it leaves
out just that scheme's split file, and `dfwb protocols materialize DATASET/SCHEME` rebuilds that
one split in the same way.

## Verifying a local copy

`dfwb protocols verify REF` joins your local inventory (built with `dfwb inventory build`)
against the videos a protocol pack publishes for that dataset (for a recipe dataset, the
materialised videos), and buckets every video into one of:

- `have` — present locally, with the same label and method as the pack publishes.
- `missing` — published by the pack, but not found locally.
- `missing_requested` — the part of `missing` whose split was actually requested (see `--split`
  below); this is what decides whether verification counts as full coverage.
- `extra` — found locally, but not part of what the pack publishes for this dataset. For a list
  dataset this is normal for a partial or differently-organised local copy, and is not itself a
  failure. A recipe dataset has no partial copy: see "Lists and recipes" above.
- `label_mismatch` — present in both, but its label or method disagree between the local
  inventory and the pack. This usually means the local copy is a different release than the one
  the pack card describes.

By default every split the scheme assigns is requested, except `exclude`; `--split NAME`
(repeatable) narrows this to specific splits, e.g. to only require full coverage of `test`. When
enough videos of the same task are both missing and extra (ten or more of each), verify adds a
warning that the local copy may be a different upstream release than the one recorded in the
dataset card, rather than reporting it as plain incompleteness. The full report — every count, a
sample of up to twenty keys per bucket, and any warnings — is written to
`<work root>/<dataset>/verify/<scheme>.json`, so a later step (training, scoring) can record a
run against the same coverage summary without re-running verify.

For a materialised recipe dataset, verify compares your inventory with lists rebuilt from that
same inventory, so as long as the inventory has not changed since, its coverage is complete by
construction: only rows of a compression the card does not list show up, as `extra`.

## Exit codes

Every `dfwb` command uses the same small set of exit codes, and the `protocols` commands are no
exception:

| Code | Meaning | Where it shows up here |
|------|---------|-------------------------|
| 0 | Success | Full coverage on `verify`; no error-level issues on `lint`; a matching hash on `materialize`. |
| 2 | Usage or configuration problem | An invalid protocol reference; `--split` names a split the scheme does not have; no local inventory to verify against. |
| 3 | Partial coverage | `verify`: at least one requested split is missing a video. |
| 4 | Contract or data mismatch | `verify`: a video's label or method disagrees with the pack; `lint`: at least one error-level issue; `materialize`: a rebuilt list does not hash to its published value; `diff --expect-bump`: the bump claimed is smaller than what the differences between the two packs require (an unchanged pack requires none). |
| 5 | A needed optional dependency is missing | Commands that need the `preprocess` extra for media probing (e.g. `dfwb inventory build --probe`) when it is not installed. |

Every error prints an `error:` line describing what went wrong and a `hint:` line describing the
remedy; nothing prints a Python traceback unless `--debug` is passed.
