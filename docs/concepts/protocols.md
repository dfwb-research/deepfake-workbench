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
- `@<pin>` pins the reference to an exact pack version (`@1.4.0`) or to a scheme's content hash,
  given as its first six or more hex characters (`@3f9a1c2`). Score files and run configs record
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
  is kept as is, and an identity-disjoint train/validation split is carved from the rest (or, for
  some datasets, from everything) using a seeded, deterministic rule, so no identity crosses
  between splits.
- **a fully derived, identity-disjoint split** — used when a dataset has no official split at
  all: every video's identity decides its split, by the same seeded, deterministic rule.
- **all-test** — every video is test. Useful for evaluating a detector on a dataset it was never
  trained on.
- **benchmark** — a small, seeded, class-balanced subset of a dataset's test videos, for a fast
  comparison across datasets rather than an exhaustive evaluation.

A scheme may also be published as a *recipe* rather than a list of keys: its rule, the rule's
parameters, and the sha256 hash the resulting split rows must match, but no key list. This lets a
dataset whose terms do not allow redistributing file names still be split reproducibly. `dfwb
protocols materialize REF` recomputes such a scheme from your own local inventory and, only if it
hashes to the published value, writes it under the work root, where every other command that
loads the protocol then finds it; a mismatch means your local inventory disagrees with the
publisher's manifest and nothing is written.

## Verifying a local copy

`dfwb protocols verify REF` joins your local inventory (built with `dfwb inventory build`)
against the videos a protocol pack publishes for that dataset, and buckets every video into one
of:

- `have` — present locally, with the same label and method as the pack publishes.
- `missing` — published by the pack, but not found locally.
- `missing_requested` — the part of `missing` whose split was actually requested (see `--split`
  below); this is what decides whether verification counts as full coverage.
- `extra` — found locally, but not part of what the pack publishes for this dataset. This is
  normal for a partial or differently-organised local copy, and is not itself a failure.
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

## Exit codes

Every `dfwb` command uses the same small set of exit codes, and the `protocols` commands are no
exception:

| Code | Meaning | Where it shows up here |
|------|---------|-------------------------|
| 0 | Success | Full coverage on `verify`; no error-level issues on `lint`; a matching hash on `materialize`. |
| 2 | Usage or configuration problem | An invalid protocol reference; `--split` names a split the scheme does not have; no local inventory to verify against. |
| 3 | Partial coverage | `verify`: at least one requested split is missing a video. |
| 4 | Contract or data mismatch | `verify`: a video's label or method disagrees with the pack; `lint`: at least one error-level issue; `materialize`: the rebuilt split does not hash to the published value; `diff --expect-bump`: the pack's actual version change is smaller than what the differences between the two packs require. |
| 5 | A needed optional dependency is missing | Commands that need the `preprocess` extra for media probing (e.g. `dfwb inventory build --probe`) when it is not installed. |

Every error prints an `error:` line describing what went wrong and a `hint:` line describing the
remedy; nothing prints a Python traceback unless `--debug` is passed.
