# Cross-dataset evaluation

This guide scores a detector over more than one protocol split in one command, then turns the
resulting score files into metrics with confidence intervals, a cross-dataset aggregate, and a
paired comparison against another detector. It uses the framework's own `toyfake` suite
throughout, so every command below runs on a bare clone with no dataset to obtain — a dataset
pack such as `dfwb-protocols` registers its own, larger suites the same way (see "Suites" below).

This assumes the toyfake quickstart in the README has already been run once: a processed store at
`toy-64-center-8f`, and a trained run at `runs/toy-cpu/latest`.

## Scoring a suite

A **suite** is a named, registered collection of `(protocol, split, where, group)` entries plus
aggregate rows — data a pack ships, not code. The framework registers one, `toyfake`, standing in
for a real cross-dataset panel: an in-domain entry (`toyfake/official`'s test split) and a second
entry on the identity-disjoint split, tagged `group: cross`.

```bash
dfwb score --detector zoo:random --suite toyfake --allow-input-mismatch
```

`--allow-input-mismatch` is needed here only because `zoo:random`'s input spec asks for a face
crop and the quickstart's `toy-64-center-8f` store is full-frame; scoring your own trained run
(whose input spec matches whatever it was trained on) needs no such override:

```bash
dfwb score --detector "run:runs/toy-cpu/latest#best" --suite toyfake
```

Either form writes one C5 file per suite entry, each named and cached exactly as a single
`--protocol`/`--split` score would be (`docs/concepts/score-files.md`), and prints where each one
landed:

```
protocol                 split  group      csv                                     coverage   cached  frames
toyfake/official          test  in-domain  runs/scores/random/.../test-....scores.csv  ok=41/41   False
toyfake/ident-72-14-14    test  cross      runs/scores/random/.../test-....scores.csv  ok=12/12   False
```

## Evaluating a suite: metrics, bootstrap CIs and the aggregate

```bash
dfwb eval runs/scores/random/toyfake-official/*.scores.csv \
          runs/scores/random/toyfake-ident-72-14-14/*.scores.csv \
          --suite toyfake --metrics auc,eer --bootstrap 2000
```

The terminal prints one row per input file: `n` (how many rows the `--missing` policy actually fed
the metric), coverage, and each metric's point estimate with a stratified bootstrap confidence
interval over videos (`--bootstrap N --seed S`; `--bootstrap 0` reports the point value alone,
with no interval — useful for a quick, small-sample sanity check where a CI would be mostly
noise). `--suite toyfake` additionally matches each file to the suite entry whose protocol, split
and `where` it was scored on and computes its aggregate rows — here, the mean AUC of the
`in-domain` group and of the `cross` group (one file each, in this small example; a real panel's
cross-dataset group averages several) — but the plain terminal table only ever shows the per-file
rows; add `--out DIR` (writes `DIR/report.md` and `DIR/metrics.json`, with a `## suite` section)
or `--json` to see the aggregate itself:

```
## suite

| group     | metric | how  | value  | n_entries | n_expected |
| --------- | ------ | ---- | ------ | --------- | ---------- |
| in-domain | auc    | mean | 0.4475 | 1         | 1          |
| cross     | auc    | mean | 0.7143 | 1         | 1          |
```

`--by method|family|compression|label_key` breaks each file down further, using the pack's own
`labels.yaml`; for a fake-side dimension a method's group is evaluated against every real row of
the same dataset(s) too (the usual per-method-AUC convention), not against no reals at all.

## Coverage policy

Every file's coverage (`expected`/`ok`/`missing`/`error`) is always reported, unaffected by
`--missing`; `--missing` only decides what a metric is actually computed over:

- **`exclude`** (the default) — drop non-`ok` rows.
- **`as-real`**, **`as-fake`**, **`as-chance`** — keep every row, substituting a score of `0.0`,
  `1.0` or `0.5` (P(fake)) for a non-`ok` one, so a detector cannot improve its apparent metric
  merely by failing to score the videos it would have done worst on.

`--min-coverage 0.99` (the default) sets the exit code to `3` — the tables are still built and
printed, nothing raises — when any file's `ok` fraction falls below it:

```bash
dfwb eval my.scores.csv --min-coverage 1.0 --bootstrap 0
echo $?   # 3 if my.scores.csv has any missing or error row
```

## Comparing two detectors: `compare` and DeLong

`dfwb eval compare` never re-joins two files against the pack: it works from the intersection of
their `ok` rows, keyed by `(dataset, key, compression)`, and always reports how big that
intersection is (`n`) and how many `ok` rows each file has that the other lacks (`only_a`,
`only_b`), which the comparison leaves out. Two files that give a shared row different labels
(scored under different label mappings, say) are refused rather than compared. Score the run and
`zoo:random` on the *same* split so their rows overlap:

```bash
dfwb score --detector "run:runs/toy-cpu/latest#best" --protocol toyfake/official --split test
dfwb score --detector zoo:random --protocol toyfake/official --split test --allow-input-mismatch

dfwb eval compare runs/scores/tiny-cnn-mean-linear/toyfake-official/*.scores.csv \
                  runs/scores/random/toyfake-official/*.scores.csv \
                  --metrics auc --bootstrap 2000
```

For every metric, `compare` reports both files' point values, a paired-bootstrap confidence
interval of their difference (stratified by label over the shared rows), and, for `auc`
specifically, the DeLong test (needs the `[eval]` extra: scipy) — a `z` statistic and p-value for
whether the two AUCs differ. Comparing more than two files runs every pair and Holm-corrects the
DeLong p-values across all of them, reported as `holm_applied`.

Two edge cases have exact answers rather than estimates:

- **Zero paired variance.** A constant score (such as `zoo:chance`) or a perfect separator has no
  spread for DeLong to estimate. Equal AUCs then give `z = 0`, `p = 1`; unequal ones give an
  infinite `z` and `p = 0`, since the AUCs certainly differ. JSON has no infinity, so `--json`
  writes that `z` as the string `"inf"` or `"-inf"`.
- **Fewer than two rows in a class.** The DeLong covariance cannot be estimated at all, so
  `delong_z` and `delong_p` are `null`, `delong_undefined` says why, and that pair is left out of
  the Holm correction. The AUCs themselves are still reported.

## Calibration and importing foreign scores

`dfwb eval calibrate --fit val.scores.csv --apply test.scores.csv --method temperature` (or
`platt`/`isotonic`) fits on one file and applies to another, writing a new C5 file whose meta
records the calibration's method and parameters — generic post-hoc calibration, not a training
change. `dfwb eval import` brings in scores your own code already produced;
`docs/guides/reproduce-a-run.md` and, for the no-training case, the README's "I only have score
files from my own code" journey both use it.

## Suites are data, not code

A suite is a plain YAML file registered by a pack under the `eval_suites` registry — the
framework ships only `toyfake`, built entirely from the synthetic pack above. A dataset pack such
as `dfwb-protocols` registers its own suites the same way (for example, a genuine cross-dataset
panel scored on several real datasets); once such a pack is installed, `dfwb score --suite
<name>` and `dfwb eval --suite <name>` work against it exactly as they do against `toyfake` here,
with no code change.

## See also

- `docs/concepts/score-files.md` — the C5 schema, statuses, the cache, and what a detector source
  can set.
- `docs/guides/add-a-detector.md` — the `py:` source and adapter cards, for scoring a detector of
  your own or a published one.
- `docs/guides/reproduce-a-run.md` — from a run directory to a score file comparable with others.
