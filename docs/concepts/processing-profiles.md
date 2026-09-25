# Processing profiles

A **processing profile** is a named, hashed recipe for turning a dataset's videos into a store of
cropped, lossless face frames: which backend finds faces, how the found face is tracked across a
clip, how it is cropped, which frames are sampled, and how the video is decoded. `dfwb preprocess
run` processes a dataset with one profile; `dfwb preprocess profiles` lists every profile dfwb
ships; `dfwb preprocess status` and `dfwb preprocess merge` inspect and combine what a run wrote.

A profile can be a shipped name (below) or the path of your own profile YAML file, in the same
shape `dfwb schema export c3` describes. Either way, its id is a readable slug plus the first
eight hex characters of the sha256 of its whole contents (`ProcessingProfile.profile_id()`), so
two profiles that differ in even one setting never share a store, and a profile YAML that changes
gets a new store rather than silently mixing with the old one's rows.

## Shipped profiles

| id | backend | detection size | crop | sampling | EMA | for |
|----|---------|-----------------|------|----------|-----|-----|
| `face-256-1.3x-64f` | insightface | 256 | 1.3x / 256 / square | uniform 64 | — | reproduces the author's earlier FaceForensics++ stores; the parity setting to reach for when comparing against past results. |
| `face-256-1.3x-32f` | insightface | 256 | 1.3x / 256 / square | uniform 32 | — | the same backend and crop as the parity profile, at half the frames, for quicker runs and smaller stores. |
| `face-256-1.3x-64fc` | insightface | 256 | 1.3x / 256 / square | first-consecutive 64 | 0.7 | 64 adjacent frames instead of spread-out ones, with the tracked box smoothed across them — for models that use frame-to-frame motion. |
| `face-256-1.3x-32f-mp` | mediapipe | — | 1.3x / 256 / square | uniform 32 | — | the permissive twin of `face-256-1.3x-32f`: same crop and sampling, MediaPipe's Apache-2.0 detector instead of insightface's non-commercial weights, so it needs no licence acknowledgement. |
| `toy-64-center-8f` | center | — | 1.0x / 64 / square | uniform 8 | — | a tiny, dependency-free profile with no detector at all, for smoke tests and for data whose frames are already face crops (e.g. WildDeepfake). |

`dfwb preprocess profiles` (add `--json` for a machine-readable form) prints this same table,
computed from the shipped YAML files rather than kept in sync by hand, so it always matches what a
run actually does. "Detection size" is the side, in pixels, of the square image insightface's
detector runs on; MediaPipe and `center` have no such setting. "EMA" is the smoothing weight
applied to the tracked box between adjacent frames (`smoothed = ema * raw + (1 - ema) *
previous_smoothed`, reset whenever the frame gap is more than two); it only helps when sampled
frames are close together, so only `face-256-1.3x-64fc` sets it.

Every insightface profile's weights need a one-time licence acknowledgement; see below.

## The store layout

A run's output lives at `<work root>/<dataset>/processed/<profile_id>/`, one store per dataset and
profile:

- `profile.json`, written once, on first use: the profile itself, its sha256 and id, and the
  backend that produced its faces (its version, licence, and whatever it records about how faces
  were found — model files and their sha256s, thresholds, the execution providers used).
- `index.jsonl`, one line per video, appended as each one finishes: its key, compression, outcome
  (`ok`, `no_face`, `decode_error`, `too_short` or `skipped`), how many frames were written, their
  indices, its output directory (relative to the store), and, when it is not `ok`, a reason.
- one output directory per video, `<key>/<compression or "_">/` (a video's key already carries a
  `/`, so this nests one directory per task inside one per dataset), holding one
  `frame_<index:06d>.png` per kept frame — lossless, `cv2.imwrite` with PNG compression level 6,
  never JPEG — and `clip.json`.

`clip.json` describes the whole clip:

```json
{
  "key": "BLEND_A/p000_p015",
  "compression": null,
  "source": {"total_frames": 24, "fps": 8.0, "width": 64, "height": 64, "decoder": "pyav"},
  "frames": [
    {"index": 0, "bbox": [0.0, 0.0, 64.0, 64.0], "score": 1.0, "landmarks5": null}
  ],
  "failed_frames": [
    {"index": 12, "reason": "no-face"}
  ],
  "track": {"strategy": "largest-then-iou", "identity_switch": false}
}
```

`frames` lists every kept frame, in the order it was written, with its source index, its crop box
in the *source* frame's pixels, the detector's score, and its five landmarks when the profile asks
for them and the backend provides them. `failed_frames` lists every sampled frame that was decoded
but produced no usable, croppable face — dropped, not interpolated, with why. `track.strategy` is
the profile's tracking strategy, and `track.identity_switch` says whether the tracker ever had to
fall back from following the previous frame's face (by IoU) to picking the largest face in the
frame instead, which can mean it followed more than one person across the clip.

A store is never written inside a datasets root: raw data is read-only, always.

## Resuming and redoing a run

`dfwb preprocess run` skips a video the store already holds an outcome for, unless that outcome's
status is one you asked to redo with `--redo`: a video with **any** row in `index.jsonl` — `ok` or
not — counts as done and is left alone, except when `--redo` names its exact status. Running
`--redo no_face` retries only the videos that found no usable face; `--redo decode_error,too_short`
retries both of those; leaving `--redo` off retries nothing that already has a row, so re-running
the same command after it finishes (or after it was interrupted) only processes what is left.

A video whose processing is killed mid-way — the process interrupted, or a worker crashing on a
video that breaks native decoding or inference — never leaves a half-written output directory
behind: every video is written into a private temporary directory first, and only swapped into
place once it is fully written and its outcome is `ok`. The next run for that video starts clean,
whatever was left behind by the one that was killed.

## Sharding across machines

A large dataset can be split across several machines. `--shard I/N` restricts a run to the share
of the videos whose key hashes to `I` (out of `N` shards, `0 <= I < N`); the same `--shard` on
every machine, with `N` matching, partitions the whole scope with nothing left out and nothing
counted twice. Each shard writes to its own `index.shard-<I>-of-<N>.jsonl` instead of
`index.jsonl`, so two machines processing different shards never write the same file, and for as
long as a shard's run may still be appending to that file it holds a `.running` marker next to it,
removed however the run ends (normally, by an error, or by an interrupt).

`dfwb preprocess merge` combines every shard file of a store (and any existing `index.jsonl`) back
into a single `index.jsonl`, then removes the shard files; a video's later attempt (as recorded in
a shard file) wins over an earlier `index.jsonl` row unless the earlier one was already `ok` and
the later one is not. `merge` refuses to run while any shard's `.running` marker still exists —
that shard's run may still be writing to its file — so merging always sees each shard's finished
state. `dfwb preprocess status` reports a still-sharded, not-yet-merged store just as accurately as
a merged one: it folds in every shard file itself, the same way `merge` would, without writing
anything.

## The licence gate

insightface's `buffalo_l` detection and recognition weights are for non-commercial research use
only — stricter terms than the MIT-licensed code that reads them. Before the `insightface` backend
is built, it checks that this has been acknowledged once on the machine; without that
acknowledgement, `dfwb preprocess run` exits with code 5 and a hint. Pass `--accept-license` to
record the acknowledgement and continue in the same command:

```bash
dfwb preprocess run ffpp --profile face-256-1.3x-64f --accept-license
```

The acknowledgement is written once, atomically, to a small JSON file under this machine's user
state directory (`DFWB_STATE_DIR` overrides the location, e.g. for a container image built once
and reused) and is not asked for again on later runs on the same machine, with the same profile or
a different one that uses the same weights. `dfwb doctor` lists every acknowledgement recorded on
the machine. MediaPipe's and `center`'s weights carry no such restriction, so profiles that use
them (`face-256-1.3x-32f-mp`, `toy-64-center-8f`) never need `--accept-license`.
