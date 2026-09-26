# Training

`dfwb train -c <config>` trains a `dfwb.train/1` config once per seed of `run.seeds`, each run into
its own self-describing run directory (`docs/concepts/detectors.md` lists what one holds).
`dfwb config init` writes a starter config that extends the `binary-frame` template;
`dfwb config show` prints the fully resolved config, and `dfwb config validate` checks it. This
page covers what the training keys do, and a few behaviours worth knowing before a long run.

## Checked before anything runs

Every problem the config alone can show is reported with its dotted path and a did-you-mean
before any data is read and before a run directory is created: an unknown key or value anywhere,
every component's parameters against the installed plugins (`model.backbone.freeze.mode`,
`train.callbacks[0].path`, ...), every `eval.metrics` spec and its parameters
(`eval.metrics[1]: metrics: unknown key 'eerr' (did you mean 'eer'?)`), `eval.aggregate`,
`train.precision`, the train transforms, the optimiser groups, the schedule, the Lightning
passthrough and the monitor. `dfwb config validate` runs the same checks without training.
Problems with the data itself (a protocol, a split, a processed store) are found next, once the
data is joined, still before any training.

## Clips: `data.clip`

```yaml
data:
  clip: {frames: 8, sampling: uniform, clips_per_video: {train: 1, eval: 4}, stride: 1}
```

A clip is `frames` (T) stored frames of one video; a video contributes `clips_per_video.train`
clips (c) to each training epoch and `clips_per_video.eval` to each validation pass.

- **`uniform`** spreads the c·T frames over the whole video. In **validation** they are evenly
  spaced and the same every time. In **training** the video's stored frames are split into c·T
  equal consecutive segments and one frame is drawn at random inside each, so every epoch trains on
  different frames, still spread across the whole video. When c·T equals the number of stored
  frames (for example `frames: 1, clips_per_video: {train: 32}` on a 32-frame store), each segment
  holds one frame and every epoch trains on every stored frame.
- **`consecutive`** and **`random-window`** take runs of T frames, `stride` apart: evenly spaced
  runs in validation, runs starting at random in training. A video with fewer stored frames than a
  run needs repeats its last frame (marked `padded` in the batch's `extras`).

Every random draw of an epoch (the frames, the transforms, the sampler's order) is seeded by the
run's seed, the epoch and the sample's index, never by the loader worker that reads it, so an
epoch draws the same batches however many workers there are, and again after a resume.

The trained detector records this regime in its `detector.json` (`meta.input.frames` and
`meta.input.sampling`; `random-window` is recorded as `consecutive`), so scoring it later builds
the same clips it was validated on. `model.input` overrides both, like any other input key.

## Data sources and loading

`data.train` and `data.val` list protocol splits (`{protocol, split, where}`). Every validation
source is scored and reported on its own (`val/<source>/<metric>`), and `val/video_<metric>` is
the mean over the sources where that metric is defined.

- **Balance.** `data.loader.balance` chooses how training draws: `none` (a seeded shuffle, the
  default), `video-label` (real and fake equally likely) or `source` (training sources drawn in
  proportion to their `weight`, default 1, whatever their sizes). A `weight` has no effect under
  another balance, and `dfwb train` warns when one is set there.
- **Pairs.** `data.pairs: true` trains on each training protocol's real/fake pairs, a pair's rows
  always in one batch and tagged with a shared `extras["dfwb/pair_id"]`. Ids are unique across
  training sources, so a pairwise loss matches partners on the id alone.
- **Workers.** `data.loader.num_workers` loader workers read each loader. The training loader's
  workers persist across epochs; each validation source's workers start for its validation pass
  and stop after it, so several validation sources never hold idle workers while training runs.
  Batches are pinned in memory when training on a CUDA GPU.
- **`data.test` is not run by `dfwb train` in this version.** It is accepted, and `dfwb train`
  warns that it is not used; score the trained run on the test split with `dfwb score` and
  evaluate the score file with `dfwb eval`.

## Precision: `train.precision`

`auto` (the default, and the templates' value) picks per device: `bf16-mixed` on a CUDA GPU with
native bfloat16 (compute capability 8.0 or later), `16-mixed` on any other CUDA GPU (older GPUs
only emulate bfloat16, slowly), and `32-true` on the CPU. It judges the GPU `--device` chose.
`32-true`, `bf16-mixed` and `16-mixed` set it explicitly. `env.json` records what it resolved to
(`"precision"`), and a resumed run keeps that precision rather than resolving `auto` again on
whatever machine it resumes on. Under either mixed precision the detector's head and the sigmoid still run in
float32, so scores keep their full resolution: metrics, checkpoint selection and the validation
score files never see half-precision rounding.

## Freezing a backbone: `freeze`

A backbone's `freeze` is `none`, `full`, `partial` (the last `trainable_blocks` of the
backbone's own blocks train), `norm-only` or `lora` (with the `peft` extra). Freezing stops
gradient updates to parameters; it does not put the backbone in evaluation mode. **BatchNorm
running statistics keep updating during training under `full` and `partial`**, since the
backbone runs in training mode like the rest of the detector. So `freeze: {mode: full}` on a
BatchNorm CNN (a ResNet, say) is not a strict linear probe: its normalisation statistics still
adapt to the training data. Transformer backbones use LayerNorm, which keeps no running
statistics, so this does not arise for them.

## Optimiser: `optim`

`adamw` and `sgd` take `lr`, `weight_decay`, per-group overrides and layer-wise decay:

```yaml
optim: {name: adamw, lr: 1.0e-4, weight_decay: 0.05, groups: {backbone: {lr_scale: 0.1}}}
```

`groups` keys on the backbone's own parameter groups (`embed`, `blocks.<i>`, `norm`, or
`backbone` for all of them) plus `head`, `pool` and `stem`, each with `lr_scale` and
`weight_decay`; `layer_decay: γ` multiplies block `i`'s learning rate by γ^(K−i).

**Weight decay applies to every parameter of a group, biases, normalisation weights and
position or class embeddings included.** Many ViT fine-tuning recipes exempt one-dimensional
parameters from decay; this version does not. To exempt a whole group, give it
`weight_decay: 0` (for example `groups: {norm: {weight_decay: 0}, embed: {weight_decay: 0}}`);
the biases and normalisation weights inside the blocks cannot be exempted on their own.

## Checkpoints, validation scores and the report

- `checkpoints/best` is rewritten whenever the monitor (`train.monitor`, `train.mode`) improves
  after a validation; `checkpoints/last` at the end of every epoch. `run:<dir>` loads `#best`
  unless `#last` is given.
- **`scores/val/<source>.scores.{csv,meta.json}` hold the last epoch's validation scores**, not
  the best checkpoint's. `dfwb eval scores/val/` reproduces the last validation's numbers exactly;
  to evaluate the checkpoint `run:<dir>` loads, score it with `dfwb score`.
- When the monitor's metric is undefined on every validation source (an AUC over a single-class
  split, say), the monitor falls back to `val/loss` (`min`) for the rest of the run, with a
  warning. `report.md` says when that happened, and lists every configured metric that was
  undefined on a source in the last validation; `metrics.json` records `"fallback": true`.
- A run without validation sources keeps `checkpoints/best` as a copy of `checkpoints/last`.
- **A corrupt stored frame never aborts training or validation.** Training and validation both
  repeat the clip's nearest still-good frame in a corrupt one's place, logging a warning and
  counting it (`train/repaired_frames`/`val/repaired_frames`, logged each epoch); a validation
  video whose every stored frame is corrupt is skipped instead (`val/videos_skipped`). `metrics.json`
  records the run's totals (`repaired_frames`, `videos_skipped`), and a one-line summary at the end
  of the run reports both when either is non-zero. A stored frame the wrong size for its
  processing profile is never tolerated this way: it is always a hard error, since it means the
  wrong store was chosen.

## Plugins in training

A plugin's loss is used by naming it in `loss:`, like the built-ins. A plugin's callbacks (the
`callbacks` registry) are added with `train.callbacks`, a list of `{name, **params}` entries
checked like every other component:

```yaml
train:
  callbacks: [{name: my-callback, every: 2}]
```

They run beside the framework's own callbacks, and their `state_dict()` is saved with the run's
resume state, so a resumed run carries it on. `train.lightning` cannot add callbacks; use
`train.callbacks`. `docs/guides/write-a-plugin.md` covers writing one.

**Samplers are not pluggable in this version**: there is no `samplers` registry. The training
sampler is one of the three `data.loader.balance` modes, or the pair-grouped batching of
`data.pairs: true`.

## Resuming

`dfwb train --resume <run dir>` carries an interrupted run on from the end of its last finished
epoch, from safetensors weights and optimiser state plus JSON counters and random states (never a
pickle), finishing where an uninterrupted run would have. It sets a few of Lightning's own loop
counters directly; resuming was written and tested against Lightning 2.6, and a Lightning release
that no longer has them is refused when fitting starts, naming the version to install.

## See also

- `docs/concepts/detectors.md`: the run directory, checkpoints and the `run:` detector source.
- `docs/guides/write-a-plugin.md`: registering a loss, a callback or a model component.
