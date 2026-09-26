# Installing extras

`dfwb` itself needs only the dependencies `uv sync` installs from a clone, and never pulls in
PyTorch on its own. Media probing, the face pipeline and its detection backends,
evaluation plots, training, the model zoo, Hugging Face backbones, PEFT adapters and Weights &
Biases logging are all optional extras, installed only when you need them.

| Extra | Adds |
|---|---|
| `preprocess` | Media probing and the face pipeline's own code (`opencv-python-headless`, `av`) |
| `face-insightface` | The `insightface` face-detection backend, on top of `preprocess` (`onnxruntime`) |
| `face-mediapipe` | The `mediapipe` face-detection backend, on top of `preprocess` (`mediapipe`) |
| `eval` | Bootstrap confidence intervals and plots (`scipy`, `matplotlib`) |
| `train` | The training stack: `torch`, `torchvision`, `lightning`, `timm`, `safetensors`, `pillow` |
| `hf` | Hugging Face backbones on top of `train` (`transformers`) |
| `peft` | Parameter-efficient fine-tuning on top of `train` (`peft`) |
| `wandb` | Weights & Biases logging (`wandb`) |
| `zoo` | Loading published checkpoints without the full training stack |
| `rich` | Nicer CLI output |
| `all` | `preprocess`, `face-mediapipe`, `eval`, `train`, `hf`, `peft`, `zoo` and `rich` together |

`face-insightface` and `wandb` are not part of `all`: insightface's `buffalo_l` weights are for
non-commercial research use only, and `wandb` needs a Weights & Biases account. Add either
explicitly if you use it. Extras compose, so install only what you need, for example:

```bash
uv sync --extra train --extra hf --extra peft
# or, into an editable install from a clone:
pip install -e ".[train,hf,peft]"
```

!!! note "Not on PyPI yet"
    dfwb is not published on PyPI: install from a clone, as above
    (`git clone https://github.com/dfwb-research/deepfake-workbench && cd deepfake-workbench`
    first). Once a release is published, `pip install "deepfake-workbench[train,hf,peft]"` will
    install the same combination directly, with no clone needed -- the same form the CLI's own
    `InstallationError` hints already use.

## Training: install a PyTorch build first

Install the PyTorch build that matches your hardware before the `train` (or `zoo`) extra, so the
extra finds it already installed instead of pulling the default build from PyPI. Pick the command
for your platform from the official selector at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), which covers both CPU
and the CUDA build matching your driver, for example:

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (check pytorch.org for the current index for your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

then:

```bash
uv sync --extra train
# or: pip install -e ".[train]"
```

## `preprocess`: media probing and the face pipeline's own code

```bash
uv sync --extra preprocess
# or, into an editable install from a clone:
pip install -e ".[preprocess]"
```

Installs PyAV and `opencv-python-headless`. Needed for `dfwb inventory build --probe`, for
`dfwb datasets synth toyfake` with real media (rather than `--no-media`), and for `dfwb preprocess
run` with any profile (every backend writes its cropped frames with OpenCV). Without it,
`dfwb preprocess run` stops before touching any video, with exit code 5 and the install command.

## Face-detection backends

Each backend is its own extra, on top of `preprocess`:

```bash
uv sync --extra face-insightface   # insightface's buffalo_l models, run with onnxruntime
uv sync --extra face-mediapipe     # Google MediaPipe's BlazeFace detector
```

(or, into an editable install from a clone, `pip install -e ".[face-insightface]"` /
`".[face-mediapipe]"`.) No extra is needed for the `center` backend (no detector, just a centred
crop) beyond `preprocess` itself.

### `face-insightface`: CPU by default, GPU by hand

`face-insightface` installs `onnxruntime`, the CPU build. To run the `insightface` backend on a
GPU, install `onnxruntime-gpu` in its place (they cannot both be installed at once — uninstall
`onnxruntime` first):

```bash
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu
```

then pass `--device cuda:<index>` to `dfwb preprocess run`. Without a working CUDA build of
onnxruntime, `--device cuda:...` runs on the CPU instead, with a warning.

insightface's `buffalo_l` weights are for non-commercial research use only; see [processing
profiles](concepts/processing-profiles.md#the-licence-gate) for the one-time acknowledgement
`dfwb preprocess run --accept-license` records before it downloads or uses them.

The backend uses two of the `buffalo_l` models, `det_10g.onnx` and `w600k_r50.onnx`, and looks for
them in `<cache root>/models/buffalo_l/`, then in `~/.insightface/models/buffalo_l/` (where
insightface itself keeps them). If neither has them, it downloads the `buffalo_l` release archive
(289 MB) once and unpacks just those two, each checked against its known sha256. A machine without
network access (or with `DFWB_OFFLINE=1`) needs the two files placed in one of those directories
by hand; a download that fails says so, naming both.

### `face-mediapipe`: two OpenCV distributions, one `cv2`

mediapipe depends on `opencv-contrib-python`, so the `face-mediapipe` extra installs it alongside
the `opencv-python-headless` that `preprocess` already installs. Both packages provide the same
`cv2` module; uninstalling either one (directly, or as a side effect of removing mediapipe) breaks
`cv2` for whichever is left, until the missing one is reinstalled:

```bash
pip install --force-reinstall opencv-python-headless
# or, with uv:
uv sync --reinstall-package opencv-python-headless
```

The `mediapipe` backend's detector model (a small, pinned `.tflite` file) is not bundled with
mediapipe; it is downloaded once, on first use, into the dfwb cache root, and checked against its
known sha256. `DFWB_OFFLINE=1` refuses to download it, so a machine without network access needs
the model file already in place (or `DFWB_OFFLINE` unset) the first time the backend runs.
