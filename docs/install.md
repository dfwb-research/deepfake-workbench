# Installing extras

`dfwb` itself needs only the dependencies in `uv sync` / `pip install deepfake-workbench`. Media
probing, the face pipeline and its detection backends are optional extras, installed only when you
need them.

## `preprocess`: media probing and the face pipeline's own code

```bash
uv sync --extra preprocess
# or, into an existing install:
pip install "deepfake-workbench[preprocess]"
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

(or `pip install "deepfake-workbench[face-insightface]"` / `[face-mediapipe]`.) No extra is
needed for the `center` backend (no detector, just a centred crop) beyond `preprocess` itself.

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
