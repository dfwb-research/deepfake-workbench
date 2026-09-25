# Installing with the training stack

`deepfake-workbench` never pulls in PyTorch on its own: `pip install deepfake-workbench` (or any
of `preprocess`, `eval`, `rich`) stays torch-free. Training, the model zoo, Hugging Face backbones,
PEFT adapters and Weights & Biases logging are all opt-in extras, and PyTorch itself is never
resolved as a transitive dependency of them — you install the build that matches your hardware
first, then add the extra.

## 1. Install a PyTorch build

Pick the command for your platform from the official selector at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), which covers both CPU
and the CUDA build matching your driver, for example:

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (check pytorch.org for the current index for your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2. Install deepfake-workbench with the extras you need

```bash
pip install "deepfake-workbench[train]"
```

Available extras:

| Extra | Adds |
|---|---|
| `preprocess` | Frame decoding and face-crop pipeline (`opencv-python-headless`, `av`) |
| `eval` | Bootstrap CIs, plots and Parquet reports (`scipy`, `matplotlib`, `pyarrow`) |
| `train` | The training stack: `torch`, `torchvision`, `lightning`, `timm`, `safetensors` |
| `hf` | Hugging Face backbones on top of `train` (`transformers`) |
| `peft` | Parameter-efficient fine-tuning on top of `train` (`peft`) |
| `wandb` | Weights & Biases logging (`wandb`) |
| `zoo` | Loading published checkpoints without the full training stack |
| `rich` | Nicer CLI output |
| `all` | `preprocess`, `eval`, `train`, `hf`, `peft`, `zoo` and `rich` together |

`wandb` needs a Weights & Biases account and is not part of `all`; add it explicitly if you use it.

Extras compose, so install only what you need, for example:

```bash
pip install "deepfake-workbench[train,hf,peft]"
```
