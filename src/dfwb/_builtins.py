"""Registers the framework's built-in components through the same plugin API as any plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from dfwb.core.plugins import PluginAPI

# The package every built-in inventory builder lives in.
_BUILDERS_PACKAGE: Final = "dfwb.preprocess.inventory.builders"

# Built-in inventory builders, one row each:
#   (dataset id, "<module>:<Class>" inside dfwb.preprocess.inventory.builders,
#    display name, the dataset's expected folder under a datasets root)
# The folder is stored as registry metadata, so `dfwb datasets list` and `dfwb doctor` can
# locate every dataset without importing a single builder module.
INVENTORY_BUILDERS: tuple[tuple[str, str, str, str], ...] = (
    ("ffpp", "ffpp:FaceForensicsBuilder", "FaceForensics++", "FaceForensics++"),
    ("dfd", "dfd:DeepFakeDetectionBuilder", "DeepFakeDetection", "FaceForensics++"),
    ("uadfv", "uadfv:UADFVBuilder", "UADFV", "UADFV"),
    ("celebdf-v1", "celebdf:CelebDFv1Builder", "Celeb-DF v1", "Celeb-DF-v1"),
    ("celebdf-v2", "celebdf:CelebDFv2Builder", "Celeb-DF v2", "Celeb-DF-v2"),
    ("celebdf-v3", "celebdf:CelebDFv3Builder", "Celeb-DF v3", "Celeb-DF-v3"),
    ("dfdc", "dfdc:DFDCBuilder", "DFDC", "DFDC"),
    ("dfdc-p", "dfdcp:DFDCPreviewBuilder", "DFDC Preview", "DFDC-P"),
    (
        "deeperforensics",
        "deeperforensics:DeeperForensicsBuilder",
        "DeeperForensics-1.0",
        "DeeperForensics-1.0",
    ),
    ("wilddeepfake", "wilddeepfake:WildDeepfakeBuilder", "WildDeepfake", "WildDeepfake"),
    ("ffiw10k", "ffiw10k:FFIW10KBuilder", "FFIW-10K", "FFIW10K"),
    ("kodf", "kodf:KoDFBuilder", "KoDF", "KoDF"),
    ("dfdm", "dfdm:DFDMBuilder", "DFDM", "DFDM"),
    (
        "fakeavceleb",
        "fakeavceleb:FakeAVCelebBuilder",
        "FakeAVCeleb v1.2",
        "FakeAVCeleb-v1_2",
    ),
    ("polyglotfake", "polyglotfake:PolyGlotFakeBuilder", "PolyGlotFake", "PolyGlotFake"),
    ("deepspeak-v1", "deepspeak:DeepSpeakV1Builder", "DeepSpeak v1", "DeepSpeak-v1"),
    ("deepspeak-v2", "deepspeak:DeepSpeakV2Builder", "DeepSpeak v2", "DeepSpeak-v2"),
    ("idforge-v1", "idforge:IDForgeV1Builder", "IDForge-v1", "IDForge-v1"),
    ("lav-df", "lavdf:LAVDFBuilder", "LAV-DF", "LAV-DF"),
    (
        "av-deepfake1m-pp",
        "avdeepfake1mpp:AVDeepfake1MPPBuilder",
        "AV-Deepfake1M++",
        "AV-Deepfake1M++",
    ),
    (
        "talkingheadbench",
        "talkingheadbench:TalkingHeadBenchBuilder",
        "TalkingHeadBench",
        "TalkingHeadBench",
    ),
    ("toyfake", "toyfake:ToyfakeBuilder", "toyfake (synthetic)", "toyfake"),
)

# Protocol packs shipped inside the framework, one row each:
#   (pack name, "<package>:<directory>" of the pack, one-line summary)
# A pack is data: registering it locates a directory and imports nothing.
PROTOCOL_PACKS: tuple[tuple[str, str, str], ...] = (
    ("toyfake", "dfwb:_packs/toyfake", "Built-in synthetic toyfake protocol pack"),
)

# The package every built-in face backend lives in.
_FACE_BACKENDS_PACKAGE: Final = "dfwb.preprocess.face.backends"

# Built-in face backends, one row each:
#   (key, "<module>:<Class>" inside dfwb.preprocess.face.backends, summary,
#    the modules it needs, which name the extra to install when one is missing)
FACE_BACKENDS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "center",
        "center:CenterBackend",
        "No detection: the centred square of each frame (pre-cropped data, smoke tests)",
        (),
    ),
    (
        "insightface",
        "insightface:InsightFaceBackend",
        "insightface buffalo_l detection and embeddings via onnxruntime "
        "(weights: non-commercial research only)",
        ("onnxruntime", "cv2"),
    ),
    (
        "mediapipe",
        "mediapipe:MediaPipeBackend",
        "MediaPipe BlazeFace detection (Apache-2.0)",
        ("mediapipe",),
    ),
)


def register(api: PluginAPI) -> None:
    """Register built-in components.

    Built-ins are added here as their layers are implemented; each one is an import-path target,
    so registering stays cheap and imports nothing heavy.
    """
    api.metrics.add(
        "auc",
        target="dfwb.eval.metrics:auc",
        summary="ROC AUC via the rank statistic, ties averaged; undefined for a single class",
    )
    api.metrics.add(
        "ap",
        target="dfwb.eval.metrics:ap",
        summary="Average precision, the step-wise area under the precision-recall curve",
    )
    api.metrics.add(
        "eer",
        target="dfwb.eval.metrics:eer",
        summary="Equal error rate: the FPR=FNR crossing of the ROC curve, linearly interpolated",
    )
    api.metrics.add(
        "acc",
        target="dfwb.eval.metrics:acc",
        summary="Accuracy at a threshold on P(fake) (thr=0.5 by default)",
    )
    api.metrics.add(
        "tpr",
        target="dfwb.eval.metrics:tpr",
        summary="Largest TPR achievable with FPR at or below fpr= (interp=true to interpolate)",
    )
    api.metrics.add(
        "fpr",
        target="dfwb.eval.metrics:fpr",
        summary="Smallest FPR needed to reach at least tpr= (interp=true to interpolate)",
    )
    api.metrics.add(
        "ece",
        target="dfwb.eval.metrics:ece",
        summary="Expected calibration error over bins= bins (adaptive=true for equal-mass bins)",
    )
    api.metrics.add(
        "brier",
        target="dfwb.eval.metrics:brier",
        summary="Brier score: the mean squared error of P(fake) against the label",
    )
    api.metrics.add(
        "nll",
        target="dfwb.eval.metrics:nll",
        summary="Binary cross-entropy of P(fake) against the label, clipped at 1e-7",
    )
    api.metrics.add(
        "aurc",
        target="dfwb.eval.metrics:aurc",
        summary="Area under the risk-coverage curve for selective prediction",
    )
    api.backbones.add(
        "tiny-cnn",
        target="dfwb.models.backbones.tiny_cnn:TinyCNN",
        summary="A tiny 3-block CNN backbone for CPU tests and toy training runs",
        requires=("torch",),
    )
    api.backbones.add(
        "timm",
        target="dfwb.models.backbones.timm_backbone:TimmBackbone",
        summary="Any timm image classification backbone, pooled the model's own way",
        requires=("torch", "timm"),
    )
    api.backbones.add(
        "hf-vision",
        target="dfwb.models.backbones.hf_vision:HFVisionBackbone",
        summary="Any Hugging Face vision transformer (CLIP, DINOv2, SigLIP 2, ...) via AutoModel",
        requires=("torch", "transformers"),
    )
    api.temporal_pools.add(
        "mean",
        target="dfwb.models.pools:MeanPool",
        summary="The mean feature over time",
        requires=("torch",),
    )
    api.temporal_pools.add(
        "max",
        target="dfwb.models.pools:MaxPool",
        summary="The per-channel maximum feature over time",
        requires=("torch",),
    )
    api.temporal_pools.add(
        "attention",
        target="dfwb.models.pools:AttentionPool",
        summary="A small learned pool: a linear score per frame, softmax-weighted over time",
        requires=("torch",),
    )
    api.heads.add(
        "linear",
        target="dfwb.models.heads:LinearHead",
        summary="Dropout, then a single Linear layer",
        requires=("torch",),
    )
    api.heads.add(
        "mlp",
        target="dfwb.models.heads:MLPHead",
        summary="Linear, an optional BatchNorm, GELU, Dropout, then a Linear output layer",
        requires=("torch",),
    )
    api.transforms.add(
        "resize",
        target="dfwb.data.transforms:Resize",
        summary="Resize every frame of the clip to size",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "center-crop",
        target="dfwb.data.transforms:CenterCrop",
        summary="Centre-crop every frame of the clip to size",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "random-resized-crop",
        target="dfwb.data.transforms:RandomResizedCrop",
        summary="One random crop (scale, ratio) per clip, resized to size, same for every frame",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "hflip",
        target="dfwb.data.transforms:HorizontalFlip",
        summary="Flips the whole clip left-right with probability p",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "color-jitter",
        target="dfwb.data.transforms:ColorJitter",
        summary="Brightness/contrast/saturation/hue jitter, one draw per clip",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "grayscale",
        target="dfwb.data.transforms:Grayscale",
        summary="Converts the whole clip to grayscale with probability p, one decision per clip",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "gaussian-blur",
        target="dfwb.data.transforms:GaussianBlur",
        summary="Gaussian blur with a sigma drawn once per clip",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "gaussian-noise",
        target="dfwb.data.transforms:GaussianNoise",
        summary="Adds one noise field per clip (std), broadcast to every frame, clamped to [0, 1]",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "jpeg",
        target="dfwb.data.transforms:Jpeg",
        summary="Round-trips the clip through JPEG at one quality per clip",
        requires=("torch", "torchvision"),
    )
    api.transforms.add(
        "normalize",
        target="dfwb.data.transforms:Normalize",
        summary="Per-channel (x - mean) / std; for input adaptation, refused inside transforms",
        requires=("torch", "torchvision"),
    )
    api.losses.add(
        "bce",
        target="dfwb.train.losses:BCELoss",
        summary="Binary cross-entropy on the logit, with an optional pos_weight",
        requires=("torch",),
    )
    api.losses.add(
        "ce",
        target="dfwb.train.losses:CELoss",
        summary="Categorical cross-entropy for multi-class heads, with label smoothing",
        requires=("torch",),
    )
    api.losses.add(
        "focal",
        target="dfwb.train.losses:FocalLoss",
        summary="Binary focal loss on the logit (alpha, gamma); gamma=0, alpha=None is bce",
        requires=("torch",),
    )
    api.losses.add(
        "label-smoothing-bce",
        target="dfwb.train.losses:LabelSmoothingBCELoss",
        summary="Binary cross-entropy with smoothed targets y(1-eps) + eps/2",
        requires=("torch",),
    )
    api.detector_sources.add(
        "run",
        target="dfwb.models.source:load_run",
        summary="Rebuild a detector saved by training, from its run directory",
        requires=("torch",),
    )
    for dataset_id, target, name, folder in INVENTORY_BUILDERS:
        api.inventory_builders.add(
            dataset_id, target=f"{_BUILDERS_PACKAGE}.{target}", summary=name, folder=folder
        )
    for key, target, summary, requires in FACE_BACKENDS:
        api.face_backends.add(
            key, target=f"{_FACE_BACKENDS_PACKAGE}.{target}", summary=summary, requires=requires
        )
    for pack, target, summary in PROTOCOL_PACKS:
        api.protocol_packs.add(pack, target=target, summary=summary)
