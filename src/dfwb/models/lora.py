"""One shared LoRA helper, used by every LoRA-capable backbone.

``peft`` is imported lazily, inside :func:`apply_lora`, so importing ``dfwb.models`` never pulls
it in: only building a backbone with ``freeze.mode == "lora"`` does.
"""

from __future__ import annotations

from torch import nn

from dfwb.core.errors import ConfigError, InstallationError
from dfwb.models.backbone import FreezeSpec

__all__ = ["apply_lora"]

_MAX_NAMED_CANDIDATES = 10


def apply_lora(module: nn.Module, spec: FreezeSpec) -> nn.Module:
    """Inject LoRA adapters into ``module``'s ``spec.targets`` layers, in place.

    peft freezes every parameter of ``module`` it did not add, so only the injected LoRA
    parameters are left trainable. Targets are checked against ``module`` before peft is called
    at all, so a bad target raises without touching ``module``.

    Raises:
        InstallationError: ``peft`` is not installed.
        ConfigError: none of ``spec.targets`` names a module of ``module``.
    """
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise InstallationError(
            "freeze.mode: 'lora' needs peft, which is not installed",
            hint='pip install "deepfake-workbench[peft]"',
        ) from exc

    targets = list(spec.targets)
    if not _any_target_matches(module, targets):
        raise _no_match_error(module, targets)

    config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=targets,
    )
    injected: nn.Module = inject_adapter_in_model(config, module)
    return injected


def _any_target_matches(module: nn.Module, targets: list[str]) -> bool:
    """peft's own target-module matching rule (checked ourselves, ahead of calling peft, so a
    bad target is reported before anything is mutated): a module's qualified name equals a
    target, or ends with ``"." + target``."""
    return any(
        name == target or name.endswith(f".{target}")
        for name, _ in module.named_modules()
        for target in targets
    )


def _no_match_error(module: nn.Module, targets: list[str]) -> ConfigError:
    candidates = [name for name, sub in module.named_modules() if isinstance(sub, nn.Linear)][
        :_MAX_NAMED_CANDIDATES
    ]
    return ConfigError(
        f"freeze.targets: {targets} matched no module of this backbone",
        hint=(
            "this backbone's Linear layers include "
            + (", ".join(candidates) if candidates else "none")
            + "; timm ViTs target 'qkv', while 'q_proj'/'v_proj' (the default) is Hugging Face "
            "naming"
        ),
    )
