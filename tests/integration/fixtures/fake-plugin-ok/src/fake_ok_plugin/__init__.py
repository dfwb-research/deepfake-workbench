"""Test fixture: registers two components without importing dfwb or torch."""

DFWB_PLUGIN_API = ">=1.0,<2"


def register(api):
    api.layers.add(
        "fake-stem",
        target="fake_ok_plugin.modules:FakeStem",
        summary="Fake stem layer",
        requires=("torch",),
    )
    api.losses.add("fake-loss", target="fake_ok_plugin:FakeLoss", summary="Fake loss")


class FakeLoss:
    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight
