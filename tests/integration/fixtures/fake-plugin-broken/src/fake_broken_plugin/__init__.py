"""Test fixture: registers one component, then fails."""


def register(api):
    api.heads.add(
        "half-registered", target="fake_broken_plugin:Head", summary="Registered before the failure"
    )
    raise RuntimeError("simulated failure inside register()")
