from importlib import metadata
from types import SimpleNamespace

import pytest

from dfwb.core import plugins
from dfwb.core.errors import UnknownKeyError
from dfwb.core.plugins import PluginStatus, api, api_compatible, load_plugins


class FakeEntryPoint:
    """Duck-typed stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, register, *, dist="fake-dist", version="1.2.3", group="dfwb.plugins"):
        self.name = name
        self.value = f"fake_module_{name}:register"
        self.group = group
        self._register = register
        self.dist = SimpleNamespace(name=dist, version=version) if dist else None

    def load(self):
        if isinstance(self._register, BaseException):
            raise self._register
        return self._register


@pytest.fixture
def entry_points(monkeypatch):
    groups = {"dfwb.plugins": [], "dfwb.builtins": []}
    monkeypatch.setattr(plugins, "_entry_points", lambda group: list(groups[group]))
    return groups


def _register_stem(api):
    api.layers.add("stem", target="tests.unit.core._targets:Stem", summary="a stem")


def test_load_is_lazy_and_idempotent(entry_points):
    calls = []

    def register(api):
        calls.append(api)
        _register_stem(api)

    entry_points["dfwb.plugins"].append(FakeEntryPoint("good", register))
    assert plugins._report is None
    assert "stem" in api.layers  # first read triggers loading
    assert calls == [api]
    report = load_plugins()
    assert load_plugins() is report
    assert calls == [api]
    (record,) = report.records
    assert record.status is PluginStatus.OK
    assert record.provider == "fake-dist"
    assert record.version == "1.2.3"
    assert record.entries == ("layers/stem",)
    assert api.layers.entry("stem").provider == "fake-dist"


def test_failing_register_is_isolated_and_rolled_back(entry_points):
    def broken(api):
        api.heads.add("half", target="tests.unit.core._targets:make_head", summary="x")
        raise RuntimeError("boom\nsecond line")

    entry_points["dfwb.plugins"] += [
        FakeEntryPoint("broken", broken, dist="broken-dist"),
        FakeEntryPoint("good", _register_stem, dist="good-dist"),
    ]
    report = load_plugins()
    statuses = {r.name: (r.status, r.reason) for r in report.records}
    assert statuses["broken"] == (PluginStatus.FAILED, "RuntimeError: boom")
    assert statuses["good"] == (PluginStatus.OK, None)
    assert "half" not in api.heads
    assert "stem" in api.layers
    with pytest.raises(UnknownKeyError) as info:
        api.heads.entry("half")
    assert "failed to load (RuntimeError: boom)" in info.value.hint


def test_entry_point_that_cannot_be_imported_is_failed(entry_points):
    entry_points["dfwb.plugins"].append(
        FakeEntryPoint("missing", ModuleNotFoundError("No module named 'x'"))
    )
    (record,) = load_plugins().records
    assert record.status is PluginStatus.FAILED
    assert record.reason == "ModuleNotFoundError: No module named 'x'"


def test_api_version_mismatch_is_skipped_with_warning(entry_points, monkeypatch, caplog):
    module = SimpleNamespace(DFWB_PLUGIN_API=">=2.0,<3")
    monkeypatch.setitem(__import__("sys").modules, "future_plugin", module)

    def register(api):  # pragma: no cover - must not run
        raise AssertionError

    register.__module__ = "future_plugin"
    entry_points["dfwb.plugins"].append(FakeEntryPoint("future", register, dist="future-dist"))
    with caplog.at_level("WARNING", logger="dfwb.core.plugins"):
        (record,) = load_plugins().records
    assert record.status is PluginStatus.SKIPPED
    assert record.reason == "needs plugin API >=2.0,<3, this dfwb provides 1.0"
    assert "future-dist" in caplog.text


def test_invalid_api_specifier_is_failed(entry_points, monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules, "odd_plugin", SimpleNamespace(DFWB_PLUGIN_API="~1")
    )

    def register(api):  # pragma: no cover
        raise AssertionError

    register.__module__ = "odd_plugin"
    entry_points["dfwb.plugins"].append(FakeEntryPoint("odd", register))
    (record,) = load_plugins().records
    assert record.status is PluginStatus.FAILED
    assert "invalid DFWB_PLUGIN_API" in record.reason


@pytest.mark.parametrize(
    ("spec", "ok"),
    [
        (">=1.0,<2", True),
        (">=1", True),
        ("<2", True),
        ("==1.0", True),
        (">=1.1", False),
        (">=2.0", False),
        ("!=1.0", False),
        (">1.0", False),
        ("<=1.0", True),
    ],
)
def test_api_compatible(spec, ok):
    assert api_compatible(spec) is ok


@pytest.mark.parametrize("spec", ["", "~=1.0", "1.0", ">=one"])
def test_api_compatible_rejects_garbage(spec):
    with pytest.raises(ValueError, match=r"specifier|cannot parse"):
        api_compatible(spec)


def test_env_switches_disable_plugins_but_not_builtins(entry_points, monkeypatch):
    builtin_calls = []
    entry_points["dfwb.builtins"].append(
        FakeEntryPoint(
            "dfwb",
            lambda api: builtin_calls.append(1),
            dist="deepfake-workbench",
            group="dfwb.builtins",
        )
    )
    entry_points["dfwb.plugins"] += [
        FakeEntryPoint("a", _register_stem, dist="dist-a"),
        FakeEntryPoint("b", lambda api: None, dist="dist-b"),
    ]
    monkeypatch.setenv("DFWB_PLUGINS", "none")
    report = load_plugins()
    assert [(r.name, r.status) for r in report.records] == [
        ("dfwb", PluginStatus.OK),
        ("a", PluginStatus.DISABLED),
        ("b", PluginStatus.DISABLED),
    ]
    assert report.records[0].provider == "dfwb"
    assert builtin_calls == [1]
    plugins.reset()
    monkeypatch.setenv("DFWB_PLUGINS", "")
    monkeypatch.setenv("DFWB_PLUGINS_DISABLE", "Dist_A, b-other")
    statuses = {r.name: r.status for r in load_plugins().records}
    assert statuses == {"dfwb": PluginStatus.OK, "a": PluginStatus.DISABLED, "b": PluginStatus.OK}


def test_catalogue_hint_for_uninstalled_well_known_key(entry_points):
    with pytest.raises(UnknownKeyError) as info:
        api.layers.entry("srm")
    assert info.value.hint == 'layers "srm" is provided by: pip install dfwb-torch-srm'


def test_catalogue_hint_when_provider_installed_but_failed(entry_points):
    def broken(api):
        raise ImportError("torch missing")

    entry_points["dfwb.plugins"].append(FakeEntryPoint("srm", broken, dist="dfwb_torch_srm"))
    with pytest.raises(UnknownKeyError) as info:
        api.layers.entry("srm")
    assert (
        "dfwb-torch-srm, which is installed but failed (ImportError: torch missing)"
        in info.value.hint
    )


def test_generic_hint_when_some_plugin_failed(entry_points):
    entry_points["dfwb.plugins"].append(FakeEntryPoint("x", RuntimeError("nope")))
    with pytest.raises(UnknownKeyError) as info:
        api.losses.entry("whatever")
    assert (
        info.value.hint
        == "1 plugin(s) failed to load and may provide it; run `dfwb plugins list --all`"
    )


def test_get_registry_and_entries_by_registry(entry_points):
    entry_points["dfwb.plugins"].append(FakeEntryPoint("good", _register_stem))
    assert plugins.get_registry("temporal-pools") is api.temporal_pools
    with pytest.raises(UnknownKeyError, match="did you mean 'layers'"):
        plugins.get_registry("layer")
    grouped = plugins.entries_by_registry()
    assert set(grouped) == set(plugins.REGISTRY_NAMES)
    assert [e.key for e in grouped["layers"]] == ["stem"]


def test_reset_clears_everything(entry_points):
    entry_points["dfwb.plugins"].append(FakeEntryPoint("good", _register_stem))
    load_plugins()
    plugins.reset()
    assert plugins._report is None
    entry_points["dfwb.plugins"].clear()
    assert api.layers.keys() == []


def test_api_shape():
    assert api.version == (1, 0)
    assert api.protocol_packs.kind == "data"
    assert api.eval_suites.kind == "data"
    assert api.layers.kind == "code"
    assert isinstance(metadata.entry_points, object)


def test_other_threads_wait_for_loading_instead_of_seeing_half_a_registry(entry_points):
    import threading

    started, release = threading.Event(), threading.Event()
    seen = {}

    def slow(api):
        api.layers.add("first", target="tests.unit.core._targets:Stem", summary="x")
        started.set()
        release.wait(5)
        api.layers.add("second", target="tests.unit.core._targets:Stem", summary="x")

    entry_points["dfwb.plugins"].append(FakeEntryPoint("slow", slow))
    loader = threading.Thread(target=load_plugins)
    loader.start()
    started.wait(5)
    reader = threading.Thread(target=lambda: seen.update(keys=api.layers.keys()))
    reader.start()
    reader.join(0.2)
    assert reader.is_alive()  # blocked on the loading lock, not reading a half-filled registry
    release.set()
    loader.join(5)
    reader.join(5)
    assert seen["keys"] == ["first", "second"]


def test_api_declared_in_the_entry_point_module_is_honoured(entry_points, monkeypatch, tmp_path):
    package = tmp_path / "reexport_plugin"
    package.mkdir()
    (package / "__init__.py").write_text(
        'DFWB_PLUGIN_API = ">=2"\nfrom reexport_plugin._hook import register\n'
    )
    (package / "_hook.py").write_text("def register(api):\n    raise AssertionError\n")
    monkeypatch.syspath_prepend(tmp_path)

    class ReexportEntryPoint(FakeEntryPoint):
        def load(self):
            import reexport_plugin

            return reexport_plugin.register

    ep = ReexportEntryPoint("reexport", None)
    ep.value = "reexport_plugin:register"
    entry_points["dfwb.plugins"].append(ep)
    (record,) = load_plugins().records
    assert record.status is PluginStatus.SKIPPED


def test_a_plugin_calling_sys_exit_is_isolated(entry_points):
    def exits(api):
        raise SystemExit(3)

    entry_points["dfwb.plugins"] += [
        FakeEntryPoint("exits", exits, dist="exits-dist"),
        FakeEntryPoint("exits-on-import", SystemExit(4), dist="exits-on-import-dist"),
        FakeEntryPoint("good", _register_stem, dist="good-dist"),
    ]
    statuses = {r.name: (r.status, r.reason) for r in load_plugins().records}
    assert statuses["exits"] == (PluginStatus.FAILED, "SystemExit: 3")
    assert statuses["exits-on-import"] == (PluginStatus.FAILED, "SystemExit: 4")
    assert statuses["good"] == (PluginStatus.OK, None)
    assert "stem" in api.layers


def test_a_distribution_without_a_name_does_not_break_loading(entry_points):
    ep = FakeEntryPoint("nameless", _register_stem, dist="placeholder")
    ep.dist = SimpleNamespace(name=None, version=None)  # METADATA without a Name field
    entry_points["dfwb.plugins"] += [ep, FakeEntryPoint("good", _register_stem, dist="good-dist")]
    records = {r.name: r for r in load_plugins().records}
    assert records["nameless"].status is PluginStatus.OK
    assert records["nameless"].provider == "fake-module-nameless"  # falls back to the module name
    assert records["good"].status is PluginStatus.OK
