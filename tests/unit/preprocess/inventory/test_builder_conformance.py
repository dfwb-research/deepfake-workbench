"""Every registered inventory builder meets the builder contract.

The parameter list is read from the registry at collection time, so a newly registered builder is
checked without touching this file. The same checks run on the demo builder too, so they are
exercised even when no builder is registered.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.preprocess.inventory._demo import install

from dfwb.core import plugins
from dfwb.preprocess.inventory.base import (
    COMPRESSION_TOKEN,
    SCHEME_RULES,
    BaseBuilder,
    InventoryBuilder,
)
from dfwb.preprocess.inventory.runner import get_builder

BUILDERS = plugins.get_registry("inventory_builders").keys()


def check_conformance(dataset_id: str, empty_dir: Path) -> None:
    builder = get_builder(dataset_id)
    assert isinstance(builder, BaseBuilder)
    assert isinstance(builder, InventoryBuilder)
    assert builder.dataset_id == dataset_id

    abbrs = [task.abbr for task in builder.tasks]
    assert abbrs, "a builder needs at least one task"
    assert len(abbrs) == len(set(abbrs)), f"task abbrs repeat: {abbrs}"
    assert set(builder.labels) == set(abbrs), "labels must cover exactly the tasks"
    if any(COMPRESSION_TOKEN in task.video_dir for task in builder.tasks):
        assert builder.known_compressions, "a {cX} task needs known_compressions"

    assert builder.default_scheme in builder.schemes
    for name, scheme in builder.schemes.items():
        assert scheme.rule in SCHEME_RULES, f"scheme {name!r} has rule {scheme.rule!r}"
    has_benchmark = any(s.rule == "benchmark" for s in builder.schemes.values())
    assert (builder.benchmark is not None) == has_benchmark

    assert builder.describe_layout().strip()
    assert list(builder.discover(empty_dir)) == []

    entry = plugins.get_registry("inventory_builders").entry(dataset_id)
    assert entry.meta.get("folder") == builder.expected_folder


@pytest.mark.parametrize("dataset_id", BUILDERS)
def test_registered_builder_conforms(dataset_id, tmp_path):
    check_conformance(dataset_id, tmp_path)


def test_the_checks_pass_for_the_demo_builder(monkeypatch, tmp_path):
    install(monkeypatch)
    check_conformance("demo", tmp_path)


def test_the_checks_catch_a_folder_that_disagrees_with_the_registry(monkeypatch, tmp_path):
    install(
        monkeypatch,
        {"demo": ("tests.unit.preprocess.inventory._demo:DemoBuilder", "Demo", "Elsewhere")},
    )
    with pytest.raises(AssertionError):
        check_conformance("demo", tmp_path)
