"""``tests._cpu_budget``: a CPU-time budget is asserted only when the machine is not busy."""

from __future__ import annotations

import os

import pytest
from tests._cpu_budget import assert_cpu_budget


def _machine(monkeypatch, *, load: float, cpus: int) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (load, load, load))
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)


def test_a_quiet_machine_asserts_the_budget(monkeypatch):
    _machine(monkeypatch, load=3.0, cpus=8)
    assert_cpu_budget(0.5, 1.0, "the read")
    with pytest.raises(AssertionError, match=r"the read took 1\.250s of CPU time"):
        assert_cpu_budget(1.25, 1.0, "the read")


def test_a_busy_machine_skips_with_the_measured_value(monkeypatch):
    _machine(monkeypatch, load=8.0, cpus=8)  # the load average is not below the CPU count
    with pytest.raises(pytest.skip.Exception) as excinfo:
        assert_cpu_budget(1.25, 1.0, "the read")
    reason = str(excinfo.value)
    assert "1.250s" in reason
    assert "8.0" in reason
    assert "8 CPU" in reason
