"""A CPU-time budget that a busy shared machine cannot fail.

The budget tests time process CPU time (``time.process_time()``), not wall-clock, so another
process's work is not counted. CPU time still grows when the machine is oversubscribed, though:
the caches, memory bandwidth and clock speed are shared, so the same code takes more CPU time
while every core is busy. So the budget is asserted only when the 1-minute load average is below
the CPU count; on a busier machine the test is skipped, with the measured value in the reason,
rather than failed for a cost that is not the code's own.
"""

from __future__ import annotations

import os

import pytest

__all__ = ["assert_cpu_budget"]


def assert_cpu_budget(elapsed: float, budget: float, what: str) -> None:
    """Assert ``elapsed`` seconds of CPU time are under ``budget``, when the machine is quiet
    enough to measure it (see the module docstring); skip otherwise."""
    load = os.getloadavg()[0]
    cpus = os.cpu_count() or 1
    if load >= cpus:
        pytest.skip(
            f"{what}: {elapsed:.3f}s of CPU time measured, budget {budget:.1f}s not asserted: "
            f"the 1-minute load average {load:.1f} is not below the {cpus} CPU(s)"
        )
    assert elapsed < budget, f"{what} took {elapsed:.3f}s of CPU time (budget {budget:.1f}s)"
