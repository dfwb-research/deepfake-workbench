"""Inventory builders: from a raw dataset release to a local inventory (contract C3b).

:mod:`~dfwb.preprocess.inventory.base` holds the builder contract and the table-driven base class,
:mod:`~dfwb.preprocess.inventory.runner` builds, writes and reads inventories, and
:mod:`~dfwb.preprocess.inventory.builders` holds one builder per supported dataset. None of them
imports torch or numpy.
"""
