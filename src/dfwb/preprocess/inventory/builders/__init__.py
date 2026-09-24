"""One inventory builder per supported dataset, each registered under its dataset id.

Builders are registered by import path in ``dfwb._builtins``, together with the folder each one
expects, so listing datasets never imports a builder module.
"""
