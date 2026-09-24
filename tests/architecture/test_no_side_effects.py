"""Importing dfwb (every module of it) changes nothing: no .env loading, no env vars, no files,
no logging handlers, no warning filters, no plugin loading and no optional-extra imports.

numpy is excluded from that last check: it is a base, always-installed dependency (unlike torch,
dotenv and rich, which are optional extras), and the layers that compute with it import it at
module level like any other required library."""

import json
import os
import subprocess
import sys

CODE = r"""
import importlib, json, logging, os, pkgutil, sys, warnings
import numpy  # noqa: F401 -- loaded first so its own import-time filter registration is not
              # mistaken for a side effect of importing dfwb (see the module docstring)
env = dict(os.environ)
handlers = list(logging.getLogger().handlers)
filters = list(warnings.filters)
files = sorted(os.listdir("."))
import dfwb
for info in pkgutil.walk_packages(dfwb.__path__, "dfwb."):
    importlib.import_module(info.name)
plugins = sys.modules.get("dfwb.core.plugins")
print(json.dumps({
    "env_changed": sorted(k for k in set(env) | set(os.environ) if env.get(k) != os.environ.get(k)),
    "root_handlers_changed": logging.getLogger().handlers != handlers,
    "warning_filters_changed": warnings.filters != filters,
    "files_changed": sorted(os.listdir(".")) != files,
    "plugins_loaded": bool(plugins is not None and plugins._report is not None),
    "heavy_modules": sorted(m for m in ("torch", "dotenv", "rich") if m in sys.modules),
}))
"""


def test_importing_every_module_has_no_side_effects(tmp_path):
    (tmp_path / ".env").write_text("DFWB_CANARY=1\nCUDA_VISIBLE_DEVICES=7\n")
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    done = subprocess.run(
        [sys.executable, "-c", CODE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout) == {
        "env_changed": [],
        "root_handlers_changed": False,
        "warning_filters_changed": False,
        "files_changed": False,
        "plugins_loaded": False,
        "heavy_modules": [],
    }
    assert list(home.iterdir()) == []


def test_package_init_only_exposes_the_version():
    code = "import dfwb; print(sorted(n for n in vars(dfwb) if not n.startswith('__')))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "['_version']"


_ENVFILE_CODE = r"""
import json, os
env = dict(os.environ)
import dfwb.cli.main
import dfwb.core.envfile
print(json.dumps(sorted(k for k in set(env) | set(os.environ) if env.get(k) != os.environ.get(k))))
"""


def test_importing_the_cli_and_envfile_never_loads_a_dotenv(tmp_path):
    """Importing ``dfwb.cli.main`` and ``dfwb.core.envfile`` must never apply a ``.env``:
    loading only happens when the CLI group callback runs, never at import."""
    (tmp_path / ".env").write_text("DFWB_CANARY=1\n")
    done = subprocess.run(
        [sys.executable, "-c", _ENVFILE_CODE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout) == []
