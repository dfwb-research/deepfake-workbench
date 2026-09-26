"""README.md is also the PyPI project description (`pyproject.toml`'s ``readme``).

PyPI renders that file on its own domain, with no knowledge of the GitHub repository it came
from: a link or an image source that is repository-relative resolves fine on GitHub but is a
dead link -- or a broken image -- on PyPI. Every link and image target must therefore be an
absolute URL, and an image specifically must point at raw file content (a GitHub ``blob`` URL
is an HTML page, not an image, and would not render as one). A same-page anchor (``#section``)
is the one exception: it is not repository-relative, and resolves correctly wherever the file
is rendered, GitHub or PyPI alike.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"

_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_IMAGE_ATTR = re.compile(r'(?:src|srcset)="([^"]+)"')


def _link_targets(text: str) -> list[str]:
    return _LINK.findall(text)


def _image_targets(text: str) -> list[str]:
    return _IMAGE_ATTR.findall(text)


def _is_absolute_or_anchor(target: str) -> bool:
    return target.startswith("#") or target.startswith("http://") or target.startswith("https://")


def test_every_readme_link_is_absolute_or_an_anchor():
    text = README.read_text("utf-8")
    offenders = [target for target in _link_targets(text) if not _is_absolute_or_anchor(target)]
    assert offenders == []


def test_every_readme_image_is_absolute_raw_content():
    text = README.read_text("utf-8")
    targets = _image_targets(text)
    assert targets  # the hero <picture> must still be there for this to check anything
    offenders = [
        target for target in targets if not target.startswith("https://raw.githubusercontent.com/")
    ]
    assert offenders == []


def test_the_check_catches_a_relative_link_and_a_relative_image():
    # Built at runtime, not written as a literal in this file, so this check's own source never
    # trips it.
    relative_link = "[install]" + "(docs/install.md)"
    relative_image = 'src="docs' + '/assets/hero-light.svg"'
    absolute_link = (
        "[install](https://github.com/dfwb-research/deepfake-workbench/blob/main/docs/install.md)"
    )
    absolute_image = (
        "https://raw.githubusercontent.com/dfwb-research/deepfake-workbench/main/"
        "docs/assets/hero-light.svg"
    )

    assert not _is_absolute_or_anchor(_link_targets(relative_link)[0])
    assert not _image_targets(relative_image)[0].startswith("https://raw.githubusercontent.com/")
    assert _is_absolute_or_anchor(_link_targets(absolute_link)[0])
    assert absolute_image.startswith("https://raw.githubusercontent.com/")
    assert _is_absolute_or_anchor("#quickstart")
