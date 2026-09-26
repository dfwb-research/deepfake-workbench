"""The parts of a dataset's generated ``NOTICE.md`` that follow its ``distribution`` (C3a).

``dfwb protocols build`` writes them, and ``dfwb protocols lint`` looks for the ones written for
another distribution than the card's (a notice not rewritten since the distribution was decided):
list text in a recipe's notice, or recipe text in a list dataset's. Both read them from here, so
they cannot drift apart. A maintainer's ``terms.notes``, which the build appends to the terms
paragraph, is free text and plays no part in the check.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = ["holds_paragraph", "offers_lists", "offers_recipe", "terms_sentence"]

Distribution = Literal["undecided", "list", "recipe"]

# What a decided distribution lets the pack publish, as the notice words it.
_MEANING: Final = {
    "list": "these lists may be redistributed",
    "recipe": "the published pack carries no key list, only each split's rule and parameters "
    "and the hashes of the video, pair and split lists; dfwb protocols materialize rebuilds the "
    "lists from a local copy of the dataset and checks every hash",
}
_UNDECIDED: Final = (
    "Terms review pending: whether these lists may be redistributed has not been decided yet, so "
    "dataset.yaml records distribution: undecided. This notice is completed once the dataset's "
    "terms have been reviewed."
)
# What a dataset folder holds: the lists themselves, or (a recipe, once the release build has
# taken its key lists out) only what rebuilds and checks them.
_HOLDS_LISTS: Final = (
    "Never media: this folder lists video keys, labels, split assignments and fake/real pairs, "
    "derived from the release's file names and metadata. It holds no videos, frames, crops, face "
    "boxes, landmarks or anything else derived from pixels."
)
_HOLDS_RECIPE: Final = (
    "Never media, and in the published pack no key list either: this folder holds the dataset "
    "card (each split's rule and parameters, and the hashes of the video, pair and split lists), "
    "the label vocabulary and this notice. dfwb protocols materialize rebuilds the lists from a "
    "local copy of the dataset. Nothing here is derived from pixels: no videos, frames, crops, "
    "face boxes or landmarks."
)


def terms_sentence(distribution: Distribution) -> str:
    """The generated start of the notice's terms paragraph, before any ``terms.notes``."""
    if distribution == "undecided":
        return _UNDECIDED
    return (
        f"Terms reviewed: dataset.yaml records distribution: {distribution}, so "
        f"{_MEANING[distribution]}."
    )


def holds_paragraph(distribution: Distribution) -> str:
    """The notice's "What these files hold" paragraph."""
    return _HOLDS_RECIPE if distribution == "recipe" else _HOLDS_LISTS


def _flat(text: str) -> str:
    return " ".join(text.split())


def offers_lists(notice: str) -> bool:
    """Whether ``notice`` holds text generated for a dataset that ships its key lists.

    That is the "What these files hold" paragraph or the terms sentence written for
    ``distribution: list`` or ``undecided``, each matched whole (whitespace aside). Free text,
    such as a maintainer's ``terms.notes``, never matches, whatever it says about
    redistribution; nor does a notice edited by hand into other words.
    """
    return _holds_any(notice, (_HOLDS_LISTS, terms_sentence("list"), terms_sentence("undecided")))


def offers_recipe(notice: str) -> bool:
    """Whether ``notice`` holds text generated for a recipe, which ships no key list.

    Matched as :func:`offers_lists` matches: whole generated paragraphs, never free text.
    """
    return _holds_any(notice, (_HOLDS_RECIPE, terms_sentence("recipe")))


def _holds_any(notice: str, generated: tuple[str, ...]) -> bool:
    text = _flat(notice)
    return any(_flat(part) in text for part in generated)
