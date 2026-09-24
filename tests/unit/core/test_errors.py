import pickle
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from dfwb.core.errors import (
    AmbiguousKeyError,
    ConfigError,
    ContractError,
    CoverageError,
    DFWBError,
    InstallationError,
    PluginError,
    UnknownKeyError,
    did_you_mean,
    format_loc,
    model_fields_at,
    validation_messages,
    validation_problems,
)


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (DFWBError, 1),
        (PluginError, 1),
        (ConfigError, 2),
        (UnknownKeyError, 2),
        (AmbiguousKeyError, 2),
        (CoverageError, 3),
        (ContractError, 4),
        (InstallationError, 5),
    ],
)
def test_exit_codes(cls, code):
    err = cls("boom", hint="do this")
    assert err.exit_code == code
    assert str(err) == "boom"
    assert err.hint == "do this"
    assert isinstance(err, DFWBError)


def test_hint_is_required():
    with pytest.raises(TypeError):
        ConfigError("no hint")  # type: ignore[call-arg]


def test_errors_survive_pickling():
    err = pickle.loads(pickle.dumps(InstallationError("x", hint="pip install y")))
    assert type(err) is InstallationError
    assert (err.message, err.hint) == ("x", "pip install y")


def test_with_prefix_keeps_type_and_hint():
    err = ConfigError("bad", hint="h").with_prefix("model.backbone: ")
    assert type(err) is ConfigError
    assert err.message == "model.backbone: bad"
    assert err.hint == "h"


def test_did_you_mean():
    assert did_you_mean("partail", ["none", "full", "partial"]) == " (did you mean 'partial'?)"
    assert did_you_mean("zzz", ["none"]) == ""


def test_format_loc():
    assert format_loc(("data", "train", 0, "split")) == "data.train[0].split"
    assert format_loc(()) == ""


class _Freeze(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "full", "partial", "lora"]
    trainable_blocks: int = 0


def test_validation_messages_literal_extra_missing():
    with pytest.raises(ValidationError) as info:
        _Freeze.model_validate({"mode": "partail", "trainable_block": 2})
    lines = validation_messages(
        info.value,
        prefix=("model", "backbone", "freeze"),
        fields_at=lambda loc: ["mode", "trainable_blocks"],
    )
    assert lines == [
        "model.backbone.freeze.mode: 'partail' is not one of ['none', 'full', 'partial', 'lora'] "
        "(did you mean 'partial'?)",
        "model.backbone.freeze.trainable_block: unknown key (did you mean 'trainable_blocks'?)",
    ]
    with pytest.raises(ValidationError) as info:
        _Freeze.model_validate({})
    assert validation_messages(info.value) == ["mode: required key is missing"]


def test_config_error_problems_survive_pickling_and_prefixing():
    err = ConfigError("bad", hint="h", problems=[(("freeze", "mode"), "wrong")])
    again = pickle.loads(pickle.dumps(err))
    assert again.problems == ((("freeze", "mode"), "wrong"),)
    prefixed = err.with_prefix("x: ")
    assert (prefixed.message, prefixed.problems) == ("x: bad", err.problems)
    assert str(prefixed) == "x: bad"


class _Pack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]


def test_literal_of_non_strings_lists_the_choices():
    with pytest.raises(ValidationError) as info:
        _Pack.model_validate({"schema_version": 2})
    assert validation_messages(info.value) == ["schema_version: 2 is not one of [1]"]


def test_union_tags_are_dropped_from_paths():
    from pydantic import Discriminator, Tag

    class Suite(BaseModel):
        model_config = ConfigDict(extra="forbid")
        suite: str

    class Holder(BaseModel):
        test: Annotated[
            Annotated[Suite, Tag("@suite")] | Annotated[list[int], Tag("@list")],
            Discriminator(lambda v: "@suite" if isinstance(v, dict) else "@list"),
        ]

    with pytest.raises(ValidationError) as info:
        Holder.model_validate({"test": {"suite": "x", "suit": 1}})
    problems = validation_problems(info.value, fields_at=lambda loc: model_fields_at(Holder, loc))
    assert problems == [(("test", "suit"), "unknown key (did you mean 'suite'?)")]
