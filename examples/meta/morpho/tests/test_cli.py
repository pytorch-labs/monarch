from dataclasses import dataclass
from typing import Any, List, Literal, Tuple

import typer
from morpho.cli import _wrap, DataclassPatch
from typer.testing import CliRunner

runner = CliRunner()


class MockClickContext:
    def get_parameter_source(self, name: str):
        return None


def apply_patches(obj, patches: List[Tuple[str, Any]]):
    ctx = MockClickContext()
    kwargs = {"base": obj}
    for name, value in patches:
        path = ["base", *name.split(".")]
        kwargs["the_value"] = value
        DataclassPatch(path, "the_value").apply(ctx, kwargs)  # type: ignore
        assert "the_value" not in kwargs
    return kwargs["base"]


@dataclass
class ExampleDataClass2:
    b: bool = False


@dataclass
class ExampleDataClass:
    """
    Attributes:
        x: the x
        y: the y
        the_rest: some more
    """

    x: int
    y: float
    z: str = "what?"
    a: ExampleDataClass2 | None = None
    the_rest: Literal["a", "c", "d"] = "a"


_default_example_data_class = ExampleDataClass(3, 4.0)


def foo(
    what: Literal["a", "b", "c"],
    b: int,
    c: bool = True,
    a: ExampleDataClass = _default_example_data_class,
):
    """
    Args:
        c: a flag
        b: an int
    """
    print(what, b, a, c)


def invoke(fn, *args):
    app = typer.Typer()
    app.command()(_wrap(fn))
    return runner.invoke(app, args, catch_exceptions=False)


def test_correct():
    result = invoke(foo, "a", "3")

    assert result.exit_code == 0
    assert "a 3" in result.stdout
    assert "x=3" in result.stdout
    assert "y=4.0" in result.stdout
    assert "a=None" in result.stdout


def test_help():
    result = invoke(foo, "--help")
    assert result.exit_code == 0
    assert "--a.a.b" in result.stdout
    assert "--no-a.a.b" in result.stdout
    assert "the x [default: 3]" in result.stdout
    assert "a flag [default: c]" in result.stdout


# Sample dataclasses for testing
@dataclass
class Inner:
    value: int


@dataclass
class Outer:
    inner: Inner
    name: str


# Test function
def test_apply_patches():
    # Create an instance of the dataclass
    original = Outer(inner=Inner(value=10), name="Original")
    # Define patches
    patches = [("inner.value", 20), ("name", "Updated")]
    # Apply patches
    updated = apply_patches(original, patches)
    # Assertions to verify the patches were applied correctly
    assert updated.inner.value == 20
    assert updated.name == "Updated"
    assert original.inner.value == 10  # Ensure original is unchanged
    assert original.name == "Original"  # Ensure original is unchanged
    # Test patching order
    patches_order = [
        ("inner.value", 30),
        ("inner", Inner(value=40)),  # This should replace the entire inner object
        ("name", "Final"),
    ]
    # Apply patches in order
    updated_order = apply_patches(original, patches_order)
    # Assertions to verify the patches were applied in order
    assert updated_order.inner.value == 40  # The entire inner object is replaced
    assert updated_order.name == "Final"
    assert original.inner.value == 10  # Ensure original is unchanged
    assert original.name == "Original"  # Ensure original is unchanged
