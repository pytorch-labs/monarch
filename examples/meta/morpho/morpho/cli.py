# pyre-unsafe
import asyncio
import dataclasses
import importlib
import inspect
import itertools
import types
from dataclasses import fields, is_dataclass, MISSING

from typing import (
    Annotated,
    Any,
    Dict,
    get_args,
    get_origin,
    List,
    Literal,
    NamedTuple,
    Optional,
)

import click
import typer

from click.core import ParameterSource
from sphinx.ext.napoleon.docstring import GoogleDocstring


def get_dataclass_type(param_type):
    """Extract the dataclass type from a parameter type, handling Optional and X | None."""
    origin = get_origin(param_type)
    if origin is types.UnionType:
        args = get_args(param_type)
        if len(args) == 2 and is_dataclass(args[0]) and args[1] is type(None):
            return args[0]
    elif is_dataclass(param_type):
        return param_type
    return None


class DataclassPatch(NamedTuple):
    path: List[str]  # full path, e.g. [top_level_argument, my_subdataclass, my_subattr]
    argument_name: str  # the name we gave the value in the argument list

    def apply(self, context: typer.Context, kwargs):
        value = kwargs.pop(self.argument_name)
        if context.get_parameter_source(self.argument_name) is ParameterSource.DEFAULT:
            return
        root, *rest = self.path
        obj = kwargs[root]
        objects = [obj]
        for item in rest:
            objects.append(getattr(objects[-1], item))
        for parent, attr in zip(reversed(objects[:-1]), reversed(rest)):
            value = dataclasses.replace(parent, **{attr: value})
        kwargs[root] = value


class StringSpecifier(NamedTuple):
    argument_name: str
    argument_type: type
    option_name: str
    default: Any

    def apply(self, context: typer.Context, kwargs):
        if context.get_parameter_source(self.argument_name) is ParameterSource.DEFAULT:
            kwargs[self.argument_name] = self.default
            return
        specifier = kwargs[self.argument_name]
        try:
            module, name = specifier.rsplit(".", maxsplit=1)
            obj = getattr(importlib.import_module(module), name)
        except Exception as e:
            raise ValueError(
                f"--{self.option_name}='{specifier}' resulted in exception."
            ) from e
        if not isinstance(obj, self.argument_type):
            raise TypeError(
                f"{self.option_name}: expected an instance of {self.argument_type} but got {repr(obj)}"
            )
        kwargs[self.argument_name] = obj


def typer_supported(p: inspect.Parameter):
    try:
        typer.main.get_click_param(
            typer.models.ParamMeta(
                name=p.name, default=p.default, annotation=p.annotation
            )
        )
        return True
    except RuntimeError:
        return False


class Builder:
    def __init__(self, fn):
        self.parameters = []
        self.names = iter(itertools.count())

        self.specifiers: List[StringSpecifier] = []
        self.patches: List[DataclassPatch] = []
        self.signature: inspect.Signature = inspect.signature(fn)

        self.fn = fn
        dochelp = parse_docstring(fn.__doc__)
        for param in self.signature.parameters.values():
            self.add_parameter(param, help=dochelp.get(param.name))

    def fresh_param(self, typ, default):
        return inspect.Parameter(
            f"unused{next(self.names)}",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default,
            annotation=typ,
        )

    def add_dataclass(self, prefix: List[str], dataclass_type, default_dataclass):
        dataclass_fields = fields(dataclass_type)
        dochelp = parse_docstring(dataclass_type.__doc__)

        for field in dataclass_fields:
            if default_dataclass is not None:
                default = getattr(default_dataclass, field.name)
            elif field.default is not MISSING:
                default = field.default
            else:
                raise ValueError(
                    f"{prefix} has a None default but {prefix}.{field.name} has no default value."
                )
            path = [*prefix, field.name]

            path_str = ".".join((*prefix, field.name))
            if field.type is bool:
                field_specifier = f"--{path_str}/--no-{path_str}"
            else:
                field_specifier = "--" + path_str
            p = self.fresh_param(field.type, default)
            help = dochelp.get(field.name)
            self.add_leaf_parameter(p, option_name=field_specifier, help=help)
            self.patches.append(DataclassPatch(path, p.name))
            subdtype = get_dataclass_type(field.type)
            if subdtype is not None:
                self.add_dataclass(path, subdtype, default)

    @property
    def build_arguments_fn(self):
        specifiers = self.specifiers
        patches = self.patches

        def build_arguments(context: typer.Context, kwargs):
            for spec in specifiers:
                spec.apply(context, kwargs)
            for patch in patches:
                patch.apply(context, kwargs)

        return build_arguments

    def add_parameter(self, p: inspect.Parameter, help: Optional[str]):
        self.add_leaf_parameter(p, help=help)
        dtype = get_dataclass_type(p.annotation)
        if dtype is None:
            return
        if p.default is inspect.Parameter.empty:
            raise ValueError(
                f"{p.name} is a dataclasses so it must have a default value"
            )
        self.add_dataclass([p.name], dtype, p.default)

    def add_leaf_parameter(
        self,
        p: inspect.Parameter,
        option_name: Optional[str] = None,
        help: Optional[str] = None,
    ):
        args = (option_name,) if option_name is not None else ()
        kwargs = {}
        if get_origin(p.annotation) is Literal:
            values = get_args(p.annotation)
            for v in values:
                if not isinstance(v, str):
                    raise ValueError(
                        "Literal annotations can only contain string values"
                    )
            kwargs["click_type"] = click.Choice(values)
        elif not typer_supported(p):
            name = getattr(p.annotation, "__name__", p.annotation)
            help = f"A qualified name (e.g. my_module.my_value) that resolves to a {name}. {'' if help is None else help}"
            if option_name is None:
                option_name = p.name
            self.specifiers.append(
                StringSpecifier(p.name, p.annotation, option_name, p.default)
            )
            if p.default is inspect.Parameter.empty:
                p = p.replace(annotation=str)
            else:
                p = p.replace(annotation=Optional[str], default=None)

        if help is not None:
            kwargs["help"] = help

        if args or kwargs:
            ctor = (
                typer.Option
                if p.default is not inspect.Parameter.empty
                else typer.Argument
            )
            p = p.replace(annotation=Annotated[p.annotation, ctor(*args, **kwargs)])  # type: ignore
        self.parameters.append(p)

    @property
    def wrapped(self):
        build_arguments = self.build_arguments_fn
        orig = self.fn

        def wrapper(_context: typer.Context, **kwargs):
            build_arguments(_context, kwargs)
            result = orig(**kwargs)
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
            return result

        wrapper.__signature__ = self.signature.replace(
            parameters=[CONTEXT_PARAM, *self.parameters]
        )  # type: ignore
        return wrapper


CONTEXT_PARAM = inspect.Parameter(
    "_context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context
)


class DocstringParser(GoogleDocstring):
    def __init__(self, *args, **kwargs):
        self.fields = []
        super().__init__(*args, **kwargs)

    def _parse_parameters_section(self, section):
        self.fields.extend(self._consume_fields())
        return []

    _parse_attributes_section = _parse_parameters_section


def parse_docstring(docstring: str) -> Dict[str, str]:
    if docstring is None:
        return {}
    parser = DocstringParser(inspect.cleandoc(docstring))
    return {name: "\n".join(help) for name, _type, help in parser.fields}


def _wrap(fn):
    return Builder(fn).wrapped


def typer_app(fn):
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(_wrap(fn))
    return app


def cli(fn):
    """
    Create a cli automatically from a function.

    This is built on the typer project, but with a few extensions that come up
    a lot for training libraries.

    * Support for @dataclass objects as arguments. If a dataclass argument 'config' has a field 'nlayers', then
      We will automatically add `--config.nlayers` as an option to the command line, with approriate defaults for the
      data class.

    * Support for `Literal["a", "b", "c"]` annotation to document what options are possible for arguments without
      having to make an enum that was not already in the program.

    * Extracting documentation for options from the docstrings of functions and dataclasses.
    """
    typer_app(fn)()
