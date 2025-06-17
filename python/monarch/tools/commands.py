# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import functools
import inspect
import os
from typing import Any, Callable, Mapping, Optional, Union

from monarch.tools.config import (  # @manual=//monarch/python/monarch/tools/config/meta:defaults
    Config,
    defaults,
)

from monarch.tools.mesh_spec import mesh_spec_from_metadata, ServerSpec

from torchx.runner import Runner
from torchx.specs import AppDef, AppDryRunInfo, CfgVal
from torchx.specs.builders import parse_args
from torchx.util.types import decode, decode_optional


def torchx_runner() -> Runner:
    # namespace is currently unused so make it empty str
    # so that server handle is short (e.g. slurm:///job-id)
    _EMPTY_NS = ""
    return Runner(_EMPTY_NS, defaults.scheduler_factories())


def component_args_from_cli(
    component_fn: Callable[..., AppDef], component_args: list[str]
) -> dict[str, Any]:
    """Parses component function's arguments from 'argname=argvalue' strings.

    Returns: component arguments kwarg-ified.
    """

    cli_fied_component_args = []
    for arg in component_args:
        argname = arg.split("=")[0]
        # torchx auto-generates an argparse parser for component function based
        # type-hints and docstring as if the component was a CLI itself so we have to
        # CLI arg-ify the component arguments by adding a "-" for
        # single-char argnames (short arg) and "--" for multi-char (long arg)
        cli_fied_component_args.append(f"-{arg}" if len(argname) == 1 else f"--{arg}")

    parsed_args: argparse.Namespace = parse_args(component_fn, cli_fied_component_args)

    # TODO kiuk@ logic below needs to move into torchx.specs.builders.parse_args()
    #  which is copied from torchx.specs.builders.materialize_appdef()
    #  parse_args() returns all the component parameters parsed from cli inputs
    #  as a string. Additional parameter type matching needs to be done (as below)
    #  to turn the CLI inputs to component function arguments.
    component_kwargs = {}

    parameters = inspect.signature(component_fn).parameters
    for param_name, parameter in parameters.items():
        arg_value = getattr(parsed_args, param_name)
        parameter_type = parameter.annotation
        parameter_type = decode_optional(parameter_type)
        arg_value = decode(arg_value, parameter_type)
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"component fn param `{param_name}` is a '*arg' which is not supported; consider changing the type to a list"
            )
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"component fn param `{param_name}` is a '**kwargs' which is not supported; consider changing the type to a dict or explicitly declare the params"
            )
        else:
            component_kwargs[param_name] = arg_value

    return component_kwargs


def create(
    config: Config,
    component_fn: Optional[Callable[..., AppDef]] = None,
) -> Callable[..., Union[str, AppDryRunInfo]]:
    """Creates a monarch server by submitting it as a job to the target scheduler.

    Note that this function returns a `Callable` that has to be called with the
    same arguments that one would call the `component_fn` to actually submit
    the job that runs the monarch server.

    Usage:

    .. doc-test::

        from monarch.tools.config import defaults

        config = defaults.config(scheduler="slurm")
        config.scheduler_args.update(
            {
                "partition": "prod",
                "mail-user": "foo@bar.com",
                "mail-type": "FAIL",
            }
        )
        config.dryrun = True

        create(default_config)(host_type="gpu.medium", num_hosts=4)


    Args:
        scheduler: where to submit a job that runs the server
        scheduler_args: scheduler configs
        component_fn: a function that returns the AppDef (job def).
            If not provided, defaults to the configured default for the scheduler
            (in most cases ``monarch.tools.components.hyperactor.proc_mesh``)
    """
    scheduler: str = config.scheduler
    cfg: Mapping[str, CfgVal] = config.scheduler_args
    component: Callable[..., AppDef] = component_fn or defaults.component_fn(scheduler)

    @functools.wraps(component)
    def _run(*args: Any, **kwargs: Any) -> Union[str, AppDryRunInfo]:
        # for logging call-site context in application metadata
        os.environ["TORCHX_CONTEXT_NAME"] = os.getenv("TORCHX_CONTEXT_NAME", "monarch")

        appdef = component(*args, **kwargs)

        with torchx_runner() as runner:
            info = runner.dryrun(appdef, scheduler, cfg, config.workspace)

            info_json_fmt = AppDryRunInfo(
                info.request,
                fmt=defaults.dryrun_info_formatter(info),
            )
            info_json_fmt._app = info._app
            info_json_fmt._cfg = info._cfg
            info_json_fmt._scheduler = info._scheduler

            if config.dryrun:
                return info_json_fmt
            else:
                server_handle = runner.schedule(info)
                return server_handle

    return _run


def info(server_handle: str) -> Optional[ServerSpec]:
    """Calls the ``describe`` API on the scheduler hosting the server to get
    information about it.

    Returns ``None`` if the server's job is not found in the scheduler's
    control-plane. This can happen if the job does not exist
    (e.g. typo in the server_handle) or the job already exited a long time ago.

    NOTE: This function can return non-empty info for jobs that have
    exited recently.
    """
    with torchx_runner() as runner:
        status = runner.status(server_handle)
        if status is None:
            return None

        appdef = runner.describe(server_handle)
        if appdef is None:
            return None

    mesh_specs = []
    for role in appdef.roles:
        spec = mesh_spec_from_metadata(appdef, role.name)
        assert spec is not None, "cannot be 'None' since we iterate over appdef's roles"
        mesh_specs.append(spec)

    return ServerSpec(name=appdef.name, state=status.state, meshes=mesh_specs)


def kill(server_handle: str) -> None:
    with torchx_runner() as runner:
        runner.cancel(server_handle)


def bounce(server_handle: str) -> None:
    """(re)starts the server's processes without tearing down the server's job."""
    raise NotImplementedError("`bounce` is not yet implemented")


def stop(server_handle: str) -> None:
    """Stops the server's unix processes without tearing down the server's job."""
    raise NotImplementedError("`stop` is not yet implemented")
