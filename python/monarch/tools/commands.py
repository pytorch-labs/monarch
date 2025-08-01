# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import asyncio
import inspect
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional, Union

from monarch.tools.components.hyperactor import DEFAULT_NAME

from monarch.tools.config import (  # @manual=//monarch/python/monarch/tools/config/meta:defaults
    Config,
    defaults,
)
from monarch.tools.mesh_spec import mesh_spec_from_metadata, ServerSpec
from torchx.runner import Runner  # @manual=//torchx/runner:lib_core
from torchx.specs import AppDef, AppDryRunInfo, AppState, CfgVal, parse_app_handle
from torchx.specs.builders import parse_args
from torchx.util.types import decode, decode_optional

logger: logging.Logger = logging.getLogger(__name__)


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
    name: str = DEFAULT_NAME,
) -> Union[str, AppDryRunInfo]:
    """Creates a monarch server by submitting it as a job to the target scheduler.

    Usage:

    .. doc-test::

        from monarch.tools.config import defaults

        config = defaults.config(scheduler="slurm")
        config.appdef = defaults.component_fn(scheduler=config.scheduler)()

        config.scheduler_args.update(
            {
                "partition": "prod",
                "mail-user": "foo@bar.com",
                "mail-type": "FAIL",
            }
        )
        config.dryrun = True

        create(config)


    Args:
        scheduler: where to submit a job that runs the server
        scheduler_args: scheduler configs
        component_fn: a function that returns the AppDef (job def).
            If not provided, defaults to the configured default for the scheduler
            (in most cases ``monarch.tools.components.hyperactor.proc_mesh``)
        name: the name of the job. If none, a default job name will be created.
    """
    scheduler: str = config.scheduler
    cfg: Mapping[str, CfgVal] = config.scheduler_args

    # for logging call-site context in application metadata
    os.environ["TORCHX_CONTEXT_NAME"] = os.getenv("TORCHX_CONTEXT_NAME", "monarch")

    with torchx_runner() as runner:
        appdef: AppDef = AppDef(name, config.appdef.roles, config.appdef.metadata)

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

    # host status grouped by mesh (role) names
    replica_status = {r.role: r.replicas for r in status.roles}

    mesh_specs = []
    for role in appdef.roles:
        spec = mesh_spec_from_metadata(appdef, role.name)
        assert spec is not None, "cannot be 'None' since we iterate over appdef's roles"

        # null-guard since some schedulers do not fill replica_status
        if host_status := replica_status.get(role.name):
            spec.hostnames = [h.hostname for h in host_status]
            # the mesh status is based on the "least progressive" replica status
            spec.state = min(h.state for h in host_status)

        mesh_specs.append(spec)

    scheduler, namespace, _ = parse_app_handle(server_handle)
    return ServerSpec(
        name=appdef.name,
        state=status.state,
        meshes=mesh_specs,
        scheduler=scheduler,
        namespace=namespace,
    )


_5_SECONDS = timedelta(seconds=5)


async def server_ready(
    server_handle: str,
    check_interval: timedelta = _5_SECONDS,
) -> Optional[ServerSpec]:
    """Waits until the server's job is in RUNNING state to returns the server spec.
    Returns `None` if the server does not exist.

    NOTE: Certain fields such as `hostnames` is only filled (and valid) when the server is RUNNING.

    Usage:

    .. code-block:: python

        server_info = await server_ready("slurm:///123")
        if not server_info:
            print(f"Job does not exist")
        else:
            if server_info.is_running:
                for mesh in server_info.meshes:
                    connect_to(mesh.hostnames)
            else:
                print(f"Job in {server_info.state} state. Hostnames are not available")

    """

    check_interval_seconds = check_interval.total_seconds()
    start = datetime.now()
    while True:
        server_spec = info(server_handle)

        if not server_spec:  # server not found
            return None

        if server_spec.state <= AppState.PENDING:  # UNSUBMITTED or SUBMITTED or PENDING
            # NOTE: TorchX currently does not have async APIs so need to loop-on-interval
            # TODO maybe inverse exponential backoff instead of constant interval?
            print(
                f"Waiting for {server_handle} to be {AppState.RUNNING} (current: {server_spec.state}); "
                f"will check again in {check_interval_seconds} seconds. "
                f"Total wait time: {datetime.now() - start}",
                end="\r",
            )
            await asyncio.sleep(check_interval_seconds)
            continue

        # check if hosts are allocated for all the meshes
        if server_spec.state == AppState.RUNNING:
            running = True
            for mesh_spec in server_spec.meshes:
                if mesh_spec.state <= AppState.PENDING:
                    print(
                        f"Job {server_handle} is running but waiting for mesh {mesh_spec.name} "
                        f"to be {AppState.RUNNING} (current: {mesh_spec.state}); "
                        f"will check again in {check_interval_seconds} seconds. "
                        f"Total wait time: {datetime.now() - start}",
                        end="\r",
                    )
                    running = False
                    break
            if not running:
                await asyncio.sleep(check_interval_seconds)
                continue

        return server_spec


# TODO: this API is overloaded. Ideally, we do not need config to get or an handle to create.
async def get_or_create(
    name: str,
    config: Config,
    check_interval: timedelta = _5_SECONDS,
) -> ServerSpec:
    """Waits for the server based on identity `name` in the scheduler specified in the `config`
    to be ready (e.g. RUNNING). If the server is not found then this function creates one
    per the `config` spec, and waits for the server to be ready before returning.

    Usage:

    .. code-block:: python

        from monarch.tools.config import defaults

        config = defaults.config(scheduler)
        config.appdef = defaults.component_fn(config.scheduler)()

        server_handle = get_or_create(name="my_job_name", config)
        server_info = info(server_handle)

    Returns: A `ServerSpec` containing information about either the existing or the newly
        created server.

    """
    assert not config.dryrun, "dryrun is not supported for get_or_create(), for dryrun use the create() API instead"

    server_handle = f"{config.scheduler}:///{name}"
    server_info = await server_ready(server_handle, check_interval)

    if not server_info or not server_info.is_running:  # then create one
        logger.info(
            "no existing RUNNING server `%s` creating new one...", server_handle
        )

        # no dryrun (see assertion above) support so will always be a handle (str)
        new_server_handle = str(create(config, name))

        logger.info(f"created new `{new_server_handle}` waiting for it to be ready...")

        server_info = await server_ready(new_server_handle, check_interval)

        if not server_info:
            raise RuntimeError(
                f"the new server `{new_server_handle}` went missing (should never happen)"
            )

        if not server_info.is_running:
            raise RuntimeError(
                f"the new server `{new_server_handle}` has {server_info.state}"
            )

        print(f"\x1b[36mNew job `{new_server_handle}` is ready to serve. \x1b[0m")
        return server_info
    else:
        print(f"\x1b[36mFound existing job `{server_handle}` ready to serve. \x1b[0m")
        return server_info


def kill(server_handle: str) -> None:
    with torchx_runner() as runner:
        runner.cancel(server_handle)


def bounce(server_handle: str) -> None:
    """(re)starts the server's processes without tearing down the server's job."""
    raise NotImplementedError("`bounce` is not yet implemented")


def stop(server_handle: str) -> None:
    """Stops the server's unix processes without tearing down the server's job."""
    raise NotImplementedError("`stop` is not yet implemented")
