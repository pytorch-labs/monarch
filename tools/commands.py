# pyre-strict

import argparse
import functools
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from monarch.tools.components import conda
from monarch.tools.mesh_spec import mesh_spec_from_metadata, ServerSpec
from torchx.runner import Runner
from torchx.specs import AppDef, AppDryRunInfo, CfgVal
from torchx.specs.builders import parse_args
from torchx.util.types import decode, decode_optional

# [note on workspaces]
#
# In the context of the TorchX mast_conda scheduler plugin a "workspace" is defined
# as the local (on devserver) directory to snapshot (as an fbpkg) and download on the
# remote job (on MAST), making local code changes available to the remote job.
#
# Workspace building only happens once at job submission time, so any changes
# to the files in the workspace directory after the job has been launched is not
# automatically synchronized.
#
# Typically, the workspace directory is the project directory that contains the
# "client program" code base. In the case of xlformers it is one of:
#   * the locally cloned xlformers git repo
#   * ~/fbsource/genai/xlformers
#   * ~/fbsource/genai/xlformers-branches/{branch_name}
#
# Configuring workspaces is a bit of a mess since torchx/workspace/fb/conda_workspace.py
# has way too many configurations, some of which cancel each other out, or have
# different meanings when compounded.
#
# Many things in the Conda-on-Mast stack rely on the env var `WORKSPACE_DIR` being set.
# This env var is set by the mast_conda & conda_workspace TorchX plugins
# (owned by the Conda-on-Mast team).
#
# There are three ways for the user to trigger this env var to be set:
#   1. (DO NOT USE) Setting it directly on the `AppDef.Role.env["WORKSPACE_DIR"]`.
#   2. (Use when workspace dir is local) Passing the `--workspace` CLI option.
#      Equivalent to passing the `workspace` parameter when calling `monarch.tools.commands.create()` programmatically.
#   3. This builds the workspace directory into a patch fbpkg and makes it available on the job.
#      (DO NOT USE) Passing the `-cfg workspace_dir=/foo/bar` scheduler_arg CLI option.
#      Equivalent to setting `monarch.tools.commands.Config.scheduler_args.workspace_dir` programmatically.
#      This skips workspace building since the workspace dir is assumed to be available remotely.
#      Typically used in xlformers for workspaces that are NFS mounts.
#
# So we follow these rules in monarch:
#
#   1. The default workspace is `None` when using programmatically.
#      No local workspace directory is built and fbpkg'ed.
#      IMPORTANT: the CLI sets the workspace to $HOME/fbsource/genai/xlformers-branches/main_monarch!
#
#   2. The fbpkg that is built for this workspace defined in DEFAULT_WORKSPACE_FBPKG
#      To override:
#        monarch create -cfg workspace_fbpkg_name=foobar
#
#   3. For stuff that runs remotely, use `torchx.specs.macros.img_root` macro to get the path
#      to the workspace dir when authoring components or `$WORKSPACE_DIR` env var if you need
#      the workspace dir after the macro resolution in torchx (e.g.
#      see: genai/xlformers-branches/main_monarch/projects/monarch/scripts/run_monarch_bootstrap.sh)
#
#   4. `activate_conda=Fase` by default.
#      We don't rely on the CondaWorkspace torchx plugin to activate conda env and setup PYTHONPATH.
#      We do this manually in run_monarch_bootstrap.sh.
#
#   5. The default base conda env fbpkg is 'xlformers_llama4_conda:stable'.
#      To override:
#        monarch create -cfg conda_fbpkg_id=foobar:123
#
#      To make the fbpkg the currently active conda env (pass empty str):
#        monarch create -cfg conda_fbpkg_id=""
#
#   6. When workspace is disabled
#      (e.g. `--workspace=""` from the CLI or `create(...,workspace=None)` programmatically)
#      WORKSPACE_DIR defaults to `/packages/{workspace_fbpkg_name}`

DEFAULT_WORKSPACE_FBPKG = "xlformers_pretrain1"


# TODO kiuk@ maybe make this readable from a .monarchrc file?
#  something simple like monarch/examples/nanoGPT/config:NanoGPTConfig._configureCLIArgs
@dataclass
class Config:
    scheduler: str = "mast_conda"
    scheduler_args: dict[str, CfgVal] = field(  # pyre-ignore[8]
        default_factory=lambda: {
            # see [note on workspaces] above
            "workspace_fbpkg_name": DEFAULT_WORKSPACE_FBPKG,
            # --- mast configs ---
            "hpcClusterUuid": "MastGenAICluster",
            "hpcIdentity": "infra_research-llm",
            "hpcJobOncall": "meta_conda",
            "rmAttribution": "gen_ai_rf_nextgen_infra",
            "localityConstraints": ["region", "pci"],
            # --- conda configs ---
            "activate_conda": False,  # download but don't activate the env
            "conda_fbpkg_id": "xlformers_llama4_conda:stable",
            "conda_path_in_fbpkg": "conda",
            # --- non overridable args ---
            "enableLegacy": True,  # required for smc bridge metadata settings to work
        }
    )
    workspace: Optional[str] = None
    dryrun: bool = False

    def apply_cli_args(self, args: argparse.Namespace) -> None:
        if args.scheduler:
            self.scheduler = args.scheduler

        if args.scheduler_args:
            with _torchx_runner() as runner:
                opts = runner.scheduler_run_opts(self.scheduler)
                for cfg_str in args.scheduler_args:
                    parsed_cfg = opts.cfg_from_str(cfg_str)
                    assert (
                        "enableLegacy" not in parsed_cfg
                    ), "`enableLegacy` cannot be overridden!"

                    self.scheduler_args.update(parsed_cfg)

        if hasattr(args, "dryrun"):
            self.dryrun = args.dryrun

        if hasattr(args, "workspace") and args.workspace:
            self.workspace = args.workspace


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


def _torchx_runner() -> Runner:
    # TODO kiuk@ need lazy import, otherwise jetter's JustKnobs errors out in sandcastle (figure out why)
    from torchx.schedulers.fb import mast_conda_scheduler

    _SCHEDULERS = {
        "mast_conda": mast_conda_scheduler.create_scheduler,
    }
    # namespace is currently unused so make it empty str
    # so that server handle is short (e.g. mast_conda:///job-id)
    _EMPTY_NS = ""
    return Runner(_EMPTY_NS, _SCHEDULERS)  # pyre-ignore[6]


def create(
    config: Config,
    component_fn: Callable[..., AppDef] = conda.hyperactor,
) -> Callable[..., Union[str, AppDryRunInfo]]:
    """Creates a monarch server by submitting it as a job to the target scheduler.

    Note that this function returns a `Callable` that has to be called with the
    same arguments that one would call the `component_fn` to actually submit
    the job that runs the monarch server.

    Usage:

    .. code-block:: python

        config = Config()
        config.scheduler_args.update("hpcIdentity", "foobar")

        server_handle = create(default_config)(host_type="zionex_80g", num_hosts=4)


    Args:
        scheduler: where to submit a job that runs the server
        scheduler_args: scheduler configs
        component_fn: a function that returns the AppDef (job def).
            Defaults to `monarch.tools.components.conda.hyperactor`
    """
    scheduler: str = config.scheduler
    cfg: dict[str, CfgVal] = config.scheduler_args

    # TODO kiuk@ make torchx's dryrun normalize the scheduler request to json
    # and allow callers to attach a custom formatter that formats json
    # for now customize the formatter here (assumes mast/mast_conda)
    def _pretty_fmt_json(request) -> str:  # pyre-ignore[2]
        # TODO kiuk@ need lazy import, otherwise jetter's JustKnobs errors out in sandcastle (figure out why)
        from torchx.schedulers.fb import mast_scheduler

        return json.dumps(mast_scheduler.to_json(request), indent=2)

    @functools.wraps(component_fn)
    def _run(*args: Any, **kwargs: Any) -> Union[str, AppDryRunInfo]:
        # for logging call-site context in application metadata
        os.environ["TORCHX_CONTEXT_NAME"] = os.getenv("TORCHX_CONTEXT_NAME", "monarch")

        appdef = component_fn(*args, **kwargs)

        workspace = config.workspace
        if config.scheduler == "mast_conda" and not workspace:
            # see [note on workspaces] above
            # this is a defect in conda-on-mast's TorchX conda_workspace plugin
            # if workspace is disabled then we have to set it to a dummy and set the "interactive" metadata
            # so that conda_workspace is triggered and sets PYTHON_EXEC, CONDA_DIR, and WORKSPACE_DIR
            # TODO kiuk@ either fix this in in conda_workspace or abstract it out for OSS
            workspace = "__WORKSPACE_DISABLED__"
            for role in appdef.roles:
                role.metadata["interactive"] = "1"

        with _torchx_runner() as runner:
            info = runner.dryrun(appdef, scheduler, cfg, workspace)

            info_json_fmt = AppDryRunInfo(info.request, fmt=_pretty_fmt_json)
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
    with _torchx_runner() as runner:
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
    with _torchx_runner() as runner:
        runner.cancel(server_handle)
