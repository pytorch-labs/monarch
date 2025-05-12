# pyre-unsafe

import datetime
import getpass
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import torchx.components.fb.conda as conda
import torchx.components.fb.conda_transforms as conda_transforms
import torchx.components.fb.interactive_lib as interactive_lib
import torchx.specs as specs
from libfb.py import fbpkg
from libfb.py.fbpkg import BuildConfig

logger: logging.Logger = logging.getLogger(__name__)

# Default port name for the system port to be used in TW.
TW_SYSTEM_PORT_NAME = "system"
# Default named ports for TW.
MAST_DEFAULT_PORTS = {TW_SYSTEM_PORT_NAME: 29500}
# SMC bridge tier name for service discovery.
SMC_TIER_NAME_ENV = "MONARCH_SMC_SYSTEM_TIER_NAME"

_DEFAULT_ENV: Dict[str, str] = {
    "ENABLE_AIRSTORE": "0",
    "AIRSTORE_DECRYPT_SERVER_PATH": "/packages/ws_airstore.client/decrypt_server",
    "AIRSTORE_LOCAL_MOUNT_ROOT": "/mnt/airstore",
    "PRELOAD_PATH": "/packages/folly.symbolizer/libFollySegFault.so",
    "FUSE_DST": "/mnt/wsfuse",
    # default to the infra directory in ws
    "FUSE_SRC_PATH": "checkpoint/infra",
    "FUSE_ENABLE_OVERWRITES": "1",
    "OILFS_EXTRA_FLAGS_GENAI": "--oilfs_cto_periodic_refresh=15s",
}

_WITH_PROXY_ENV_VARS = {
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}

_HYPERACTOR_FBPKG = "monarch.hyperactor"
_RUN_SCRIPT = "/packages/conda_mast_core/run/torchx_run.sh"
_MOUNT_SCRIPT = "/packages/conda_mast_core/mount/mount.sh"
_HYPERACTOR_MAST_BOOTSTRAP = "/packages/" + _HYPERACTOR_FBPKG + "/main"
_MAX_NODES_IN_INTERACTIVE_MODE = 4
_ADDITIONAL_PACKAGES_FBPKG_NAME = "monarch_additional_packages"
# TorchX does not support enum decoding yet
_HYPERACTOR_BUILDS: Set[str] = {"STABLE", "LOCAL"}

_OILFS_MOUNT_DIR = os.path.join(_DEFAULT_ENV["FUSE_DST"], "aidev")


def train(
    *script_args: str,
    script: Optional[str] = None,
    module: Optional[str] = None,
    nodes: int = 1,
    nproc_per_node: int = 8,
    name: Optional[str] = None,
    h: str = "gtt_any",
    env: Optional[Dict[str, str]] = None,
    retry_policy: str = "APPLICATION",
    run_as_root: bool = True,
    dump_dir_id: str = "${app_id}",
    conda_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    additional_libraries: Optional[List[str]] = None,
    additional_folders: Optional[List[str]] = None,
    additional_python_paths: Optional[List[str]] = None,
    py_spy_startup: bool = False,
    retries: int = 2,
    tags: Optional[List[str]] = None,
    enable_ttls: bool = False,
    ports: Optional[Dict[str, int]] = MAST_DEFAULT_PORTS,
    advertised_port_name: Optional[str] = TW_SYSTEM_PORT_NAME,
    use_hyperactor: Optional[str] = None,
    world: str = "default",
    oilfs_workspace_dir: Optional[str] = None,
) -> specs.AppDef:
    """
    Kick off a training job on MAST.
    Sane defaults are specified in the .torchxconfig. Note that you must specify enableLegacy=True in
        your scheduler_args when running for smc discovery (arg advertised_port_name) to work. See:
        https://fb.workplace.com/groups/140700188041197/permalink/874386661339209/
    By defdault, the created job will advertise port TW_SYSTEM_PORT_NAME (29500) to SMC.

    Args:
        script_args: additional args to pass through to the script
        script: defaults to train.py, but you can run a different script
        module: if provided, run Python module instead of script
        nodes: total hosts to use
        nproc_per_node: processes per node
        name: custom name for this job
        h: hardware to use, eg. t1, tc_any, etc.
        env: custom environment parameters to pass through
        retry_policy: as title
        run_as_root: run the job as root; should be set to true for mounting
        dump_dir_id: Explicitly specify an mast job to continue training (defaults to new job id)
        conda_dir: an absolute path, or path relative to your homedir to the conda env on OILFS
        workspace_dir: absolute path where the workers will run
        additional_libraries: copy these folders into xlformers_pretrain2 and add them to python path
        additional_folders: copy these folders into the fbpkg xlformers_pretrain2
        additional_python_paths: add these paths to $PYTHONPATH before executing
        py_spy_startup: trace script startup; see tools/mast/py_spy_startup.sh for configuration
        retries: number of retries to attempt on the job
        tags: list of tags to add to the job
        enable_ttls: enable TTLS for the mast job, will allow internet access
        ports: Dict of port names and values to define as part of the job. Port value of 0
            automatically allocates ones. You can find more information about this here:
            https://fburl.com/wiki/icewdczt
        advertised_port_name: Port name from `ports` to advertise via smcBridge. See:
            https://www.internalfb.com/wiki/Infra_Cloud/Service_Hosting/Tupperware/Tupperware_Reference/SmcBridge/
        use_hyperactor: use the Rust-based hyperactor to run the job
        world: the world name to run in hyperactor
        oilfs_workspace_dir: path relative to /mnt/wsfuse/aidev/$USER where the workers will run
    """

    if name is None:
        # If a user does not specify job name, generate random name to avoid smc registration collision.
        # We don't want users using generic job names to override each others.
        name = f"monarch-{getpass.getuser()}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    if use_hyperactor:
        assert (
            use_hyperactor in _HYPERACTOR_BUILDS
        ), f"Invalid use_hyperactor: {use_hyperactor}"

    # Set up the environment variables
    mast_env = dict(_DEFAULT_ENV)
    username = getpass.getuser()

    assert (
        int(oilfs_workspace_dir is not None) + int(workspace_dir is not None) <= 1
    ), "Only one of oilfs_workspace_dir, and workspace_dir can be set."

    if oilfs_workspace_dir is not None:
        mast_env["MONARCH_OILFS_HOME_DIR"] = os.path.join(_OILFS_MOUNT_DIR, username)
        mast_env["WORKSPACE_DIR"] = os.path.join(
            mast_env["MONARCH_OILFS_HOME_DIR"], oilfs_workspace_dir
        )
    elif workspace_dir is not None:
        mast_env["WORKSPACE_DIR"] = workspace_dir

    if conda_dir:
        if oilfs_workspace_dir is not None:
            mast_env["CONDA_DIR"] = os.path.join(mast_env["WORKSPACE_DIR"], conda_dir)
        else:
            mast_env["CONDA_DIR"] = conda_dir

    dump_dir = Path(mast_env["FUSE_DST"]) / "outputs" / dump_dir_id
    # Make the dump dir available for shell scripts
    mast_env["DUMP_DIR"] = str(dump_dir)

    # Dependencies libraries for picking up latest site package
    additional_python_paths = additional_python_paths or []
    additional_folders = additional_folders or []
    additional_libraries = [
        *(additional_libraries or []),
        "../python",
    ]
    additional_pkgs = []

    if mast_env["ENABLE_AIRSTORE"]:
        additional_python_paths.append("/packages/ws_airstore.client/lib")

    if additional_libraries or additional_folders:
        additional_folders.extend(additional_libraries)
        additional_pkgs.append(_make_fbpkg(additional_folders))
        for folder in additional_libraries:
            additional_python_paths.append(
                f"/packages/{_ADDITIONAL_PACKAGES_FBPKG_NAME}/{os.path.basename(folder.rstrip('/'))}"
            )
    if use_hyperactor == "LOCAL":
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = f"{tmp_dir}/main"
            pkg = fbpkg.build_version(
                pkg_name=_HYPERACTOR_FBPKG,
                build_config=BuildConfig(
                    build_command="buck2 build -c fbcode.enable_gpu_sections=true "
                    f"@//mode/opt //monarch/hyperactor_meta:hyperactor_meta --out {output}",
                    paths=[output],
                ),
                ephemeral=True,
                expire="4w",
                silent_duplicate_error=True,
            )[0].identifier
            logger.info(f"Use hyperactor fbpkg {pkg}")
            additional_pkgs.append(pkg)
    elif use_hyperactor == "STABLE":
        # TODO: use stable version once we mark it in conveyor
        additional_pkgs.append(f"{_HYPERACTOR_FBPKG}")

    if enable_ttls:
        mast_env.update(_WITH_PROXY_ENV_VARS)
        mast_env["TTLS_ENABLED"] = "1"
        mast_env["PRELOAD_PATH"] = ":".join(
            (
                [mast_env["PRELOAD_PATH"]]
                if mast_env.get("PRELOAD_PATH", None) is not None
                else []
            )
            + [
                # TODO: Figure out why setting enable_ttls isn't enough to get TTLS
                # working properly.
                "/packages/ttls_so/TransparentTls3.so",
                # Adding TransparentTls3.so to the preload path will cause libzmq to
                # attempt to link against a version of libstdc++ that doesn't have the
                # right GLIBCXX version. So we need to force load the version of libstdc++
                # contained in the packaged conda env to ensure the right GLIBCXX version is
                # present.
                "/packages/torchx_base_conda_env/lib/libstdc++.so.6",
            ]
        )

    mast_env["TORCHX_RUN_PYTHONPATH"] = ":".join(additional_python_paths)

    if env:
        mast_env.update(env)

    script_cmd = (
        [
            _RUN_SCRIPT,
            f"-m{module}" if module else script,
            *script_args,
        ]
        if script is not None or module is not None
        else []
    )

    if use_hyperactor:
        logger.info(
            f"Launching job {name} with hyperactor",
        )
        # TODO: Get reply file done for system.
        cmd = [
            _HYPERACTOR_MAST_BOOTSTRAP,
            "--world",
            world,
            "--host-world",
            "host" + world,
            "--program",
            "/packages/monarch_workspace/run_monarch_worker.sh",
            "--num-procs-per-host",
            str(nproc_per_node),
            "--num-hosts",
            str(nodes),
        ]
        if ports is not None and TW_SYSTEM_PORT_NAME in ports:
            cmd.extend(
                [
                    "--system-port",
                    str(ports[TW_SYSTEM_PORT_NAME]),
                ]
            )
        if len(script_cmd) > 0:
            cmd.extend(["--main-script", *script_cmd])
    else:
        assert (
            len(script_cmd) > 0
        ), "Must specify either python script or python module when not using hyperactor"
        cmd = script_cmd

    job_spec = conda.run(
        *cmd,
        name=name,
        h=h,
        num_nodes=nodes,
        env=mast_env,
        retry_policy=retry_policy,
        run_as_root=run_as_root,
        enable_ttls=enable_ttls,
        max_retries=retries,
    )
    job_spec.roles[0].entrypoint = f"{_MOUNT_SCRIPT} && {job_spec.roles[0].entrypoint}"

    # append tb logdir as app metadata in the job spec
    job_spec = conda_transforms.append_tb_logdir_metadata(job_spec)

    if tags:
        job_spec.metadata["tags"] = ",".join(tags)

    packages = [
        "oil.oilfs:stable",
        "conda_mast_core:stable",
        "folly.symbolizer:prod",
        "fb-py-spy:prod",
        "scribe_cat:stable",
    ]
    if job_spec.roles[0].image:
        packages.append(job_spec.roles[0].image)
    if enable_ttls:
        packages.append("ttls_so:stable")

    if additional_pkgs:
        packages.extend(additional_pkgs)

    if mast_env["ENABLE_AIRSTORE"]:
        packages.append("ws_airstore.client:stable")

    job_spec.roles[0].image = ";".join(packages)

    if ports:
        for port_name in ports:
            job_spec.roles[0].port_map[port_name] = ports[port_name]
        if advertised_port_name is not None:
            if advertised_port_name not in ports:
                raise ValueError(
                    f"advertised_port_name {advertised_port_name} not found in ports"
                )
            smc_tier = f"mast.monarch.{name}-{advertised_port_name}"
            job_spec.roles[0].metadata = {
                "mast": {
                    "HpcTaskGroupSpec": {
                        "smcBridge": {
                            "portName": advertised_port_name,
                            "smcTier": smc_tier,
                        },
                    },
                },
            }
            job_spec.roles[0].env[SMC_TIER_NAME_ENV] = smc_tier

    return job_spec


def train_interactive(
    *script_args: str,
    script: str = "llama3/train.py",
    module: Optional[str] = None,
    nodes: int = 1,
    nproc_per_node: int = 8,
    name: Optional[str] = None,
    h: str = "gtt_any",
    env: Optional[Dict[str, str]] = None,
    retry_policy: str = "APPLICATION",
    run_as_root: bool = True,
    dump_dir_id: str = "${app_id}",
    conda_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    additional_libraries: Optional[List[str]] = None,
    additional_folders: Optional[List[str]] = None,
    additional_python_paths: Optional[List[str]] = None,
    py_spy_startup: bool = False,
    retries: int = 2,
    tags: Optional[List[str]] = None,
    enable_ttls: bool = False,
    ports: Optional[Dict[str, int]] = MAST_DEFAULT_PORTS,
    advertised_port_name: Optional[str] = TW_SYSTEM_PORT_NAME,
    sleep_hrs: int = 4,
    use_hyperactor: Optional[str] = None,
    world: str = "default",
    oilfs_workspace_dir: Optional[str] = None,
) -> specs.AppDef:
    """
    Minimal support for interactive workflows.

    Updates your trainer job to run on a single node, and then runs sleep on the job.

    This command will generate an `interactive.sh` file, available at
    $WORKSPACE_DIR which you can use to quickly run the command that
    would have been run automatically for you.

    It will also create a "sync.sh" file to copy over any changes you make back to your
    devserver if you're not using OILFS.

    Must be run in the repo root to function correctly.

    Args:
        script_args: additional args to pass through to the script
        script: defaults to train.py, but you can run a different script
        module: if provided, run Python module instead of script
        nodes: total hosts to use
        nproc_per_node: processes per node
        name: custom name for this job
        h: hardware to use, eg. t1, tc_any, etc.
        env: custom environment parameters to pass through
        retry_policy: as title
        run_as_root: run the job as root; should be set to true for mounting
        dump_dir_id: Explicitly specify an mast job to continue training (defaults to new job id)
        conda_dir: an absolute path, or path relative to your homedir to the conda env on OILFS
        workspace_dir: absolute path where the workers will run
        additional_libraries: copy these folders into xlformers_pretrain2 and add them to python path
        additional_folders: copy these folders into the fbpkg xlformers_pretrain2
        additional_python_paths: add these paths to $PYTHONPATH before executing
        py_spy_startup: trace script startup; see tools/mast/py_spy_startup.sh for configuration
        retries: number of retries to attempt on the job
        tags: list of tags to add to the job
        enable_ttls: enable TTLS for the mast job, will allow internet access
        ports: Dict of port names and values to define as part of the job. Port value of 0
            automatically allocates ones. You can find more information about this here:
            https://fburl.com/wiki/icewdczt
        advertised_port_name: Port name from `ports` to advertise via smcBridge. See:
            https://www.internalfb.com/wiki/Infra_Cloud/Service_Hosting/Tupperware/Tupperware_Reference/SmcBridge/
        sleep_hrs: how long to hold the host in hours; maxes out at 8
        oilfs_workspace_dir: path relative to /mnt/wsfuse/aidev/$USER where the workers will run
    """
    if nodes > _MAX_NODES_IN_INTERACTIVE_MODE:
        nodes = _MAX_NODES_IN_INTERACTIVE_MODE
        logger.warning(
            "The number of nodes is overridden to the maximum allowed in interactive mode, %d",
            _MAX_NODES_IN_INTERACTIVE_MODE,
        )

    if not run_as_root:
        logger.warning(
            "Interactive jobs must be run as root! Overriding run_as_root to True"
        )
        run_as_root = True

    if retries != 0:
        logger.warning(
            "Interactive jobs must be run with retries=0, overriding retries to 0"
        )
        retries = 0

    job_spec = train(
        *script_args,
        script=script,
        module=module,
        nodes=nodes,
        nproc_per_node=nproc_per_node,
        name=f"interactive-monarch-{name}"
        if name is not None
        else "interactive-monarch",
        h=h,
        env=env,
        retry_policy=retry_policy,
        run_as_root=run_as_root,
        dump_dir_id=dump_dir_id,
        conda_dir=conda_dir,
        workspace_dir=workspace_dir,
        additional_folders=additional_folders,
        additional_python_paths=additional_python_paths,
        additional_libraries=additional_libraries,
        py_spy_startup=py_spy_startup,
        retries=retries,
        tags=tags,
        enable_ttls=enable_ttls,
        ports=ports,
        advertised_port_name=advertised_port_name,
        use_hyperactor=use_hyperactor,
        world=world,
        oilfs_workspace_dir=oilfs_workspace_dir,
    )
    return interactive_lib.as_interactive(
        job_spec=job_spec,
        local_workspace_path=".",
        overlay_workspace_dir=oilfs_workspace_dir is None and workspace_dir is None,
        additional_dirs_to_overlay=None if conda_dir is not None else {"$CONDA_DIR"},
        interactive_duration_hrs=4,
        prerun_commands={
            "torchrun": f"( {_MOUNT_SCRIPT} || echo 'Unable to set up mounts! Please debug or escalate.' 1&>2 )"
        },
    )


def _make_fbpkg(paths: List[str]):
    """
    After llama4 or as bandwidth opens up we can try to move this into a single workspace fbpkg
    """
    from torchx.workspace.fb import fbpkg_utils

    return fbpkg_utils.build_fbpkg(
        fbpkg_name=_ADDITIONAL_PACKAGES_FBPKG_NAME,
        paths=paths,
        expiration="4w",
    )
