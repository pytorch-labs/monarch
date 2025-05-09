# pyre-strict
import argparse
import json
import pathlib
import sys

from fastcli.argparse import inject_fastcli
from monarch.tools.commands import component_args_from_cli, Config, create, info, kill
from monarch.tools.components import conda
from torchx.specs.finder import get_component

_DEFAULT_WORKSPACE = str(
    pathlib.Path.home() / "fbsource" / "genai" / "xlformers-branches" / "main_monarch"
)


class CreateCmd:
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-s",
            "--scheduler",
            type=str,
            default="mast_conda",
            help="Scheduler to submit to",
        )
        subparser.add_argument(
            "-cfg",
            "--scheduler_args",
            default=[],
            action="append",
            help="Scheduler args (e.g. `-cfg cluster=foo -cfg user=bar`)",
        )
        subparser.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Just prints the scheduler request",
        )
        subparser.add_argument(
            "--workspace",
            default=_DEFAULT_WORKSPACE,
            help="The local directory to fbpkg and make available on the job",
        )
        subparser.add_argument(
            "--component",
            help="A custom TorchX component to use (defaults to monarch.tools.components.conda.hyperactor)",
        )
        subparser.add_argument(
            "-arg",
            "--component_args",
            default=[],
            action="append",
            help="Arguments to the component fn (e.g. `-arg a=b -arg c=d` to pass as `component_fn(a=b, c=d)`)",
        )

    def run(self, args: argparse.Namespace) -> None:
        config = Config()
        config.apply_cli_args(args)

        component_fn = (
            get_component(args.component).fn if args.component else conda.hyperactor
        )
        component_args = component_args_from_cli(component_fn, args.component_args)
        handle = create(config, component_fn)(**component_args)
        print(handle)


class InfoCmd:
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "server_handle",
            type=str,
            help="monarch server handle (e.g. mast:///job_id)",
        )

    def run(self, args: argparse.Namespace) -> None:
        server_spec = info(args.server_handle)
        if server_spec is None:
            print(
                f"Server: {args.server_handle} does not exist",
                file=sys.stderr,
            )
        else:
            json.dump(server_spec.to_json(), fp=sys.stdout)


class KillCmd:
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "server_handle",
            type=str,
            help="monarch server handle (e.g. mast:///job_id)",
        )

    def run(self, args: argparse.Namespace) -> None:
        kill(args.server_handle)


def main(argv: list[str] = sys.argv[1:]) -> None:
    parser = argparse.ArgumentParser(description="Fallback Monarch Python CLI")
    subparser = parser.add_subparsers(title="COMMANDS")

    for cmd_name, cmd in {
        "create": CreateCmd(),
        "info": InfoCmd(),
        "kill": KillCmd(),
    }.items():
        cmd_parser = subparser.add_parser(cmd_name)
        cmd.add_arguments(cmd_parser)
        cmd_parser.set_defaults(func=cmd.run)

    # merges this (python) CLI args with rust's
    # see: https://fburl.com/wiki/kznp42c4
    inject_fastcli(parser)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
