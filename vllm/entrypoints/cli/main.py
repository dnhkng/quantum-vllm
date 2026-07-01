# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_quantum_api_check(argv: list[str]) -> bool:
    return "--quantum-api-key" in argv


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ("-v", "--version"):
        print(importlib.metadata.version("vllm"))
        return

    if _is_quantum_api_check(sys.argv[1:]):
        from vllm.quantum.cli import run_quantum_api_check_from_argv

        run_quantum_api_check_from_argv(sys.argv[1:])
        return

    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.launch
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.quantum.cli import add_quantum_cli_args, maybe_run_quantum_api_check
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,
        vllm.entrypoints.cli.launch,
        vllm.entrypoints.cli.benchmark.main,
        vllm.entrypoints.cli.collect_env,
        vllm.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" in sys.argv:
        # NOTE: Check the spec instead of importing directly here, since things could
        # fail with ImportError due to mismatched versions if things are moved around.
        spec = find_spec("vllm_omni")
        if spec is None:
            logger.error(
                "--omni flag requires a valid instance of vllm-omni to be installed."
            )
            sys.exit(1)

        from vllm_omni.entrypoints.cli.main import main as omni_main

        logger.info("Delegating entrypoint handling to vllm-omni")
        omni_main()
    else:
        # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
        if len(sys.argv) > 1 and sys.argv[1] == "bench":
            logger.debug(
                "Bench command detected, must ensure current platform is not "
                "UnspecifiedPlatform to avoid device type inference error"
            )
            from vllm import platforms

            if platforms.current_platform.is_unspecified():
                from vllm.platforms.cpu import CpuPlatform

                platforms.current_platform = CpuPlatform()
                logger.info(
                    "Unspecified platform detected, switching to CPU Platform instead."
                )

        parser = FlexibleArgumentParser(
            description="quantum-vLLM CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm"),
        )
        add_quantum_cli_args(parser)
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if maybe_run_quantum_api_check(args):
            return

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
