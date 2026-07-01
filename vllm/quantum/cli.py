# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI helpers for Quantum Lever-backed sampling."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from vllm.quantum.params import QuantumSeedParams
from vllm.quantum.quantum_floor import QuantumLeverClient
from vllm.quantum.validation import verify_quantum_seed_args


def add_quantum_cli_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--quantum-api-key",
        default=None,
        help=(
            "Fetch one Quantum Lever entropy word with this API key and exit. "
            "This smoke-tests the configured Quantum Lever endpoint."
        ),
    )
    parser.add_argument(
        "--quantum-api-url",
        default=QuantumSeedParams.api_url,
        help=(
            "Quantum Lever entropy snapshot URL used with --quantum-api-key. "
            "Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--quantum-buffer-size",
        type=int,
        default=QuantumSeedParams.buffer_size,
        help=(
            "Number of entropy bytes requested during the Quantum Lever CLI "
            "smoke test. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--quantum-recv-timeout",
        type=int,
        default=QuantumSeedParams.recv_timeout_ms,
        help=(
            "Quantum Lever receive timeout in milliseconds during the CLI smoke "
            "test. Defaults to %(default)s."
        ),
    )


def run_quantum_api_check_from_argv(argv: list[str]) -> bool:
    parser = ArgumentParser(
        prog="quantum-vllm",
        description="quantum-vLLM Quantum Lever API smoke test",
    )
    add_quantum_cli_args(parser)
    args = parser.parse_args(argv)
    return maybe_run_quantum_api_check(args)


def maybe_run_quantum_api_check(args: Namespace) -> bool:
    api_key = getattr(args, "quantum_api_key", None)
    if api_key is None:
        return False

    params = QuantumSeedParams(
        api_url=args.quantum_api_url,
        api_key=api_key,
        buffer_size=args.quantum_buffer_size,
        recv_timeout_ms=args.quantum_recv_timeout,
    )
    verify_quantum_seed_args(params, seed=None, quantum_floor=None)

    client = QuantumLeverClient(params)
    try:
        word = client.read_u32()
    finally:
        client.close()

    print(f"Quantum Lever API check passed: read entropy word 0x{word:08x}")
    return True
