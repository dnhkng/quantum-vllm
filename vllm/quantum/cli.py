# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI helpers for Quantum Lever-backed sampling."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from vllm.quantum.compat import resolve_quantum_params
from vllm.quantum.params import QuantumSeedParams
from vllm.quantum.quantum_floor import QuantumLeverClient
from vllm.quantum.validation import verify_quantum_floor_args, verify_quantum_seed_args


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
            "Quantum Lever API base URL used with --quantum-api-key. "
            "Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--quantum-source",
        choices=("qrng", "lever"),
        default=QuantumSeedParams.source,
        help="Quantum Lever entropy source. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--quantum-sampler",
        action="store_true",
        default=False,
        help="Enable the subscriber-only quantum_floor sampler.",
    )
    parser.add_argument(
        "--quantum-personalization",
        default="",
        help=(
            "Personalize Quantum Lever entropy locally with a non-secret "
            "ChaCha20 label."
        ),
    )
    parser.add_argument(
        "--quantum-k",
        type=int,
        default=64,
        help=(
            "quantum_floor minimum integer-CDF slots per token. "
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

    quantum_floor, quantum_seed = resolve_quantum_params(
        args, default_sampling_params=None
    )
    params = quantum_floor or quantum_seed
    if params is None:
        raise ValueError("Quantum Lever API check requires --quantum-api-key.")
    params.buffer_size = args.quantum_buffer_size
    if quantum_floor is not None:
        verify_quantum_floor_args(
            quantum_floor,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            ignore_eos=False,
            min_tokens=0,
            structured_outputs=None,
            allowed_token_ids=None,
            logit_bias=None,
            bad_words=None,
            sampling_eps=1e-5,
        )
    else:
        verify_quantum_seed_args(quantum_seed, seed=None, quantum_floor=None)

    client = QuantumLeverClient(params)
    try:
        word = client.read_u32()
    finally:
        client.close()

    print(f"Quantum Lever API check passed: read entropy word 0x{word:08x}")
    return True
