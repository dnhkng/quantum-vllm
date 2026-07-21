# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI helpers for Quantum Lever-backed sampling."""

from __future__ import annotations

from argparse import ArgumentParser

from vllm.quantum.params import QuantumDistParams


def add_quantum_cli_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--quantum-api-key",
        default=None,
        help="Use Quantum Lever entropy with this bearer API key.",
    )
    parser.add_argument(
        "--quantum-api-url",
        default=QuantumDistParams.api_url,
        help=(
            "Quantum Lever API base URL used with --quantum-api-key. "
            "Defaults to %(default)s."
        ),
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
        "--quantum-recv-timeout",
        type=int,
        default=QuantumDistParams.recv_timeout_ms,
        help=(
            "Quantum Lever receive timeout in milliseconds. "
            "Defaults to %(default)s."
        ),
    )
