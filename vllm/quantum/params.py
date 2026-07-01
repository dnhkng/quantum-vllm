# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request parameters for Quantum Lever-backed sampling."""

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass
class QuantumFloorParams:
    """Parameters for Quantum Lever-backed quantum-floor sampling."""

    api_url: str = "https://quantumlever.stream/v1/entropy/snapshot"
    api_key: str = field(default="", repr=False)
    k: int = 64
    buffer_size: int = 256
    recv_timeout_ms: int = 2000
    require_full_vocab: bool = True
    debug_tax: bool = False
    debug_samples: bool = False
    log_path: str = ""


@dataclass
class QuantumSeedParams:
    """Parameters for Quantum Lever-backed per-request PRNG seeding."""

    api_url: str = "https://quantumlever.stream/v1/entropy/snapshot"
    api_key: str = field(default="", repr=False)
    buffer_size: int = 256
    recv_timeout_ms: int = 2000
