# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request parameters for Quantum Lever-backed sampling."""

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass
class QuantumFloorParams:
    """Parameters for Quantum Lever-backed quantum-floor sampling."""

    api_url: str = "https://quantumlever.stream"
    api_key: str = field(default="", repr=False)
    personalization: str = ""
    k: int = 64
    recv_timeout_ms: int = 2000


@dataclass
class QuantumDistParams:
    """Parameters for proportional sampling from Quantum Lever QRNG entropy."""

    api_url: str = "https://quantumlever.stream"
    api_key: str = field(default="", repr=False)
    personalization: str = ""
    recv_timeout_ms: int = 2000
