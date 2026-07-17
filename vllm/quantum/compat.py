# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility helpers for quantum-llama-server style options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.quantum.params import QuantumDistParams, QuantumFloorParams


@dataclass(frozen=True)
class QuantumFlatOptions:
    api_key: str | None = None
    api_url: str | None = None
    sampler: bool | None = None
    personalization: str | None = None
    k: int | None = None
    recv_timeout_ms: int | None = None


def quantum_defaults_from_args(args: Any) -> dict[str, Any]:
    values = _flat_options_from_obj(args)
    if not _activates_quantum(values):
        return {}
    return {
        "quantum_api_key": values.api_key or "",
        "quantum_api_url": values.api_url or QuantumDistParams.api_url,
        "quantum_sampler": bool(values.sampler),
        "quantum_personalization": values.personalization or "",
        "quantum_k": values.k if values.k is not None else QuantumFloorParams.k,
        "quantum_recv_timeout": values.recv_timeout_ms
        if values.recv_timeout_ms is not None
        else QuantumDistParams.recv_timeout_ms,
    }


def resolve_quantum_params(
    request: Any,
    default_sampling_params: dict[str, Any] | None,
) -> tuple[QuantumFloorParams | None, QuantumDistParams | None]:
    if (
        getattr(request, "quantum_floor", None) is not None
        or getattr(request, "quantum_dist", None) is not None
    ):
        return request.quantum_floor, request.quantum_dist

    defaults = default_sampling_params or {}
    values = _merge_flat_options(_flat_options_from_defaults(defaults), request)
    if not _activates_quantum(values):
        return None, None

    api_key = values.api_key or ""
    common = dict(
        api_url=values.api_url or QuantumDistParams.api_url,
        api_key=api_key,
        personalization=values.personalization or "",
        recv_timeout_ms=values.recv_timeout_ms
        if values.recv_timeout_ms is not None
        else QuantumDistParams.recv_timeout_ms,
    )
    if values.sampler:
        return (
            QuantumFloorParams(
                **common,
                k=values.k if values.k is not None else QuantumFloorParams.k,
            ),
            None,
        )
    return None, QuantumDistParams(**common)


def _activates_quantum(values: QuantumFlatOptions) -> bool:
    return bool(values.api_key) or bool(values.sampler)


def _flat_options_from_obj(obj: Any) -> QuantumFlatOptions:
    return QuantumFlatOptions(
        api_key=getattr(obj, "quantum_api_key", None),
        api_url=getattr(obj, "quantum_api_url", None),
        sampler=getattr(obj, "quantum_sampler", None),
        personalization=getattr(obj, "quantum_personalization", None),
        k=getattr(obj, "quantum_k", None),
        recv_timeout_ms=getattr(obj, "quantum_recv_timeout", None),
    )


def _flat_options_from_defaults(defaults: dict[str, Any]) -> QuantumFlatOptions:
    return QuantumFlatOptions(
        api_key=defaults.get("quantum_api_key"),
        api_url=defaults.get("quantum_api_url"),
        sampler=defaults.get("quantum_sampler"),
        personalization=defaults.get("quantum_personalization"),
        k=defaults.get("quantum_k"),
        recv_timeout_ms=defaults.get("quantum_recv_timeout"),
    )


def _merge_flat_options(
    defaults: QuantumFlatOptions,
    request: Any,
) -> QuantumFlatOptions:
    values = _flat_options_from_obj(request)

    def pick(name: str) -> Any:
        value = getattr(values, name)
        if value is not None:
            return value
        return getattr(defaults, name)

    return QuantumFlatOptions(
        api_key=pick("api_key"),
        api_url=pick("api_url"),
        sampler=pick("sampler"),
        personalization=pick("personalization"),
        k=pick("k"),
        recv_timeout_ms=pick("recv_timeout_ms"),
    )
