# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from vllm.sampling_params import QuantumFloorParams, QuantumSeedParams

QUANTUM_FLOOR_M = 1 << 32
QUANTUM_FLOOR_MASK = QUANTUM_FLOOR_M - 1
QUANTUM_FLOOR_K_MAX = 1 << 20


class QRNGSource(Protocol):
    def read_u32(self) -> int: ...


@dataclass(frozen=True)
class QuantumFloorAllocation:
    slots: np.ndarray
    cdf: np.ndarray


def compute_g_rows() -> tuple[int, ...]:
    rows: list[int] = []
    for i in range(32):
        row = 0
        for j in range(32):
            i_ij = int(i == j)
            h_ij = ((i & j).bit_count()) & 1
            g_ij = i_ij ^ h_ij ^ 1
            if g_ij:
                row |= 1 << j
        rows.append(row)
    return tuple(rows)


@lru_cache(maxsize=1)
def default_g_rows() -> tuple[int, ...]:
    return compute_g_rows()


def spread_u32(value: int, rows: tuple[int, ...] | None = None) -> int:
    if rows is None:
        rows = default_g_rows()
    if len(rows) != 32:
        raise ValueError("quantum_floor spreader requires 32 row masks")

    value &= QUANTUM_FLOOR_MASK
    spread = 0
    for i, row in enumerate(rows):
        if ((value & row).bit_count() & 1) != 0:
            spread |= 1 << i
    return spread


def build_allocation(probs: np.ndarray, k: int) -> QuantumFloorAllocation:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("quantum_floor requires a non-empty 1D probability vector")
    if k < 1:
        raise ValueError("quantum_floor k must be >= 1")
    if k > QUANTUM_FLOOR_K_MAX:
        raise ValueError("quantum_floor k must be <= 2^20")
    if k * probs.size > QUANTUM_FLOOR_M:
        raise ValueError("quantum_floor k is too large for vocabulary size")

    clean = np.where(np.isfinite(probs) & (probs > 0.0), probs, 0.0)
    slots = np.floor(clean * float(QUANTUM_FLOOR_M)).astype(np.uint64)
    floored = slots < k
    total = int(slots.sum(dtype=np.uint64))

    if np.any(floored):
        total += int((np.uint64(k) - slots[floored]).sum(dtype=np.uint64))
        slots[floored] = k

    resolved = ~floored
    resolved_total = int(slots[resolved].sum(dtype=np.uint64))
    if resolved_total == 0 and total != QUANTUM_FLOOR_M:
        raise ValueError("quantum_floor has no resolved tokens for compensation")

    delta = total - QUANTUM_FLOOR_M
    if delta != 0:
        adjusted = slots.copy()
        resolved_indices = np.flatnonzero(resolved)
        for idx in resolved_indices:
            share = abs(delta) * int(slots[idx]) / resolved_total
            adj = int(math.floor(share + 0.5))
            if delta > 0:
                adjusted[idx] = max(0, int(adjusted[idx]) - adj)
            else:
                adjusted[idx] = int(adjusted[idx]) + adj
        slots = adjusted
        total = int(slots.sum(dtype=np.uint64))

    resolved_indices = np.flatnonzero(resolved)
    if resolved_indices.size == 0:
        if total != QUANTUM_FLOOR_M:
            raise ValueError("quantum_floor unable to correct slot residual")
    else:
        largest_resolved = int(resolved_indices[np.argmax(slots[resolved_indices])])
        while total != QUANTUM_FLOOR_M:
            if total > QUANTUM_FLOOR_M:
                if slots[largest_resolved] == 0:
                    raise ValueError("quantum_floor slot residual underflow")
                slots[largest_resolved] -= 1
                total -= 1
            else:
                slots[largest_resolved] += 1
                total += 1

    cdf = np.cumsum(slots, dtype=np.uint64)
    cdf[-1] = QUANTUM_FLOOR_MASK
    return QuantumFloorAllocation(slots=slots, cdf=cdf)


def select_token_index(probs: np.ndarray, raw_u32: int, k: int) -> int:
    alloc = build_allocation(probs, k)
    spread = spread_u32(raw_u32)
    return int(np.searchsorted(alloc.cdf, spread, side="right"))


QuantumLeverParams = QuantumFloorParams | QuantumSeedParams


class QuantumLeverClient:
    def __init__(self, params: QuantumLeverParams):
        self._params = params
        self._bytes: deque[int] = deque()

    def read_u32(self) -> int:
        while len(self._bytes) < 4:
            self._refill()

        value = 0
        for shift in range(0, 32, 8):
            value |= self._bytes.popleft() << shift
        return value

    def close(self) -> None:
        self._bytes.clear()

    def _refill(self) -> None:
        url = self._snapshot_url()
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._params.api_key}",
                "User-Agent": "vllm quantum_floor",
            },
        )
        timeout = self._params.recv_timeout_ms / 1000.0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "quantum_floor Quantum Lever entropy snapshot "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "quantum_floor failed to fetch Quantum Lever entropy snapshot"
            ) from exc

        try:
            payload = json.loads(body)
            encoded = payload["bytes_b64"]
            if not isinstance(encoded, str):
                raise TypeError("bytes_b64 must be a string")
            decoded = base64.b64decode(encoded, validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "quantum_floor Quantum Lever response missing valid bytes_b64"
            ) from exc

        if not decoded:
            raise RuntimeError(
                "quantum_floor Quantum Lever response contained no entropy bytes"
            )
        self._bytes.extend(decoded)

    def _snapshot_url(self) -> str:
        parsed = urllib.parse.urlsplit(self._params.api_url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query.append(("bytes", str(max(4, self._params.buffer_size))))
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urllib.parse.urlencode(query),
                parsed.fragment,
            )
        )
