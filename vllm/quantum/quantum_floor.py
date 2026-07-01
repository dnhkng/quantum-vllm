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

from vllm.quantum.params import QuantumFloorParams, QuantumSeedParams

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


def quantum_lever_cache_key(params: QuantumLeverParams) -> tuple[object, ...]:
    return tuple(
        (name, getattr(params, name))
        for name in (
            "api_url",
            "api_key",
            "source",
            "personalization",
            "buffer_size",
            "recv_timeout_ms",
            "k",
            "require_full_vocab",
        )
        if hasattr(params, name)
    )


class QuantumLeverClient:
    def __init__(self, params: QuantumLeverParams):
        self._params = params
        self._bytes: deque[int] = deque()
        self._words_produced = 0

    def read_u32(self) -> int:
        while len(self._bytes) < 4:
            self._refill()

        value = 0
        for shift in range(0, 32, 8):
            value |= self._bytes.popleft() << shift
        word_index = self._words_produced
        self._words_produced += 1
        return quantum_personalize_word(
            value, getattr(self._params, "personalization", ""), word_index
        )

    def close(self) -> None:
        self._bytes.clear()

    def _refill(self) -> None:
        url = self._entropy_url()
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
            encoded = payload.get("payload_b64", payload.get("bytes_b64"))
            if not isinstance(encoded, str):
                raise TypeError("payload_b64 or bytes_b64 must be a string")
            decoded = base64.b64decode(encoded, validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "quantum_floor Quantum Lever response missing valid entropy payload"
            ) from exc

        if not decoded:
            raise RuntimeError(
                "quantum_floor Quantum Lever response contained no entropy bytes"
            )
        self._bytes.extend(decoded)

    def _entropy_url(self) -> str:
        parsed = urllib.parse.urlsplit(self._params.api_url)
        if parsed.path not in ("", "/"):
            return self._snapshot_url(parsed)

        source = getattr(self._params, "source", "qrng")
        if source == "lever":
            path = "/v1/lever/latest"
        elif source in ("", "qrng"):
            path = "/v1/qrng/latest"
        else:
            raise RuntimeError("quantum: --quantum-source must be 'qrng' or 'lever'")

        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
        )

    def _snapshot_url(self, parsed: urllib.parse.SplitResult) -> str:
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


def quantum_personalize_word(
    word: int, personalization: str, word_index: int
) -> int:
    if not personalization:
        return word & QUANTUM_FLOOR_MASK
    key = _quantum_personalization_key(personalization)
    block = _quantum_chacha20_block(key, word_index // 16)
    return (word ^ block[word_index % 16]) & QUANTUM_FLOOR_MASK


def _quantum_personalization_key(personalization: str) -> tuple[int, ...]:
    words: list[int] = []
    for i in range(4):
        h = _quantum_fnv1a64(
            personalization, 0x9E3779B97F4A7C15 * (i + 1)
        )
        words.append(h & QUANTUM_FLOOR_MASK)
        words.append((h >> 32) & QUANTUM_FLOOR_MASK)
    return tuple(words)


def _quantum_fnv1a64(text: str, seed: int) -> int:
    value = (0xCBF29CE484222325 ^ seed) & 0xFFFFFFFFFFFFFFFF
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def _quantum_chacha20_block(key: tuple[int, ...], counter: int) -> tuple[int, ...]:
    block = [
        0x61707865,
        0x3320646E,
        0x79622D32,
        0x6B206574,
        *key,
        counter & QUANTUM_FLOOR_MASK,
        0,
        0,
        0,
    ]
    out = block.copy()
    for _ in range(10):
        _quantum_chacha_quarter_round(out, 0, 4, 8, 12)
        _quantum_chacha_quarter_round(out, 1, 5, 9, 13)
        _quantum_chacha_quarter_round(out, 2, 6, 10, 14)
        _quantum_chacha_quarter_round(out, 3, 7, 11, 15)
        _quantum_chacha_quarter_round(out, 0, 5, 10, 15)
        _quantum_chacha_quarter_round(out, 1, 6, 11, 12)
        _quantum_chacha_quarter_round(out, 2, 7, 8, 13)
        _quantum_chacha_quarter_round(out, 3, 4, 9, 14)
    return tuple((out[i] + block[i]) & QUANTUM_FLOOR_MASK for i in range(16))


def _quantum_chacha_quarter_round(
    state: list[int], a: int, b: int, c: int, d: int
) -> None:
    state[a] = (state[a] + state[b]) & QUANTUM_FLOOR_MASK
    state[d] ^= state[a]
    state[d] = _quantum_rotl32(state[d], 16)
    state[c] = (state[c] + state[d]) & QUANTUM_FLOOR_MASK
    state[b] ^= state[c]
    state[b] = _quantum_rotl32(state[b], 12)
    state[a] = (state[a] + state[b]) & QUANTUM_FLOOR_MASK
    state[d] ^= state[a]
    state[d] = _quantum_rotl32(state[d], 8)
    state[c] = (state[c] + state[d]) & QUANTUM_FLOOR_MASK
    state[b] ^= state[c]
    state[b] = _quantum_rotl32(state[b], 7)


def _quantum_rotl32(value: int, shift: int) -> int:
    value &= QUANTUM_FLOOR_MASK
    return ((value << shift) | (value >> (32 - shift))) & QUANTUM_FLOOR_MASK
