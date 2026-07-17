# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from vllm.logger import init_logger
from vllm.quantum.params import QuantumDistParams, QuantumFloorParams

QUANTUM_FLOOR_M = 1 << 32
QUANTUM_FLOOR_MASK = QUANTUM_FLOOR_M - 1
QUANTUM_FLOOR_K_MAX = 1 << 20
QUANTUM_BIAS_WINDOW_BITS = 65536

logger = init_logger(__name__)


class QRNGSource(Protocol):
    def read_u32(self) -> int: ...


@dataclass(frozen=True)
class QuantumFloorAllocation:
    slots: np.ndarray
    cdf: np.ndarray


@dataclass(frozen=True)
class QuantumDistAllocation:
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
    index = int(np.searchsorted(alloc.cdf, spread, side="right"))
    return min(index, alloc.cdf.size - 1)


def build_dist_allocation(probs: np.ndarray) -> QuantumDistAllocation:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("quantum_dist requires a non-empty 1D probability vector")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("quantum_dist received invalid probability mass")

    total_prob = math.fsum(float(prob) for prob in probs)
    if not math.isfinite(total_prob) or total_prob <= 0.0:
        raise ValueError("quantum_dist received invalid probability mass")

    quotas = probs * (float(QUANTUM_FLOOR_M) / total_prob)
    slots = np.floor(quotas).astype(np.uint64)
    residual = QUANTUM_FLOOR_M - int(slots.sum(dtype=np.uint64))
    if residual < 0 or residual > probs.size:
        raise ValueError("quantum_dist produced an invalid slot residual")

    if residual:
        remainders = quotas - slots
        cutoff = np.partition(remainders, probs.size - residual)[probs.size - residual]
        winners = np.flatnonzero(remainders > cutoff)
        slots[winners] += 1
        ties_needed = residual - winners.size
        if ties_needed:
            # Hamilton apportionment assigns tied residuals by token order.
            ties = np.flatnonzero(remainders == cutoff)[:ties_needed]
            slots[ties] += 1

    cdf = np.cumsum(slots, dtype=np.uint64)
    if int(cdf[-1]) != QUANTUM_FLOOR_M:
        raise ValueError("quantum_dist slot allocation did not fill address space")
    return QuantumDistAllocation(slots=slots, cdf=cdf)


def select_dist_token_index(probs: np.ndarray, raw_u32: int) -> int:
    alloc = build_dist_allocation(probs)
    return int(np.searchsorted(alloc.cdf, raw_u32 & QUANTUM_FLOOR_MASK, side="right"))


QuantumLeverParams = QuantumFloorParams | QuantumDistParams
QuantumMode = Literal["dist", "floor"]


class QuantumLeverClient:
    def __init__(self, params: QuantumLeverParams, mode: QuantumMode):
        self._params = params
        self._mode = mode
        self._bytes: deque[int] = deque()
        self._words_produced = 0
        self._bytes_received = 0
        self._last_payload_hash = ""
        self._next_fetch_time = 0.0
        self._bits: deque[int] = deque()
        self._bit_ones = 0
        self._last_diag_time = time.monotonic()
        self._last_bias_warning = float("-inf")

    def start(self) -> None:
        self._check_key()
        self._refill()
        if self._mode == "floor":
            self._bytes.clear()
            while not self._bytes:
                self._wait_for_next_fetch()
                self._refill()

    def read_u32(self) -> int:
        while len(self._bytes) < 4:
            before = len(self._bytes)
            self._refill()
            if len(self._bytes) == before:
                self._wait_for_next_fetch()

        value = 0
        for shift in range(0, 32, 8):
            value |= self._bytes.popleft() << shift
        word_index = self._words_produced
        self._words_produced += 1
        self._log_diagnostics_if_due()
        return quantum_personalize_word(
            value, getattr(self._params, "personalization", ""), word_index
        )

    def close(self) -> None:
        self._bytes.clear()

    def _refill(self) -> None:
        url = self._api_url(
            "/v1/lever/latest" if self._mode == "floor" else "/v1/qrng/latest"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._params.api_key}",
                "User-Agent": f"vllm quantum_{self._mode}",
            },
        )
        timeout = self._params.recv_timeout_ms / 1000.0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"quantum_{self._mode} Quantum Lever entropy snapshot "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"quantum_{self._mode} failed to fetch Quantum Lever entropy snapshot"
            ) from exc

        try:
            payload = json.loads(body)
            encoded = payload.get("payload_b64")
            if not isinstance(encoded, str):
                raise TypeError("payload_b64 must be a string")
            decoded = base64.b64decode(encoded, validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"quantum_{self._mode} Quantum Lever response missing valid "
                "entropy payload"
            ) from exc

        if not decoded:
            raise RuntimeError(
                f"quantum_{self._mode} Quantum Lever response contained no "
                "entropy bytes"
            )
        payload_hash = payload.get("payload_hash", "")
        if self._mode == "floor" and not payload_hash:
            raise RuntimeError(
                "quantum_floor Quantum Lever response missing payload_hash"
            )
        self._next_fetch_time = self._compute_next_fetch_time(payload)
        if payload_hash and payload_hash == self._last_payload_hash:
            return
        self._last_payload_hash = payload_hash
        self._bytes.extend(decoded)
        self._bytes_received += len(decoded)
        self._note_bytes(decoded)

    def _api_url(self, path: str) -> str:
        parsed = urllib.parse.urlsplit(self._params.api_url)
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
        )

    def _check_key(self) -> None:
        request = urllib.request.Request(
            self._api_url("/v1/auth/check-key"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._params.api_key}",
                "User-Agent": f"vllm quantum_{self._mode}",
            },
        )
        timeout = self._params.recv_timeout_ms / 1000.0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "quantum: failed to validate Quantum Lever API key"
            ) from exc

        capabilities = payload.get("capabilities", {})
        required = ("qrng",) if self._mode == "dist" else ("lever", "quantum_sampler")
        for capability in required:
            if capabilities.get(capability) is not True:
                if self._mode == "dist":
                    tier = "a free ql_day_ or ql_week_ key"
                else:
                    tier = "a subscriber ql_lever_ key"
                raise RuntimeError(
                    f"quantum_{self._mode}: API key lacks '{capability}' capability "
                    f"(requires {tier})"
                )

    def _compute_next_fetch_time(self, payload: dict[str, object]) -> float:
        cadence_ms = int(payload.get("cadence_ms", 2000))
        wait_ms = cadence_ms
        try:
            closed = _parse_api_time(payload["batch_closed_at"])
            server = _parse_api_time(payload["server_time"])
            wait_ms = cadence_ms - int((server - closed).total_seconds() * 1000) + 25
        except (KeyError, TypeError, ValueError):
            pass
        wait_ms = max(50, min(wait_ms, max(50, cadence_ms)))
        return time.monotonic() + wait_ms / 1000.0

    def _wait_for_next_fetch(self) -> None:
        delay = self._next_fetch_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _note_bytes(self, data: bytes) -> None:
        for byte in data:
            for shift in range(8):
                bit = (byte >> shift) & 1
                self._bits.append(bit)
                self._bit_ones += bit
                if len(self._bits) > QUANTUM_BIAS_WINDOW_BITS:
                    self._bit_ones -= self._bits.popleft()

    def _log_diagnostics_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_diag_time < 1.0:
            return
        mean = self._bit_ones / len(self._bits) if self._bits else 0.0
        if self._mode == "floor":
            logger.info(
                "quantum_floor bytes=%d words=%d mean=%.6f lag1_corr=%.6f",
                self._bytes_received,
                self._words_produced,
                mean,
                _lag1_correlation(self._bits),
            )
            warning = quantum_floor_bias_warning(
                len(self._bits), mean, now, self._last_bias_warning
            )
            if warning is not None:
                logger.warning("quantum_floor %s", warning)
                self._last_bias_warning = now
        else:
            logger.info(
                "quantum_dist bytes=%d words=%d mean=%.6f",
                self._bytes_received,
                self._words_produced,
                mean,
            )
        self._last_diag_time = now


def _parse_api_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("API timestamp must be a string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def quantum_floor_bias_warning(
    bit_count: int,
    mean: float,
    now: float,
    last_warning: float,
) -> str | None:
    if bit_count < QUANTUM_BIAS_WINDOW_BITS or 0.45 <= mean <= 0.55:
        return None
    if now - last_warning < 60.0:
        return None
    delta = abs(2.0 * mean - 1.0)
    return (
        f"detector bias mean={mean:.6f} implied_delta={delta:.6f} exceeds "
        "the K=64 design point for detector imbalance"
    )


def _lag1_correlation(bits: deque[int]) -> float:
    if len(bits) < 2:
        return 0.0
    values = np.fromiter(bits, dtype=np.float64)
    x = values[:-1]
    y = values[1:]
    mx = float(x.mean())
    my = float(y.mean())
    variance = mx * (1.0 - mx) * my * (1.0 - my)
    if variance <= 0.0:
        return 0.0
    covariance = float((x * y).mean()) - mx * my
    return covariance / math.sqrt(variance)


def quantum_personalize_word(word: int, personalization: str, word_index: int) -> int:
    if not personalization:
        return word & QUANTUM_FLOOR_MASK
    key = _quantum_personalization_key(personalization)
    block = _quantum_chacha20_block(key, word_index // 16)
    return (word ^ block[word_index % 16]) & QUANTUM_FLOOR_MASK


def _quantum_personalization_key(personalization: str) -> tuple[int, ...]:
    words: list[int] = []
    for i in range(4):
        h = _quantum_fnv1a64(personalization, 0x9E3779B97F4A7C15 * (i + 1))
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
