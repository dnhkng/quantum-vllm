# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation helpers for Quantum Lever-backed sampling."""

from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.quantum.params import QuantumFloorParams, QuantumSeedParams


def verify_quantum_seed_args(
    quantum_seed: QuantumSeedParams | None,
    *,
    seed: int | None,
    quantum_floor: QuantumFloorParams | None,
) -> None:
    if quantum_seed is None:
        return

    qs = quantum_seed
    if not qs.api_url:
        raise VLLMValidationError(
            "quantum_seed.api_url must be non-empty.",
            parameter="quantum_seed.api_url",
            value=qs.api_url,
        )
    if not qs.api_key:
        raise VLLMValidationError(
            "quantum_seed.api_key must be non-empty.",
            parameter="quantum_seed.api_key",
            value=qs.api_key,
        )
    if qs.source not in ("qrng", "lever"):
        raise VLLMValidationError(
            "quantum_seed.source must be 'qrng' or 'lever'.",
            parameter="quantum_seed.source",
            value=qs.source,
        )
    if qs.buffer_size < 4:
        raise VLLMValidationError(
            "quantum_seed.buffer_size must be >= 4.",
            parameter="quantum_seed.buffer_size",
            value=qs.buffer_size,
        )
    if qs.recv_timeout_ms <= 0:
        raise VLLMValidationError(
            "quantum_seed.recv_timeout_ms must be positive.",
            parameter="quantum_seed.recv_timeout_ms",
            value=qs.recv_timeout_ms,
        )
    if seed is not None:
        raise VLLMValidationError(
            "quantum_seed cannot be used with an explicit seed.",
            parameter="seed",
            value=seed,
        )
    if quantum_floor is not None:
        raise VLLMValidationError(
            "quantum_seed cannot be used with quantum_floor.",
            parameter="quantum_seed",
            value=qs,
        )


def verify_quantum_floor_args(
    quantum_floor: QuantumFloorParams | None,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    ignore_eos: bool,
    min_tokens: int,
    structured_outputs: Any,
    allowed_token_ids: list[int] | None,
    logit_bias: dict[int, float] | None,
    bad_words: list[str] | None,
    sampling_eps: float,
) -> None:
    if quantum_floor is None:
        return

    qf = quantum_floor
    if not qf.api_url:
        raise VLLMValidationError(
            "quantum_floor.api_url must be non-empty.",
            parameter="quantum_floor.api_url",
            value=qf.api_url,
        )
    if not qf.api_key:
        raise VLLMValidationError(
            "quantum_floor.api_key must be non-empty.",
            parameter="quantum_floor.api_key",
            value=qf.api_key,
        )
    if qf.source not in ("qrng", "lever"):
        raise VLLMValidationError(
            "quantum_floor.source must be 'qrng' or 'lever'.",
            parameter="quantum_floor.source",
            value=qf.source,
        )
    if qf.k < 1:
        raise VLLMValidationError(
            "quantum_floor.k must be >= 1.",
            parameter="quantum_floor.k",
            value=qf.k,
        )
    if not qf.require_full_vocab:
        raise VLLMValidationError(
            "quantum_floor.require_full_vocab must be true.",
            parameter="quantum_floor.require_full_vocab",
            value=qf.require_full_vocab,
        )
    if qf.buffer_size < 4:
        raise VLLMValidationError(
            "quantum_floor.buffer_size must be >= 4.",
            parameter="quantum_floor.buffer_size",
            value=qf.buffer_size,
        )
    if qf.recv_timeout_ms <= 0:
        raise VLLMValidationError(
            "quantum_floor.recv_timeout_ms must be positive.",
            parameter="quantum_floor.recv_timeout_ms",
            value=qf.recv_timeout_ms,
        )
    if temperature < sampling_eps:
        raise VLLMValidationError(
            "quantum_floor requires temperature > 0.",
            parameter="temperature",
            value=temperature,
        )
    if top_k not in (0, -1):
        raise VLLMValidationError(
            "quantum_floor requires top_k to be disabled.",
            parameter="top_k",
            value=top_k,
        )
    if top_p != 1.0:
        raise VLLMValidationError(
            "quantum_floor requires top_p == 1.0.",
            parameter="top_p",
            value=top_p,
        )
    if min_p != 0.0:
        raise VLLMValidationError(
            "quantum_floor requires min_p == 0.0.",
            parameter="min_p",
            value=min_p,
        )
    if ignore_eos:
        raise VLLMValidationError(
            "quantum_floor is incompatible with ignore_eos.",
            parameter="ignore_eos",
            value=ignore_eos,
        )
    if min_tokens != 0:
        raise VLLMValidationError(
            "quantum_floor is incompatible with min_tokens.",
            parameter="min_tokens",
            value=min_tokens,
        )
    if structured_outputs is not None:
        raise VLLMValidationError(
            "quantum_floor is incompatible with structured_outputs.",
            parameter="structured_outputs",
            value=structured_outputs,
        )
    if allowed_token_ids:
        raise VLLMValidationError(
            "quantum_floor is incompatible with allowed_token_ids.",
            parameter="allowed_token_ids",
            value=allowed_token_ids,
        )
    if logit_bias:
        raise VLLMValidationError(
            "quantum_floor is incompatible with logit_bias.",
            parameter="logit_bias",
            value=logit_bias,
        )
    if bad_words:
        raise VLLMValidationError(
            "quantum_floor is incompatible with bad_words.",
            parameter="bad_words",
            value=bad_words,
        )


def validate_quantum_floor_model(
    quantum_floor: QuantumFloorParams | None,
    *,
    model_config: Any,
    speculative_config: Any,
) -> None:
    if quantum_floor is None:
        return
    if speculative_config is not None:
        raise VLLMValidationError(
            "quantum_floor is incompatible with speculative decoding.",
            parameter="quantum_floor",
            value=quantum_floor,
        )
    vocab_size = model_config.get_vocab_size()
    if quantum_floor.k * vocab_size > 2**32:
        raise VLLMValidationError(
            "quantum_floor.k * vocab_size exceeds the 2^32 address space.",
            parameter="quantum_floor.k",
            value=quantum_floor.k,
        )
