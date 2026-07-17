# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation helpers for Quantum Lever-backed sampling."""

from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.quantum.params import QuantumDistParams, QuantumFloorParams
from vllm.quantum.quantum_floor import QUANTUM_FLOOR_K_MAX


def verify_quantum_dist_args(
    quantum_dist: QuantumDistParams | None,
    *,
    quantum_floor: QuantumFloorParams | None,
) -> None:
    if quantum_dist is None:
        return

    qd = quantum_dist
    if not qd.api_url:
        raise VLLMValidationError(
            "quantum_dist.api_url must be non-empty.",
            parameter="quantum_dist.api_url",
            value=qd.api_url,
        )
    if not qd.api_key:
        raise VLLMValidationError(
            "quantum_dist.api_key must be non-empty.",
            parameter="quantum_dist.api_key",
            value=qd.api_key,
        )
    if qd.recv_timeout_ms <= 0:
        raise VLLMValidationError(
            "quantum_dist.recv_timeout_ms must be positive.",
            parameter="quantum_dist.recv_timeout_ms",
            value=qd.recv_timeout_ms,
        )
    if quantum_floor is not None:
        raise VLLMValidationError(
            "quantum_dist cannot be used with quantum_floor.",
            parameter="quantum_dist",
            value=qd,
        )


def verify_quantum_floor_args(
    quantum_floor: QuantumFloorParams | None,
    *,
    temperature: float,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
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
    if qf.k < 1:
        raise VLLMValidationError(
            "quantum_floor.k must be >= 1.",
            parameter="quantum_floor.k",
            value=qf.k,
        )
    if qf.k > QUANTUM_FLOOR_K_MAX:
        raise VLLMValidationError(
            "quantum_floor.k must be <= 2^20.",
            parameter="quantum_floor.k",
            value=qf.k,
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
    if presence_penalty != 0.0:
        raise VLLMValidationError(
            "quantum_floor requires presence_penalty == 0.0.",
            parameter="presence_penalty",
            value=presence_penalty,
        )
    if frequency_penalty != 0.0:
        raise VLLMValidationError(
            "quantum_floor requires frequency_penalty == 0.0.",
            parameter="frequency_penalty",
            value=frequency_penalty,
        )
    if repetition_penalty != 1.0:
        raise VLLMValidationError(
            "quantum_floor requires repetition_penalty == 1.0.",
            parameter="repetition_penalty",
            value=repetition_penalty,
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


def validate_quantum_model(
    quantum_dist: QuantumDistParams | None,
    quantum_floor: QuantumFloorParams | None,
    *,
    model_config: Any,
    speculative_config: Any,
) -> None:
    if quantum_dist is None and quantum_floor is None:
        return
    if speculative_config is not None:
        mode = "quantum_floor" if quantum_floor is not None else "quantum_dist"
        raise VLLMValidationError(
            f"{mode} is incompatible with speculative decoding.",
            parameter=mode,
            value=quantum_floor or quantum_dist,
        )
    if quantum_floor is None:
        return
    vocab_size = model_config.get_vocab_size()
    if quantum_floor.k * vocab_size > 2**32:
        raise VLLMValidationError(
            "quantum_floor.k * vocab_size exceeds the 2^32 address space.",
            parameter="quantum_floor.k",
            value=quantum_floor.k,
        )
