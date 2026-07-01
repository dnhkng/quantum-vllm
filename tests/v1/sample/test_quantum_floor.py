# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from random import Random
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import torch

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import QuantumFloorParams, QuantumSeedParams, SamplingParams
from vllm.v1.worker.gpu.sample.quantum_floor import (
    QUANTUM_FLOOR_M,
    QUANTUM_FLOOR_MASK,
    QuantumLeverClient,
    build_allocation,
    compute_g_rows,
    quantum_lever_cache_key,
    select_token_index,
    spread_u32,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler


class FakeQRNGSource:
    def __init__(self, words):
        self.words = list(words)

    def read_u32(self):
        return self.words.pop(0)


class EntropyHandler(BaseHTTPRequestHandler):
    entropy = b""
    offset = 0
    requests = []

    def do_GET(self):
        EntropyHandler.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
            }
        )
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            return

        requested = int(parse_qs(parsed.query).get("bytes", ["4"])[0])
        start = EntropyHandler.offset
        end = start + requested
        EntropyHandler.offset = end
        payload = {
            "bytes_b64": base64.b64encode(EntropyHandler.entropy[start:end]).decode(
                "ascii"
            )
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):
        return


def test_quantum_floor_spreader_is_involutive():
    rows = compute_g_rows()
    assert len(rows) == 32
    assert rows[0] == 0xFFFFFFFE
    assert rows[0].bit_count() == 31

    for value in range(10000):
        assert spread_u32(spread_u32(value, rows), rows) == value
    rng = Random(0xC0FFEE)
    for _ in range(10000):
        value = rng.getrandbits(32)
        assert spread_u32(spread_u32(value, rows), rows) == value


def test_quantum_floor_allocation_applies_floor_and_preserves_address_space():
    probs = np.array([0.5, 0.25, 0.125, 0.12499996, 1.0e-8, 3.0e-8])
    alloc = build_allocation(probs, k=64)

    assert alloc.slots.shape == probs.shape
    assert alloc.cdf.shape == probs.shape
    assert int(alloc.slots.sum(dtype=np.uint64)) == QUANTUM_FLOOR_M
    assert int(alloc.cdf[-1]) == QUANTUM_FLOOR_MASK
    assert int(alloc.slots[4]) == 64

    r0 = alloc.slots[0] / np.floor(probs[0] * QUANTUM_FLOOR_M)
    r1 = alloc.slots[1] / np.floor(probs[1] * QUANTUM_FLOOR_M)
    r2 = alloc.slots[2] / np.floor(probs[2] * QUANTUM_FLOOR_M)
    assert abs(float(r0 - r1)) < 1e-6
    assert abs(float(r0 - r2)) < 1e-6


def test_quantum_floor_selects_expected_token_for_simple_distribution():
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    assert select_token_index(probs, raw_u32=0, k=1) == 0


def test_quantum_lever_client_reads_little_endian_words():
    rng = Random(12345)
    words = [rng.getrandbits(32) for _ in range(16)]
    EntropyHandler.entropy = b"".join(word.to_bytes(4, "little") for word in words)
    EntropyHandler.offset = 0
    EntropyHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), EntropyHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = QuantumLeverClient(
        QuantumFloorParams(
            api_url=f"http://127.0.0.1:{server.server_port}/entropy",
            api_key="test-key",
            buffer_size=16,
            recv_timeout_ms=1000,
        )
    )
    try:
        assert [client.read_u32() for _ in words] == words
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2)

    assert EntropyHandler.requests
    assert EntropyHandler.requests[0]["authorization"] == "Bearer test-key"
    assert parse_qs(urlsplit(EntropyHandler.requests[0]["path"]).query)["bytes"] == [
        "16"
    ]


def test_sampler_quantum_floor_branch_replaces_only_qrng_rows():
    qf = QuantumFloorParams(api_key="test-key", k=1)
    sampler = Sampler.__new__(Sampler)
    sampler.sampling_states = type(
        "SamplingStates",
        (),
        {"quantum_floor": [qf, None]},
    )()
    sampler._qrng_clients = {
        quantum_lever_cache_key(qf): FakeQRNGSource([0])
    }

    sampled = torch.tensor([9, 8])
    logits = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]))
    out = sampler._sample_quantum_floor(
        sampled,
        logits,
        expanded_idx_mapping=torch.tensor([0, 1]),
    )

    assert out.tolist() == [0, 8]


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"temperature": 0.0}, "temperature"),
        ({"top_k": 1}, "top_k"),
        ({"top_p": 0.9}, "top_p"),
        ({"min_p": 0.1}, "min_p"),
        ({"ignore_eos": True}, "ignore_eos"),
        ({"min_tokens": 1}, "min_tokens"),
        ({"allowed_token_ids": [1]}, "allowed_token_ids"),
        ({"logit_bias": {1: 1.0}}, "logit_bias"),
        ({"bad_words": ["bad"]}, "bad_words"),
        (
            {"quantum_floor": QuantumFloorParams(api_key="")},
            "quantum_floor.api_key",
        ),
        (
            {"quantum_floor": QuantumFloorParams(api_url="", api_key="test-key")},
            "quantum_floor.api_url",
        ),
        (
            {
                "quantum_floor": QuantumFloorParams(
                    api_key="test-key", buffer_size=3
                )
            },
            "quantum_floor.buffer_size",
        ),
        (
            {
                "quantum_floor": QuantumFloorParams(
                    api_key="test-key", require_full_vocab=False
                )
            },
            "quantum_floor.require_full_vocab",
        ),
    ],
)
def test_sampling_params_rejects_incompatible_quantum_floor(kwargs, parameter):
    kwargs.setdefault("quantum_floor", QuantumFloorParams(api_key="test-key"))
    with pytest.raises(VLLMValidationError) as exc_info:
        SamplingParams(**kwargs)
    assert exc_info.value.parameter == parameter


def test_sampling_params_rejects_quantum_floor_spec_decode_and_large_k():
    params = SamplingParams(
        quantum_floor=QuantumFloorParams(api_key="test-key", k=2)
    )

    class ModelConfig:
        @staticmethod
        def get_vocab_size():
            return 2**31 + 1

    with pytest.raises(VLLMValidationError) as exc_info:
        params._validate_quantum_floor(ModelConfig(), speculative_config=None)
    assert exc_info.value.parameter == "quantum_floor.k"

    with pytest.raises(VLLMValidationError) as exc_info:
        params._validate_quantum_floor(ModelConfig(), speculative_config=object())
    assert exc_info.value.parameter == "quantum_floor"


def test_sampling_params_repr_masks_quantum_api_key():
    params = SamplingParams(
        quantum_floor=QuantumFloorParams(api_key="secret-floor"),
    )
    assert "secret-floor" not in repr(params)
    assert "api_key" not in repr(params.quantum_floor)


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"quantum_seed": QuantumSeedParams(api_url="")}, "quantum_seed.api_url"),
        ({"quantum_seed": QuantumSeedParams(api_key="")}, "quantum_seed.api_key"),
        (
            {"quantum_seed": QuantumSeedParams(api_key="test-key", buffer_size=3)},
            "quantum_seed.buffer_size",
        ),
        (
            {
                "quantum_seed": QuantumSeedParams(
                    api_key="test-key", recv_timeout_ms=0
                )
            },
            "quantum_seed.recv_timeout_ms",
        ),
        (
            {
                "quantum_seed": QuantumSeedParams(api_key="test-key"),
                "seed": 1234,
            },
            "seed",
        ),
        (
            {
                "quantum_seed": QuantumSeedParams(api_key="test-key"),
                "quantum_floor": QuantumFloorParams(api_key="test-key"),
            },
            "quantum_seed",
        ),
    ],
)
def test_sampling_params_rejects_invalid_quantum_seed(kwargs, parameter):
    with pytest.raises(VLLMValidationError) as exc_info:
        SamplingParams(**kwargs)
    assert exc_info.value.parameter == parameter


def test_openai_completion_request_maps_quantum_floor():
    request = CompletionRequest(
        prompt="hello",
        quantum_floor={"api_key": "test-key", "k": 7, "buffer_size": 32},
    )
    params = request.to_sampling_params(max_tokens=4)
    assert params.quantum_floor == QuantumFloorParams(
        api_key="test-key", k=7, buffer_size=32
    )


def test_openai_completion_request_maps_quantum_seed():
    request = CompletionRequest(
        prompt="hello",
        quantum_seed={"api_key": "test-key", "buffer_size": 32},
    )
    params = request.to_sampling_params(max_tokens=4)
    assert params.quantum_seed == QuantumSeedParams(api_key="test-key", buffer_size=32)


def test_openai_chat_request_maps_quantum_floor():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        quantum_floor={"api_key": "test-key", "k": 7, "buffer_size": 32},
    )
    params = request.to_sampling_params(max_tokens=4, default_sampling_params={})
    assert params.quantum_floor == QuantumFloorParams(
        api_key="test-key", k=7, buffer_size=32
    )


def test_openai_chat_request_maps_quantum_seed():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        quantum_seed={"api_key": "test-key", "buffer_size": 32},
    )
    params = request.to_sampling_params(max_tokens=4, default_sampling_params={})
    assert params.quantum_seed == QuantumSeedParams(api_key="test-key", buffer_size=32)


def test_openai_responses_request_maps_quantum_floor():
    request = ResponsesRequest(
        input="hello",
        quantum_floor={"api_key": "test-key", "k": 7, "buffer_size": 32},
    )
    params = request.to_sampling_params(default_max_tokens=4)
    assert params.quantum_floor == QuantumFloorParams(
        api_key="test-key", k=7, buffer_size=32
    )


def test_openai_responses_request_maps_quantum_seed():
    request = ResponsesRequest(
        input="hello",
        quantum_seed={"api_key": "test-key", "buffer_size": 32},
    )
    params = request.to_sampling_params(default_max_tokens=4)
    assert params.quantum_seed == QuantumSeedParams(api_key="test-key", buffer_size=32)
