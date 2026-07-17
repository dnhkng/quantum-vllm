# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from random import Random

import numpy as np
import pytest

from vllm.quantum.params import QuantumDistParams, QuantumFloorParams
from vllm.quantum.quantum_floor import (
    QUANTUM_BIAS_WINDOW_BITS,
    QUANTUM_FLOOR_M,
    QUANTUM_FLOOR_MASK,
    QuantumLeverClient,
    build_allocation,
    build_dist_allocation,
    compute_g_rows,
    quantum_floor_bias_warning,
    quantum_personalize_word,
    select_dist_token_index,
    select_token_index,
    spread_u32,
)


class FakeQRNGSource:
    def __init__(self, words):
        self.words = list(words)

    def read_u32(self):
        return self.words.pop(0)


class QuantumAPIHandler(BaseHTTPRequestHandler):
    capabilities = {"qrng": True, "lever": True, "quantum_sampler": True}
    qrng_payloads = [b"\x01\x02\x03\x04"]
    lever_payloads = [b"old!", b"new!"]
    lever_hashes = ["lever-0", "lever-1"]
    requests = []
    lever_index = 0

    def do_GET(self):
        QuantumAPIHandler.requests.append(self.path)
        if self.path == "/v1/auth/check-key":
            payload = {"capabilities": QuantumAPIHandler.capabilities}
        elif self.path == "/v1/qrng/latest":
            payload = self._snapshot(QuantumAPIHandler.qrng_payloads[0], "qrng-0")
        elif self.path == "/v1/lever/latest":
            index = min(
                QuantumAPIHandler.lever_index,
                len(QuantumAPIHandler.lever_payloads) - 1,
            )
            QuantumAPIHandler.lever_index += 1
            payload = self._snapshot(
                QuantumAPIHandler.lever_payloads[index],
                QuantumAPIHandler.lever_hashes[index],
            )
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    @staticmethod
    def _snapshot(data, payload_hash):
        return {
            "payload_b64": base64.b64encode(data).decode(),
            "payload_hash": payload_hash,
            "cadence_ms": 50,
        }

    def log_message(self, format, *args):
        return


@pytest.fixture
def quantum_api():
    QuantumAPIHandler.capabilities = {
        "qrng": True,
        "lever": True,
        "quantum_sampler": True,
    }
    QuantumAPIHandler.requests = []
    QuantumAPIHandler.lever_index = 0
    QuantumAPIHandler.lever_payloads = [b"old!", b"new!"]
    QuantumAPIHandler.lever_hashes = ["lever-0", "lever-1"]
    server = HTTPServer(("127.0.0.1", 0), QuantumAPIHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_quantum_floor_spreader_is_involutive():
    rows = compute_g_rows()
    assert rows[0] == 0xFFFFFFFE
    rng = Random(0xC0FFEE)
    for _ in range(10000):
        value = rng.getrandbits(32)
        assert spread_u32(spread_u32(value, rows), rows) == value


def test_tier_boundary_dist_unreachable_floor_reachable():
    probs = np.full(250000, (1.0 - 1.0e-30) / 249999)
    probs[-1] = 1.0e-30
    dist = build_dist_allocation(probs)
    floor = build_allocation(probs, k=64)

    assert dist.slots[-1] == 0
    assert floor.slots[-1] >= 64
    address = int(floor.cdf[-2])
    raw = spread_u32(address)
    assert select_token_index(probs, raw, k=64) == probs.size - 1


def test_dist_resolution_edge_and_determinism():
    unit = 1.0 / QUANTUM_FLOOR_M
    probs = np.array([1.0 - 2.0 * unit, 1.5 * unit, 0.5 * unit])
    first = build_dist_allocation(probs)
    second = build_dist_allocation(probs)

    assert first.slots[1] >= 1
    assert first.slots[2] == 0
    np.testing.assert_array_equal(first.slots, second.slots)
    for word in (0, 123456789, QUANTUM_FLOOR_MASK):
        assert select_dist_token_index(probs, word) == select_dist_token_index(
            probs, word
        )


@pytest.mark.parametrize("size", [32000, 128000, 250000])
def test_dist_zipf_allocation_fills_address_space(size):
    probs = 1.0 / np.arange(1, size + 1, dtype=np.float64)
    alloc = build_dist_allocation(probs)
    assert int(alloc.slots.sum(dtype=np.uint64)) == QUANTUM_FLOOR_M
    assert int(alloc.cdf[-1]) == QUANTUM_FLOOR_M


def test_dist_degenerate_distributions():
    assert build_dist_allocation(np.array([1.0])).slots.tolist() == [QUANTUM_FLOOR_M]
    equal = build_dist_allocation(np.ones(250000))
    assert set(equal.slots.tolist()) == {17179, 17180}

    dominant = build_dist_allocation(np.array([1.0 - 1.0e-12, 1.0e-12]))
    assert int(dominant.slots.sum()) == QUANTUM_FLOOR_M
    assert dominant.slots[0] > dominant.slots[1]


def test_floor_behavior_and_max_address_remain_stable():
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    alloc = build_allocation(probs, k=64)
    assert int(alloc.slots.sum(dtype=np.uint64)) == QUANTUM_FLOOR_M
    assert int(alloc.cdf[-1]) == QUANTUM_FLOOR_MASK
    assert select_token_index(probs, 0, k=1) == 0
    assert select_token_index(probs, spread_u32(QUANTUM_FLOOR_MASK), k=1) == 3


def test_mode_endpoints_and_floor_discards_initial_batch(quantum_api):
    QuantumAPIHandler.capabilities = {"qrng": True}
    dist = QuantumLeverClient(
        QuantumDistParams(api_url=quantum_api, api_key="test"), "dist"
    )
    dist.start()
    assert dist.read_u32() == 0x04030201

    QuantumAPIHandler.capabilities = {"lever": True, "quantum_sampler": True}
    floor = QuantumLeverClient(
        QuantumFloorParams(api_url=quantum_api, api_key="test"), "floor"
    )
    floor.start()
    assert floor.read_u32() == int.from_bytes(b"new!", "little")
    assert QuantumAPIHandler.requests == [
        "/v1/auth/check-key",
        "/v1/qrng/latest",
        "/v1/auth/check-key",
        "/v1/lever/latest",
        "/v1/lever/latest",
    ]


def test_floor_does_not_reingest_duplicate_payload_hash(quantum_api):
    QuantumAPIHandler.capabilities = {"lever": True, "quantum_sampler": True}
    QuantumAPIHandler.lever_payloads = [b"old!", b"old!", b"new!"]
    QuantumAPIHandler.lever_hashes = ["old", "old", "new"]
    client = QuantumLeverClient(
        QuantumFloorParams(api_url=quantum_api, api_key="test"), "floor"
    )
    client.start()
    assert client.read_u32() == int.from_bytes(b"new!", "little")
    assert QuantumAPIHandler.requests.count("/v1/lever/latest") == 3


@pytest.mark.parametrize(
    "mode, capabilities, missing",
    [
        ("dist", {}, "qrng"),
        ("floor", {"qrng": True}, "lever"),
        ("floor", {"lever": True}, "quantum_sampler"),
    ],
)
def test_capability_matrix_rejects_missing_capability(
    quantum_api, mode, capabilities, missing
):
    QuantumAPIHandler.capabilities = capabilities
    params = (
        QuantumDistParams(api_url=quantum_api, api_key="test")
        if mode == "dist"
        else QuantumFloorParams(api_url=quantum_api, api_key="test")
    )
    with pytest.raises(RuntimeError, match=missing):
        QuantumLeverClient(params, mode).start()


def test_floor_bias_warning_and_rate_limit():
    warning = quantum_floor_bias_warning(
        QUANTUM_BIAS_WINDOW_BITS, 0.58, 100.0, float("-inf")
    )
    assert warning is not None
    assert "mean=0.580000" in warning
    assert "implied_delta=0.160000" in warning
    assert (
        quantum_floor_bias_warning(QUANTUM_BIAS_WINDOW_BITS, 0.50, 100.0, float("-inf"))
        is None
    )
    assert (
        quantum_floor_bias_warning(QUANTUM_BIAS_WINDOW_BITS, 0.58, 159.0, 100.0) is None
    )


def test_quantum_personalization_matches_llama_reference_values():
    assert quantum_personalize_word(0x04030201, "", 0) == 0x04030201
    assert quantum_personalize_word(0x04030201, "alice", 0) == 0x7A920679
    assert quantum_personalize_word(0x04030201, "alice", 16) == 0xEAA9FC26


def test_sampler_replaces_only_quantum_rows():
    import torch

    from vllm.v1.worker.gpu.sample.sampler import Sampler

    qd = QuantumDistParams(api_key="test")
    sampler = Sampler.__new__(Sampler)
    sampler.sampling_states = type(
        "SamplingStates",
        (),
        {
            "quantum_floor": [None, None],
            "quantum_dist": [qd, None],
            "temperature": type("Temperature", (), {"np": np.array([1.0, 1.0])})(),
        },
    )()
    sampler._quantum_clients = {0: FakeQRNGSource([0])}
    sampled = torch.tensor([9, 8])
    logits = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]))
    out = sampler._sample_quantum(sampled, logits, torch.tensor([0, 1]))
    assert out.tolist() == [0, 8]


def test_quantum_dist_preserves_greedy_sampling_but_consumes_one_word():
    import torch

    from vllm.v1.worker.gpu.sample.sampler import Sampler

    qd = QuantumDistParams(api_key="test")
    source = FakeQRNGSource([QUANTUM_FLOOR_MASK])
    sampler = Sampler.__new__(Sampler)
    sampler.sampling_states = type(
        "SamplingStates",
        (),
        {
            "quantum_floor": [None],
            "quantum_dist": [qd],
            "temperature": type("Temperature", (), {"np": np.array([0.0])})(),
        },
    )()
    sampler._quantum_clients = {0: source}
    out = sampler._sample_quantum(
        torch.tensor([2]), torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([0])
    )
    assert out.tolist() == [2]
    assert source.words == []


@pytest.mark.parametrize(
    "params, parameter",
    [
        (QuantumDistParams(api_url="", api_key="test"), "quantum_dist.api_url"),
        (QuantumDistParams(api_key=""), "quantum_dist.api_key"),
        (
            QuantumDistParams(api_key="test", recv_timeout_ms=0),
            "quantum_dist.recv_timeout_ms",
        ),
    ],
)
def test_sampling_params_rejects_invalid_quantum_dist(params, parameter):
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    with pytest.raises(VLLMValidationError) as exc_info:
        SamplingParams(quantum_dist=params)
    assert exc_info.value.parameter == parameter


def test_sampling_params_rejects_dist_with_floor():
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    with pytest.raises(VLLMValidationError) as exc_info:
        SamplingParams(
            quantum_dist=QuantumDistParams(api_key="test"),
            quantum_floor=QuantumFloorParams(api_key="test"),
        )
    assert exc_info.value.parameter == "quantum_dist"


def test_quantum_dist_rejects_speculative_decoding():
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(quantum_dist=QuantumDistParams(api_key="test"))

    class ModelConfig:
        @staticmethod
        def get_vocab_size():
            return 1000

    with pytest.raises(VLLMValidationError) as exc_info:
        params._validate_quantum(ModelConfig(), speculative_config=object())
    assert exc_info.value.parameter == "quantum_dist"


@pytest.mark.parametrize(
    "name, value",
    [
        ("presence_penalty", 0.1),
        ("frequency_penalty", 0.1),
        ("repetition_penalty", 1.1),
        ("top_p", 0.9),
        ("top_k", 10),
        ("min_p", 0.1),
    ],
)
def test_quantum_floor_allows_only_temperature_to_modify_distribution(name, value):
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    with pytest.raises(VLLMValidationError) as exc_info:
        SamplingParams(
            quantum_floor=QuantumFloorParams(api_key="test"), **{name: value}
        )
    assert exc_info.value.parameter == name


def test_openai_protocols_map_quantum_modes():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    completion = CompletionRequest(
        prompt="hello", quantum_dist={"api_key": "free"}
    ).to_sampling_params(max_tokens=4)
    assert completion.quantum_dist == QuantumDistParams(api_key="free")

    chat = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        quantum_api_key="subscriber",
        quantum_sampler=True,
        quantum_k=7,
    ).to_sampling_params(max_tokens=4, default_sampling_params={})
    assert chat.quantum_floor == QuantumFloorParams(api_key="subscriber", k=7)

    responses = ResponsesRequest(
        input="hello", quantum_api_key="free"
    ).to_sampling_params(default_max_tokens=4)
    assert responses.quantum_dist == QuantumDistParams(api_key="free")


def test_quantum_cli_check_uses_dist_mode(quantum_api, capsys):
    from vllm.quantum.cli import maybe_run_quantum_api_check

    args = Namespace(
        quantum_api_key="test",
        quantum_api_url=quantum_api,
        quantum_sampler=False,
        quantum_personalization="",
        quantum_k=64,
        quantum_recv_timeout=1000,
    )
    assert maybe_run_quantum_api_check(args)
    output = capsys.readouterr().out
    assert "0x04030201" in output
    assert "test" not in output
