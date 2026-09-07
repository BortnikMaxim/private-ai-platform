"""EmbeddingService: lazy loading, concurrency safety and tensor handling.

`sentence_transformers` is stubbed in sys.modules, so the real load/encode/
rerank code paths run without downloading a model.
"""

import asyncio
import sys
import time
import types

import numpy as np
import pytest

from backend.services.embeddings import EmbeddingService


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    counters = {"encoder": 0, "reranker": 0}

    class FakeSentenceTransformer:
        def __init__(self, name):
            counters["encoder"] += 1
            self.name = name
            # Slow enough that a missing lock would let a second loader in.
            time.sleep(0.05)

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            assert normalize_embeddings is True
            assert show_progress_bar is False
            # Real encoders return a numpy array, not a list.
            return np.array([[float(len(text)), 0.5, -0.25] for text in texts])

    class FakeCrossEncoder:
        def __init__(self, name):
            counters["reranker"] += 1
            self.name = name

        def predict(self, pairs, show_progress_bar=False):
            assert show_progress_bar is False
            return np.array([float(len(passage)) for _query, passage in pairs])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    module.CrossEncoder = FakeCrossEncoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    return counters


@pytest.fixture
def service(stub_sentence_transformers):
    return EmbeddingService(embedding_model="emb-model", reranker_model="rank-model")


async def test_models_are_not_loaded_until_first_use(service, stub_sentence_transformers):
    assert service.is_loaded is False
    assert stub_sentence_transformers["encoder"] == 0


async def test_first_use_loads_the_models_once(service, stub_sentence_transformers):
    await service.embed_query("вопрос")

    assert service.is_loaded is True
    assert stub_sentence_transformers["encoder"] == 1
    assert stub_sentence_transformers["reranker"] == 1

    await service.embed_query("другой вопрос")
    await service.embed_passages(["текст"])

    # Still one load for the whole process.
    assert stub_sentence_transformers["encoder"] == 1


async def test_concurrent_first_requests_load_the_models_only_once(
    service,
    stub_sentence_transformers,
):
    await asyncio.gather(*(service.ensure_loaded() for _ in range(10)))

    assert stub_sentence_transformers["encoder"] == 1
    assert stub_sentence_transformers["reranker"] == 1


async def test_concurrent_mixed_calls_load_the_models_only_once(
    service,
    stub_sentence_transformers,
):
    await asyncio.gather(
        service.embed_query("q1"),
        service.embed_passages(["p1", "p2"]),
        service.rerank("q2", [{"text": "t", "score": 0.5}], top_k=1),
        service.ensure_loaded(),
    )

    assert stub_sentence_transformers["encoder"] == 1
    assert stub_sentence_transformers["reranker"] == 1


async def test_embed_passages_uses_the_e5_passage_prefix(service, monkeypatch):
    await service.ensure_loaded()

    seen: list[list[str]] = []
    original_encode = service._encoder.encode

    def spy(texts, **kwargs):
        seen.append(list(texts))
        return original_encode(texts, **kwargs)

    monkeypatch.setattr(service._encoder, "encode", spy)

    await service.embed_passages(["первый", "второй"])

    assert seen[0] == ["passage: первый", "passage: второй"]


async def test_embed_query_uses_the_e5_query_prefix(service, monkeypatch):
    await service.ensure_loaded()

    seen: list[list[str]] = []
    original_encode = service._encoder.encode

    def spy(texts, **kwargs):
        seen.append(list(texts))
        return original_encode(texts, **kwargs)

    monkeypatch.setattr(service._encoder, "encode", spy)

    vector = await service.embed_query("вопрос")

    assert seen[0] == ["query: вопрос"]
    # Qdrant needs plain floats, not numpy scalars.
    assert isinstance(vector, list)
    assert all(type(value) is float for value in vector)


async def test_embed_passages_returns_plain_float_lists(service):
    vectors = await service.embed_passages(["a", "bb"])

    assert len(vectors) == 2
    assert all(type(value) is float for vector in vectors for value in vector)


async def test_embed_passages_of_nothing_skips_the_model(
    service,
    stub_sentence_transformers,
):
    assert await service.embed_passages([]) == []
    assert stub_sentence_transformers["encoder"] == 0


async def test_rerank_sorts_by_score_and_truncates(service):
    candidates = [
        {"text": "short", "score": 0.9},
        {"text": "a much longer chunk of text", "score": 0.1},
        {"text": "medium text", "score": 0.5},
    ]

    reranked = await service.rerank("вопрос", candidates, top_k=2)

    assert len(reranked) == 2
    # The stub scores by passage length, so the longest chunk wins.
    assert reranked[0]["text"] == "a much longer chunk of text"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
    # The original vector score is preserved alongside the new one.
    assert reranked[0]["vector_score"] == 0.1
    assert type(reranked[0]["rerank_score"]) is float


async def test_rerank_of_nothing_skips_the_model(service, stub_sentence_transformers):
    assert await service.rerank("вопрос", [], top_k=5) == []
    assert stub_sentence_transformers["reranker"] == 0
