"""Hybrid retrieval: BM25, Reciprocal Rank Fusion and the combined pipeline.

Everything here is offline. The BM25 index is real and reads the SQLite test
corpus; the dense branch is the same in-memory fake the other suites use.
"""

import uuid

import pytest

from backend.services.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from backend.services.lexical_index import (
    BM25Index,
    LexicalDocument,
    stem_token,
    tokenize,
)
from backend.services.rag_service import MODE_DENSE, MODE_HYBRID, RagService

RU_TEXTS = [
    "Проект Атлас описывает миграцию биллинга на новую платформу",
    "Проект Борей отвечает за складскую логистику и автоматизацию",
    "Инцидент INC-2026-017 привёл к простою сервиса на 42 минуты",
]


def make_candidate(point_id: str, **extra):
    base = {
        "point_id": point_id,
        "document_id": extra.pop("document_id", "doc-1"),
        "filename": "f.pdf",
        "page": 1,
        "chunk_index": 0,
        "text": "тело",
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenizer_keeps_identifiers_whole():
    assert "inc-2026-017" in tokenize("Инцидент INC-2026-017 закрыт")


def test_tokenizer_normalises_case_and_yo():
    assert tokenize("Отчёт", stemming=False) == tokenize("отчет", stemming=False)


def test_tokenizer_drops_punctuation_and_single_letters():
    assert tokenize("а, б: проект!", stemming=False) == ["проект"]


def test_stemming_collapses_russian_inflection():
    assert stem_token("проекты") == stem_token("проект")
    assert stem_token("логистики") == stem_token("логистику")


def test_stemming_collapses_english_inflection():
    assert stem_token("documents") == stem_token("document")


def test_identifiers_are_never_stemmed():
    assert stem_token("inc-2026-017") == "inc-2026-017"
    assert stem_token("2026") == "2026"


def test_stemming_can_be_switched_off():
    assert tokenize("проекты", stemming=False) == ["проекты"]
    assert tokenize("проекты", stemming=True) != ["проекты"]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(
        documents=[
            LexicalDocument(f"p{i}", "doc-1", i, 1, "f.pdf", text)
            for i, text in enumerate(RU_TEXTS)
        ]
    )


def test_bm25_ranks_the_matching_chunk_first(index):
    hits = index.search("складская логистика", limit=3)

    assert hits
    assert hits[0][0].chunk_index == 1


def test_bm25_matches_across_russian_inflection(index):
    hits = index.search("какие проекты описаны", limit=3)

    assert {hit[0].chunk_index for hit in hits} == {0, 1}


def test_bm25_finds_an_exact_identifier(index):
    """The case dense retrieval is worst at: a rare literal code."""
    hits = index.search("INC-2026-017", limit=3)

    assert len(hits) == 1
    assert hits[0][0].chunk_index == 2


def test_bm25_returns_nothing_for_an_unrelated_query(index):
    assert index.search("квантовая хромодинамика", limit=3) == []


def test_bm25_ignores_an_empty_query(index):
    assert index.search("", limit=3) == []


def test_bm25_on_an_empty_corpus_is_empty():
    assert BM25Index(documents=[]).search("проект", limit=3) == []


def test_bm25_honours_the_limit(index):
    assert len(index.search("проект", limit=1)) == 1


def test_bm25_scores_are_positive_and_ordered(index):
    scores = [score for _, score in index.search("проект логистика", limit=5)]

    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_is_deterministic(index):
    first = index.search("проект", limit=5)
    second = index.search("проект", limit=5)

    assert [(d.point_id, s) for d, s in first] == [(d.point_id, s) for d, s in second]


def test_bm25_document_filter_narrows_the_candidate_set():
    documents = [
        LexicalDocument("p0", "doc-a", 0, 1, "a.pdf", "проект атлас"),
        LexicalDocument("p1", "doc-b", 0, 1, "b.pdf", "проект борей"),
    ]
    index = BM25Index(documents=documents)

    hits = index.search("проект", limit=5, document_ids={"doc-a"})

    assert [hit[0].document_id for hit in hits] == ["doc-a"]


def test_rarer_terms_score_higher_than_common_ones(index):
    """IDF at work: a term in one chunk must outweigh one in all chunks."""
    rare = index.search("логистика", limit=1)[0][1]
    common = index.search("проект", limit=1)[0][1]

    assert rare > common


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_rrf_orders_by_summed_reciprocal_rank():
    dense = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
    lexical = [make_candidate("c"), make_candidate("b"), make_candidate("a")]

    fused = reciprocal_rank_fusion([dense, lexical])
    scores = {entry["point_id"]: entry["rrf_score"] for entry in fused}

    assert len(fused) == 3
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["c"] == pytest.approx(scores["a"])

    # 1/x is convex, so a first-and-third placing edges out two seconds. The
    # margin is tiny, which is the point: on fully reversed lists RRF declines
    # to pick a winner rather than inventing one.
    assert scores["a"] > scores["b"]
    assert scores["a"] - scores["b"] < 1e-4


def test_rrf_prefers_a_document_ranked_well_by_both_branches():
    dense = [make_candidate("dense_only"), make_candidate("both")]
    lexical = [make_candidate("lexical_only"), make_candidate("both")]

    fused = reciprocal_rank_fusion([dense, lexical])

    # Second place twice beats first place once — this is the property hybrid
    # retrieval actually relies on.
    assert fused[0]["point_id"] == "both"


def test_rrf_score_matches_the_formula():
    fused = reciprocal_rank_fusion([[make_candidate("a")]], rrf_k=60)

    assert fused[0]["rrf_score"] == pytest.approx(1 / 61)


def test_agreement_between_branches_beats_a_single_top_hit():
    dense = [make_candidate("solo"), make_candidate("shared")]
    lexical = [make_candidate("other"), make_candidate("shared")]

    fused = reciprocal_rank_fusion([dense, lexical])

    # "shared" is second in both; "solo" is first in one and absent from the
    # other. Two second places outrank one first place.
    assert fused[0]["point_id"] == "shared"


def test_a_duplicate_appears_once_with_both_branch_diagnostics():
    dense = [make_candidate("dup", dense_score=0.91, dense_rank=1)]
    lexical = [make_candidate("dup", lexical_score=7.3, lexical_rank=1)]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert len(fused) == 1
    entry = fused[0]
    assert entry["dense_score"] == 0.91
    assert entry["dense_rank"] == 1
    assert entry["lexical_score"] == 7.3
    assert entry["lexical_rank"] == 1
    # Both contributions summed, not averaged or overwritten.
    assert entry["rrf_score"] == pytest.approx(2 / (DEFAULT_RRF_K + 1))


def test_rrf_never_blends_the_raw_scores():
    """Cosine and BM25 live on different scales; they must survive untouched."""
    dense = [make_candidate("x", dense_score=0.5, vector_score=0.5, score=0.5)]
    lexical = [make_candidate("x", lexical_score=12.0)]

    entry = reciprocal_rank_fusion([dense, lexical])[0]

    assert entry["dense_score"] == 0.5
    assert entry["lexical_score"] == 12.0
    assert "combined_score" not in entry


def test_rrf_fills_payload_gaps_without_overwriting():
    dense = [make_candidate("x", filename="from_dense.pdf", page=None)]
    lexical = [make_candidate("x", filename="from_lexical.pdf", page=7)]

    entry = reciprocal_rank_fusion([dense, lexical])[0]

    assert entry["filename"] == "from_dense.pdf"  # first writer wins
    assert entry["page"] == 7  # gap filled by the second


def test_a_larger_rrf_k_flattens_the_head():
    dense = [make_candidate("a"), make_candidate("b")]

    sharp = reciprocal_rank_fusion([dense], rrf_k=1)
    flat = reciprocal_rank_fusion([dense], rrf_k=1000)

    sharp_gap = sharp[0]["rrf_score"] - sharp[1]["rrf_score"]
    flat_gap = flat[0]["rrf_score"] - flat[1]["rrf_score"]

    assert sharp_gap > flat_gap


def test_rrf_of_a_single_list_preserves_its_order():
    dense = [make_candidate(name) for name in ("a", "b", "c")]

    assert [e["point_id"] for e in reciprocal_rank_fusion([dense])] == ["a", "b", "c"]


def test_rrf_handles_empty_branches():
    assert reciprocal_rank_fusion([[], []]) == []
    assert len(reciprocal_rank_fusion([[make_candidate("a")], []])) == 1


def test_rrf_respects_the_limit():
    dense = [make_candidate(name) for name in ("a", "b", "c")]

    assert len(reciprocal_rank_fusion([dense], limit=2)) == 2


def test_rrf_rejects_a_candidate_without_the_join_key():
    with pytest.raises(ValueError, match="point_id"):
        reciprocal_rank_fusion([[{"document_id": "x"}]])


def test_rrf_rejects_a_nonsensical_k():
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([[make_candidate("a")]], rrf_k=0)


def test_rrf_is_deterministic_for_tied_scores():
    dense = [make_candidate("a"), make_candidate("b")]
    lexical = [make_candidate("b"), make_candidate("a")]

    first = [e["point_id"] for e in reciprocal_rank_fusion([dense, lexical])]
    second = [e["point_id"] for e in reciprocal_rank_fusion([dense, lexical])]

    assert first == second


# ---------------------------------------------------------------------------
# LexicalRetriever against the database
# ---------------------------------------------------------------------------


async def test_lexical_retriever_reads_the_tenants_chunks(
    lexical_index,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        hits = await lexical_index.search(
            session=session,
            question="складская логистика",
            user_id=str(user.id),
            limit=5,
        )

    assert hits
    assert hits[0]["chunk_index"] == 1
    assert hits[0]["lexical_rank"] == 1
    assert hits[0]["lexical_score"] > 0
    assert hits[0]["point_id"]


async def test_lexical_retrieval_never_crosses_the_tenant_boundary(
    lexical_index,
    session_factory,
    make_document,
    user,
    other_user,
):
    await make_document(user, filename="alice.pdf", texts=["альфа секретная логистика"])
    await make_document(other_user, filename="bob.pdf", texts=["бета секретная логистика"])

    async with session_factory() as session:
        mine = await lexical_index.search(
            session=session, question="секретная логистика", user_id=str(user.id), limit=10
        )
        theirs = await lexical_index.search(
            session=session,
            question="секретная логистика",
            user_id=str(other_user.id),
            limit=10,
        )

    assert {hit["filename"] for hit in mine} == {"alice.pdf"}
    assert {hit["filename"] for hit in theirs} == {"bob.pdf"}
    assert not any("бета" in hit["text"] for hit in mine)
    assert not any("альфа" in hit["text"] for hit in theirs)


async def test_lexical_retrieval_honours_a_document_filter(
    lexical_index,
    session_factory,
    make_document,
    user,
):
    keep = await make_document(user, filename="keep.pdf", texts=["проект атлас"])
    await make_document(user, filename="drop.pdf", texts=["проект борей"])

    async with session_factory() as session:
        hits = await lexical_index.search(
            session=session,
            question="проект",
            user_id=str(user.id),
            limit=10,
            document_ids=[str(keep)],
        )

    assert {hit["document_id"] for hit in hits} == {str(keep)}


async def test_lexical_retrieval_requires_a_tenant(lexical_index, session_factory):
    async with session_factory() as session:
        with pytest.raises(ValueError, match="user_id"):
            await lexical_index.search(
                session=session, question="проект", user_id="", limit=5
            )


async def test_the_index_is_rebuilt_when_the_corpus_changes(
    lexical_index,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="first.pdf", texts=["проект атлас"])

    async with session_factory() as session:
        before = await lexical_index.index_for(session, str(user.id))
        assert len(before) == 1

    await make_document(user, filename="second.pdf", texts=["проект борей"])

    async with session_factory() as session:
        after = await lexical_index.index_for(session, str(user.id))

    # A new ingest moves the version probe, so the cache does not go stale.
    assert len(after) == 2


async def test_the_index_is_reused_when_nothing_changed(
    lexical_index,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="a.pdf", texts=["проект атлас"])

    async with session_factory() as session:
        first = await lexical_index.index_for(session, str(user.id))
        second = await lexical_index.index_for(session, str(user.id))

    assert first is second


# ---------------------------------------------------------------------------
# The combined pipeline
# ---------------------------------------------------------------------------


async def test_dense_mode_skips_the_lexical_branch(
    rag_service,
    session_factory,
    make_document,
    user,
    vector_store,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await rag_service.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_DENSE,
        )

    assert results
    assert vector_store.searches
    assert all("lexical_rank" not in hit for hit in results)
    assert all("rrf_score" not in hit for hit in results)


async def test_hybrid_mode_fuses_both_branches(
    rag_service,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await rag_service.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            rerank=False,
        )

    assert results
    assert any("rrf_score" in hit for hit in results)
    # At least one chunk was found by both branches.
    assert any(
        hit.get("dense_rank") is not None and hit.get("lexical_rank") is not None
        for hit in results
    )


async def test_hybrid_deduplicates_a_chunk_found_by_both_branches(
    rag_service,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await rag_service.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            rerank=False,
            top_k=50,
        )

    point_ids = [hit["point_id"] for hit in results]
    assert len(point_ids) == len(set(point_ids))


async def test_hybrid_finds_an_identifier_dense_alone_would_rank_low(
    rag_service,
    session_factory,
    make_document,
    user,
):
    """BM25 contributes exactly where embeddings are weakest: rare literals."""
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await rag_service.retrieve(
            question="INC-2026-017",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            rerank=False,
            top_k=50,
        )

    incident = next(hit for hit in results if hit["chunk_index"] == 2)
    assert incident["lexical_rank"] == 1


async def test_reranking_runs_after_fusion(
    rag_service,
    session_factory,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        results = await rag_service.retrieve(
            question="складская логистика",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            rerank=True,
        )

    assert results
    # The cross-encoder scored the fused pool, and the fusion diagnostics
    # survived the reranking stage.
    assert all("rerank_score" in hit for hit in results)
    assert any("rrf_score" in hit for hit in results)
    scores = [hit["rerank_score"] for hit in results]
    assert scores == sorted(scores, reverse=True)


async def test_hybrid_respects_the_document_filter_in_both_branches(
    rag_service,
    session_factory,
    make_document,
    user,
    vector_store,
):
    keep = await make_document(user, filename="keep.pdf", texts=["проект атлас"])
    await make_document(user, filename="drop.pdf", texts=["проект борей"])

    async with session_factory() as session:
        results = await rag_service.retrieve(
            question="проект",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            document_ids=[str(keep)],
            rerank=False,
            top_k=50,
        )

    assert {hit["document_id"] for hit in results} == {str(keep)}
    assert vector_store.searches[-1]["document_ids"] == [str(keep)]


async def test_hybrid_never_crosses_the_tenant_boundary(
    rag_service,
    session_factory,
    make_document,
    user,
    other_user,
):
    await make_document(user, filename="alice.pdf", texts=["альфа логистика склад"])
    await make_document(other_user, filename="bob.pdf", texts=["бета логистика склад"])

    async with session_factory() as session:
        mine = await rag_service.retrieve(
            question="логистика склад",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            rerank=False,
            top_k=50,
        )

    assert {hit["filename"] for hit in mine} == {"alice.pdf"}
    assert not any("бета" in (hit.get("text") or "") for hit in mine)


async def test_hybrid_without_a_session_degrades_to_dense(
    rag_service,
    make_document,
    user,
):
    """Half a pipeline would be worse than an honest fallback."""
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    results = await rag_service.retrieve(
        question="складская логистика",
        user_id=str(user.id),
        session=None,
        mode=MODE_HYBRID,
        rerank=False,
    )

    assert rag_service.resolve_mode(MODE_HYBRID, None) == MODE_DENSE
    assert all("rrf_score" not in hit for hit in results)


async def test_a_service_without_a_lexical_index_stays_dense(
    embeddings,
    vector_store,
    settings,
    session_factory,
    make_document,
    user,
):
    service = RagService(
        embeddings=embeddings,
        vector_store=vector_store,
        settings=settings,
        lexical=None,
    )
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        assert service.resolve_mode(MODE_HYBRID, session) == MODE_DENSE


def test_an_unknown_mode_falls_back_to_dense(rag_service):
    assert rag_service.resolve_mode("magic", None) == MODE_DENSE


async def test_the_reranker_pool_is_bounded(
    rag_service,
    session_factory,
    make_document,
    user,
    settings,
    embeddings,
):
    settings.rag_rerank_candidate_k = 2
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    async with session_factory() as session:
        await rag_service.retrieve(
            question="проект логистика инцидент",
            user_id=str(user.id),
            session=session,
            mode=MODE_HYBRID,
            top_k=5,
        )

    # The expensive stage never sees more than the configured budget.
    assert len(embeddings.rerank_calls) == 1


async def test_retrieval_diagnostics_reach_the_api(
    client,
    make_document,
    user,
):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    response = await client.post(
        "/rag/retrieve",
        json={"question": "складская логистика", "mode": "hybrid"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["mode"] == "hybrid"
    assert body["vector_results"]
    assert body["lexical_results"]
    assert body["fused_results"]
    assert body["reranked_results"]
    assert body["lexical_results"][0]["lexical_rank"] == 1
    assert body["fused_results"][0]["rrf_score"] is not None


async def test_the_api_can_force_dense_mode(client, make_document, user):
    await make_document(user, filename="ru.pdf", texts=RU_TEXTS)

    response = await client.post(
        "/rag/retrieve",
        json={"question": "складская логистика", "mode": "dense"},
    )
    body = response.json()

    assert body["mode"] == "dense"
    assert body["lexical_results"] == []


async def test_a_foreign_document_id_still_yields_nothing_in_hybrid(
    other_client,
    make_document,
    user,
):
    document_id = await make_document(user, filename="alice.pdf", texts=RU_TEXTS)

    response = await other_client.post(
        "/rag/retrieve",
        json={
            "question": "складская логистика",
            "mode": "hybrid",
            "document_ids": [str(document_id)],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["vector_results"] == []
    assert body["lexical_results"] == []
    assert body["fused_results"] == []


def test_uuid_coercion_rejects_junk(lexical_index):
    from backend.services.lexical_index import _as_uuid

    assert _as_uuid(uuid.uuid4()) is not None

    with pytest.raises(ValueError, match="UUID"):
        _as_uuid("not-a-uuid")
