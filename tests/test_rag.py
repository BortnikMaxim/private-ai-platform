import uuid


async def test_retrieve_reranks_vector_hits(client, seeded_document):
    response = await client.post(
        "/rag/retrieve",
        json={"question": "складскую логистику", "top_k": 2, "candidate_k": 10},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["question"] == "складскую логистику"

    # The vector stage returns candidates in index order...
    assert [item["chunk_index"] for item in body["vector_results"]] == [0, 1, 2]

    # ...and the cross-encoder promotes the chunk that actually matches.
    assert len(body["reranked_results"]) == 2
    assert body["reranked_results"][0]["chunk_index"] == 1
    assert body["reranked_results"][0]["rerank_score"] > 0
    assert body["reranked_results"][0]["vector_score"] is not None
    assert body["reranked_results"][0]["text"]


async def test_retrieve_honours_candidate_k(client, seeded_document, vector_store):
    await client.post(
        "/rag/retrieve",
        json={"question": "проект", "top_k": 1, "candidate_k": 2},
    )

    assert vector_store.searches[-1]["limit"] == 2


async def test_retrieve_can_be_scoped_to_documents(client, seeded_document, vector_store):
    other_id = str(uuid.uuid4())

    response = await client.post(
        "/rag/retrieve",
        json={"question": "проект", "document_ids": [other_id]},
    )

    assert response.status_code == 200
    assert vector_store.searches[-1]["document_ids"] == [other_id]
    # Nothing is indexed for that document.
    assert response.json()["vector_results"] == []


async def test_ask_answers_from_retrieved_context(client, seeded_document, inference):
    response = await client.post(
        "/rag/ask",
        json={"question": "Какие проекты описаны?", "top_k": 2},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == inference.answer
    assert len(body["sources"]) == 2

    source = body["sources"][0]
    assert source["document_id"] == str(seeded_document)
    assert source["filename"] == "projects.pdf"
    # Legacy field kept alongside the explicit vector/rerank scores.
    assert source["score"] == source["vector_score"]

    prompt = inference.calls[-1]
    assert prompt[0]["role"] == "system"
    assert "[SOURCE 1]" in prompt[1]["content"]
    assert "Какие проекты описаны?" in prompt[1]["content"]


async def test_ask_returns_404_when_nothing_is_indexed(client):
    response = await client.post("/rag/ask", json={"question": "что-нибудь"})

    assert response.status_code == 404
    assert response.json()["detail"] == "No relevant documents found"


async def test_ask_reports_inference_outage_as_502(client, seeded_document, inference):
    inference.available = False

    response = await client.post("/rag/ask", json={"question": "Какие проекты?"})

    assert response.status_code == 502
    assert "traceback" not in response.text.lower()


async def test_ask_rejects_an_empty_question(client):
    response = await client.post("/rag/ask", json={"question": ""})

    assert response.status_code == 422


def test_build_context_marks_an_empty_retrieval(rag_service):
    context = rag_service.build_context([])

    assert "не найдено" in context


def test_build_context_respects_the_character_budget(rag_service, settings):
    chunks = [
        {"filename": "a.pdf", "page": 1, "chunk_index": index, "text": "x" * 5000}
        for index in range(10)
    ]

    context = rag_service.build_context(chunks)

    assert len(context) <= settings.max_context_chars
    assert context.startswith("[SOURCE 1]")
