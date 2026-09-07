import uuid

import pytest
from pydantic import ValidationError

from backend.schemas import (
    AskRequest,
    ConversationCreate,
    MessageCreate,
    RetrieveRequest,
    to_retrieved_chunk,
    to_source,
)


def test_message_create_defaults_to_plain_chat():
    payload = MessageCreate(content="Привет")

    assert payload.use_rag is False
    assert payload.document_ids is None
    assert payload.top_k is None


def test_message_create_rejects_empty_content():
    with pytest.raises(ValidationError):
        MessageCreate(content="")


def test_message_create_rejects_oversized_content():
    with pytest.raises(ValidationError):
        MessageCreate(content="x" * 5001)


def test_message_create_parses_document_ids():
    document_id = uuid.uuid4()
    payload = MessageCreate(content="q", use_rag=True, document_ids=[str(document_id)])

    assert payload.document_ids == [document_id]


def test_message_create_rejects_malformed_document_id():
    with pytest.raises(ValidationError):
        MessageCreate(content="q", use_rag=True, document_ids=["not-a-uuid"])


def test_conversation_create_has_default_title():
    assert ConversationCreate().title == "New conversation"


def test_conversation_create_rejects_long_title():
    with pytest.raises(ValidationError):
        ConversationCreate(title="t" * 256)


@pytest.mark.parametrize("top_k", [0, 21])
def test_ask_request_enforces_top_k_bounds(top_k):
    with pytest.raises(ValidationError):
        AskRequest(question="q", top_k=top_k)


def test_ask_request_defaults():
    request = AskRequest(question="q")

    assert request.top_k == 5
    assert request.candidate_k is None
    assert request.document_ids is None


def test_retrieve_request_enforces_candidate_bounds():
    with pytest.raises(ValidationError):
        RetrieveRequest(question="q", candidate_k=51)


def test_to_source_rounds_scores_and_stringifies_document_id():
    document_id = uuid.uuid4()

    source = to_source(
        {
            "document_id": document_id,
            "filename": "a.pdf",
            "page": 2,
            "chunk_index": 7,
            "score": 0.123456789,
            "vector_score": 0.123456789,
            "rerank_score": 4.987654321,
            "text": "body",
        }
    )

    assert source.document_id == str(document_id)
    assert source.score == 0.1235
    assert source.vector_score == 0.1235
    assert source.rerank_score == 4.9877


def test_to_source_tolerates_missing_fields():
    source = to_source({})

    assert source.document_id is None
    assert source.rerank_score is None
    assert source.score is None


def test_to_retrieved_chunk_keeps_text():
    chunk = to_retrieved_chunk({"text": "тело чанка", "score": 0.5})

    assert chunk.text == "тело чанка"
    assert chunk.score == 0.5
