from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

import scripts.rag_chatbot as rag


def make_document(title="Concert à Tours"):
    return SimpleNamespace(
        metadata={
            "title": title,
            "date_begin": "2026-12-15",
            "city": "Tours",
            "address": "Place Plumereau"
        },
        page_content="Un événement musical ouvert au public."
    )


def test_build_context():
    document = make_document()

    context = rag.build_context([document])

    assert "Concert à Tours" in context
    assert "2026-12-15" in context
    assert "Tours" in context
    assert "Place Plumereau" in context
    assert "événement musical" in context


def test_build_context_empty():
    assert rag.build_context([]) == ""


def test_ask_chatbot():
    document = make_document()

    class FakeVectorstore:
        def similarity_search(self, question, k):
            assert question == "Je cherche un concert à Tours"
            assert k == 10
            return [document]

    class FakeLLM:
        def invoke(self, prompt):
            assert "Concert à Tours" in prompt
            assert "Je cherche un concert à Tours" in prompt
            return SimpleNamespace(
                content="Un concert est proposé à Tours le 15 décembre 2026."
            )

    answer, documents = rag.ask_chatbot(
        "Je cherche un concert à Tours",
        FakeVectorstore(),
        FakeLLM()
    )

    assert "concert" in answer.lower()
    assert documents == [document]


def test_ask_chatbot_without_documents():
    class FakeVectorstore:
        def similarity_search(self, question, k):
            return []

    class FakeLLM:
        def invoke(self, prompt):
            assert "CONTEXTE :" in prompt
            return SimpleNamespace(
                content="Je ne dispose pas d'informations suffisantes."
            )

    answer, documents = rag.ask_chatbot(
        "Que me proposes-tu ?",
        FakeVectorstore(),
        FakeLLM()
    )

    assert answer is not None
    assert documents == []


def test_ask_chatbot_mistral_rate_limit():
    document = make_document()

    class FakeVectorstore:
        def similarity_search(self, question, k):
            return [document]

    class FakeLLM:
        def invoke(self, prompt):
            request = httpx.Request("POST", "https://api.mistral.ai")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError(
                "Limite de débit",
                request=request,
                response=response
            )

    answer, documents = rag.ask_chatbot(
        "Concert à Tours",
        FakeVectorstore(),
        FakeLLM()
    )

    assert answer is None
    assert documents == [document]


def test_ask_chatbot_other_http_error():
    class FakeVectorstore:
        def similarity_search(self, question, k):
            return []

    class FakeLLM:
        def invoke(self, prompt):
            request = httpx.Request("POST", "https://api.mistral.ai")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError(
                "Erreur serveur",
                request=request,
                response=response
            )

    with pytest.raises(httpx.HTTPStatusError):
        rag.ask_chatbot(
            "Question de test",
            FakeVectorstore(),
            FakeLLM()
        )


def test_create_llm_without_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        rag.create_llm()


def test_create_llm_with_api_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-de-test")

    with patch.object(rag, "ChatMistralAI") as mock_llm:
        rag.create_llm()

    mock_llm.assert_called_once_with(
        model="ministral-3b-2512",
        temperature=0.2,
        max_tokens=300,
        api_key="cle-de-test"
    )


def test_load_vectorstore():
    fake_embeddings = object()
    fake_vectorstore = object()

    with patch.object(
        rag,
        "HuggingFaceEmbeddings",
        return_value=fake_embeddings
    ) as mock_embeddings, patch.object(
        rag.FAISS,
        "load_local",
        return_value=fake_vectorstore
    ) as mock_load:

        result = rag.load_vectorstore()

    mock_embeddings.assert_called_once_with(
        model_name=rag.EMBEDDING_MODEL
    )
    mock_load.assert_called_once_with(
        str(rag.VECTORSTORE_DIR),
        fake_embeddings,
        allow_dangerous_deserialization=True
    )
    assert result is fake_vectorstore
