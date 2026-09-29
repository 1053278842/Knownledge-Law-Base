import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from config import (
    get_chat_config,
    get_embedding_config,
    get_embedding_profile,
    get_index_dir,
    get_rerank_config,
)
from model_clients import (
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleChatClient,
    OpenAICompatibleRerankerClient,
    get_chat_client,
    get_embedding_client,
    get_reranker_client,
)
from llm_client import OpenAICompatibleChatClient as ChatClient


def _mock_json_response(payload: dict):
    response = MagicMock()
    response.read.return_value = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")
    response.__enter__.return_value = response
    return response


class ProviderConfigTest(unittest.TestCase):
    def test_legacy_deepseek_variables_are_ignored(self):
        legacy_environment = {
            "DEEPSEEK_API_KEY": "legacy-key",
            "DEEPSEEK_BASE_URL": "https://legacy.test",
            "DEEPSEEK_MODEL": "legacy-channel",
        }
        with patch.dict(os.environ, legacy_environment, clear=True):
            chat = get_chat_config()
            with self.assertRaises(ValueError):
                get_chat_client()

        self.assertEqual("", chat["url"])
        self.assertEqual("", chat["api_key"])
        self.assertEqual("", chat["channel"])

    def test_bocai_clients_use_separate_credentials(self):
        environment = {
            "CHAT_URL": "https://example.test/chat",
            "CHAT_API_KEY": "chat-key",
            "CHAT_CHANNEL": "chat-channel",
            "EMBEDDING_PROVIDER": "bocai",
            "EMBEDDING_URL": "https://example.test/embedding",
            "EMBEDDING_API_KEY": "embedding-key",
            "EMBEDDING_CHANNEL": "embedding-channel",
            "EMBEDDING_DIM": "1024",
            "RERANK_PROVIDER": "bocai",
            "RERANK_URL": "https://example.test/rerank",
            "RERANK_API_KEY": "rerank-key",
            "RERANK_CHANNEL": "rerank-channel",
        }
        with patch.dict(os.environ, environment, clear=True):
            chat = get_chat_config()
            embedding = get_embedding_config()
            rerank = get_rerank_config()
            profile = get_embedding_profile()
            index_dir = get_index_dir()
            chat_client = get_chat_client()
            embedding_client = get_embedding_client()
            rerank_client = get_reranker_client()

        self.assertEqual("https://example.test/chat", chat["url"])
        self.assertEqual("chat-key", chat["api_key"])
        self.assertEqual("chat-channel", chat["channel"])
        self.assertEqual(
            "https://example.test/embedding",
            embedding["url"],
        )
        self.assertEqual("embedding-key", embedding["api_key"])
        self.assertEqual("embedding-channel", embedding["channel"])
        self.assertEqual(
            "https://example.test/rerank",
            rerank["url"],
        )
        self.assertEqual("rerank-key", rerank["api_key"])
        self.assertEqual("rerank-channel", rerank["channel"])
        self.assertEqual("embedding-channel", profile["model"])
        self.assertEqual(1024, profile["dim"])
        self.assertIn("bocai-", index_dir.name)
        self.assertIsInstance(chat_client, OpenAICompatibleChatClient)
        self.assertIsInstance(
            embedding_client,
            OpenAICompatibleEmbeddingClient,
        )
        self.assertIsInstance(
            rerank_client,
            OpenAICompatibleRerankerClient,
        )


class RemoteEmbeddingClientTest(unittest.TestCase):
    @patch("model_clients.urlopen")
    def test_embedding_payload_order_and_normalization(self, mock_urlopen):
        mock_urlopen.return_value = _mock_json_response({
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ],
        })
        client = OpenAICompatibleEmbeddingClient(
            url="https://example.test/embedding",
            api_key="embedding-key",
            channel="embedding-channel",
            dim=2,
            batch_size=8,
            normalize=True,
            query_prefix="query:",
            document_prefix="doc:",
            timeout=12.0,
        )

        vectors = client.embed_texts(
            ["first", "second"],
            kind="document",
        )

        np.testing.assert_allclose(
            vectors,
            np.array([
                [0.6, 0.8],
                [0.0, 1.0],
            ], dtype="float32"),
        )
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(
            "Bearer embedding-key",
            request.get_header("Authorization"),
        )
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("embedding-channel", payload["model"])
        self.assertEqual(["doc:first", "doc:second"], payload["input"])


class OpenAICompatibleChatClientTest(unittest.TestCase):
    @patch("llm_client.urlopen")
    def test_chat_uses_configured_channel(self, mock_urlopen):
        mock_urlopen.return_value = _mock_json_response({
            "choices": [{"message": {"content": "answer"}}],
        })
        client = ChatClient(
            url="https://example.test/chat",
            api_key="chat-key",
            channel="chat-channel",
            timeout=12.0,
        )

        result = client.chat(
            [{"role": "user", "content": "question"}],
            temperature=0.1,
        )

        self.assertEqual("answer", result)
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("chat-channel", payload["model"])
        self.assertEqual(
            [{"role": "user", "content": "question"}],
            payload["messages"],
        )
        self.assertEqual(0.1, payload["temperature"])


class RemoteRerankerClientTest(unittest.TestCase):
    @patch("model_clients.urlopen")
    def test_rerank_maps_indexes_back_to_candidates(self, mock_urlopen):
        mock_urlopen.return_value = _mock_json_response({
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ],
        })
        candidates = [
            {"id": 10, "text": "first"},
            {"id": 20, "text": "second"},
            {"id": 30, "text": "third"},
        ]
        client = OpenAICompatibleRerankerClient(
            url="https://example.test/rerank",
            api_key="rerank-key",
            channel="rerank-channel",
            timeout=12.0,
        )

        results = client.rerank("query", candidates, top_k=2)

        self.assertEqual([30, 10], [item["id"] for item in results])
        self.assertEqual([0.9, 0.5], [
            item["rerank_score"] for item in results
        ])
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("rerank-channel", payload["model"])
        self.assertEqual("query", payload["query"])
        self.assertEqual(
            ["first", "second", "third"],
            payload["documents"],
        )
        self.assertEqual(2, payload["top_n"])


if __name__ == "__main__":
    unittest.main()
