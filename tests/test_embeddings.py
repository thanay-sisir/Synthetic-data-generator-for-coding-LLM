import json
import threading
import unittest
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from kernel_lab.embed_server import DIMENSIONS, embed_request, make_handler


class FakeEncoder:
    model_id = "fake@immutable-revision"

    def encode(self, texts):
        return [[1.0] + [0.0] * (DIMENSIONS - 1) for _ in texts], len(texts), []


class EmbeddingTests(unittest.TestCase):
    def test_batch_order_and_dimensions(self):
        result = embed_request(FakeEncoder(), {"model": FakeEncoder.model_id, "input": ["a", "b"]})
        self.assertEqual([r["index"] for r in result["data"]], [0, 1])
        self.assertEqual(len(result["data"][0]["embedding"]), 384)

    def test_invalid_requests(self):
        for fields in ({"model": "wrong", "input": "a"}, {"input": []}, {"input": [12]},
                       {"input": "a", "dimensions": 12}, {"input": "a", "encoding_format": "base64"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                embed_request(FakeEncoder(), {"model": FakeEncoder.model_id, **fields})

    def test_http_health_authorization_and_embedding(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(FakeEncoder(), "test-key"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/health") as response:
                self.assertEqual(json.load(response)["model"], FakeEncoder.model_id)
            payload = json.dumps({"model": FakeEncoder.model_id, "input": "a"}).encode()
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + "/v1/embeddings", data=payload))
            self.assertEqual(caught.exception.code, 401)
            request = Request(base + "/v1/embeddings", data=payload,
                              headers={"Authorization": "Bearer test-key"})
            with urlopen(request) as response:
                self.assertEqual(json.load(response)["usage"]["total_tokens"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
