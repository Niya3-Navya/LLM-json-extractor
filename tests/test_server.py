"""Backend API tests with a fake LLM (no API key, no network).
Run with:  python -m unittest discover -s tests -v   (needs flask installed)"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server
from prompts import EXAMPLES

GOOD_INVOICE = EXAMPLES["invoice"][0][1]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_index_page_served(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_sample_endpoint(self):
        self.assertIn("text", self.client.get("/api/sample/invoice").get_json())
        self.assertEqual(self.client.get("/api/sample/../../etc/passwd").status_code, 404)

    def test_extract_success(self):
        fake = lambda prompt: json.dumps(GOOD_INVOICE)
        with mock.patch.object(server, "make_client", return_value=fake):
            res = self.client.post("/api/extract", data={"doc_type": "invoice", "text": "some invoice"})
        body = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["total"], 49.5)
        self.assertEqual(body["warnings"], [])

    def test_extract_validation_failure_returns_422(self):
        fake = lambda prompt: "not json at all"
        with mock.patch.object(server, "make_client", return_value=fake):
            res = self.client.post("/api/extract", data={"doc_type": "invoice", "text": "x"})
        self.assertEqual(res.status_code, 422)
        self.assertFalse(res.get_json()["success"])

    def test_bad_requests(self):
        self.assertEqual(self.client.post("/api/extract", data={"doc_type": "nope", "text": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/api/extract", data={"doc_type": "invoice", "text": "  "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
