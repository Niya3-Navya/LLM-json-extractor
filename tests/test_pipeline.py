"""Run with:  python -m unittest discover -s tests -v
Uses a fake LLM, so no API key or network is needed."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract import ExtractionFailed, extract
from json_extractor import JSONExtractionError, extract_json
from prompts import EXAMPLES, build_prompt
from schemas import SCHEMAS, Invoice, invoice_warnings

GOOD_INVOICE = EXAMPLES["invoice"][0][1]


class FakeLLM:
    """Returns the queued replies one by one and records the prompts it receives."""
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class JsonExtractorTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fence_and_chatter(self):
        raw = 'Sure! Here you go:\n```json\n{"a": 1}\n```\nHope it helps.'
        self.assertEqual(extract_json(raw), {"a": 1})

    def test_chatter_without_fence(self):
        self.assertEqual(extract_json('Result: {"a": [1, 2]} done'), {"a": [1, 2]})

    def test_trailing_comma(self):
        self.assertEqual(extract_json('{"a": [1, 2,], "b": 2,}'), {"a": [1, 2], "b": 2})

    def test_errors(self):
        for bad in ["", "no json here", "{broken", "[1, 2]"]:
            with self.assertRaises(JSONExtractionError):
                extract_json(bad)


class SchemaTests(unittest.TestCase):
    def test_few_shot_examples_are_valid(self):
        # The examples shown to the model must themselves satisfy the schemas.
        for doc_type, examples in EXAMPLES.items():
            for _, output in examples:
                SCHEMAS[doc_type].model_validate(output)

    def test_warnings_flag_bad_arithmetic(self):
        bad = dict(GOOD_INVOICE, total=99.0)
        self.assertTrue(invoice_warnings(Invoice.model_validate(bad)))
        self.assertEqual(invoice_warnings(Invoice.model_validate(GOOD_INVOICE)), [])


    def test_discount_and_shipping_are_accounted_for(self):
        inv = dict(GOOD_INVOICE, subtotal=45.0, discount=5.0, shipping=10.0, tax=4.5, total=54.5)
        self.assertEqual(invoice_warnings(Invoice.model_validate(inv)), [])

    def test_numeric_invoice_number_is_coerced(self):
        inv = Invoice.model_validate(dict(GOOD_INVOICE, invoice_number=4521))
        self.assertEqual(inv.invoice_number, "4521")

    def test_extra_keys_and_bad_values_rejected(self):
        from pydantic import ValidationError
        for bad in (dict(GOOD_INVOICE, surprise="x"),
                    dict(GOOD_INVOICE, invoice_date="not a date"),
                    dict(GOOD_INVOICE, currency="DOLLARS"),
                    dict(GOOD_INVOICE, line_items=[])):
            with self.assertRaises(ValidationError):
                Invoice.model_validate(bad)


class PipelineTests(unittest.TestCase):
    def test_prompt_is_parameterized(self):
        p = build_prompt("resume", "Jane Doe, engineer")
        self.assertIn("extract resume data", p)
        self.assertIn("Jane Doe, engineer", p)
        self.assertIn("Example 1 output", p)

    def test_success_first_try_with_fenced_json(self):
        llm = FakeLLM("```json\n" + json.dumps(GOOD_INVOICE) + "\n```")
        result = extract("some invoice", "invoice", llm)
        self.assertEqual(result.total, 49.5)
        self.assertEqual(len(llm.prompts), 1)

    def test_retry_sends_error_back_and_recovers(self):
        bad = dict(GOOD_INVOICE, total="$49.50")  # currency symbol -> invalid number
        llm = FakeLLM(json.dumps(bad), json.dumps(GOOD_INVOICE))
        result = extract("some invoice", "invoice", llm)
        self.assertEqual(result.total, 49.5)
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("rejected with this error", llm.prompts[1])
        self.assertIn("total", llm.prompts[1])

    def test_gives_up_after_retries(self):
        llm = FakeLLM("garbage", "still garbage", "nope")
        with self.assertRaises(ExtractionFailed):
            extract("x", "invoice", llm, max_retries=2)
        self.assertEqual(len(llm.prompts), 3)


if __name__ == "__main__":
    unittest.main()
