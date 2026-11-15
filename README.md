# Structured LLM Prompt Pipeline + JSON Extractor

Parses unstructured invoice or resume text into **strictly validated JSON** using
Google Gemini (or OpenAI), parameterized prompt templates with few-shot examples, and Pydantic.

## Requirements checklist

| Requirement | Where it is implemented |
|---|---|
| Connect Python to Gemini or OpenAI | `llm_client.py` (`GeminiClient`, `OpenAIClient`, `--provider`) |
| Parameterized prompt templates | `prompts.py` (`TEMPLATE` with `{document_type}`, `{rules}`, `{schema}`, `{examples}`, `{text}`) |
| Few-shot examples | `prompts.py` (`EXAMPLES`, one per document type; unit-tested against the schemas) |
| Strictly validated JSON (Pydantic) | `schemas.py` (`Invoice`, `Resume`; `extra="forbid"`, typed fields, validators) |
| JSON extractor | `json_extractor.py` (fences, chatter, trailing commas) |
| Retry on invalid output | `extract.py` (`extract()` sends the error back to the model) |
| Invoice and resume analysis | `--type invoice` / `--type resume` |
| Complex unstructured documents | `samples/invoice_complex.txt` (EU number format, discount, shipping), `samples/resume_complex.txt` (non-standard layout) |
| Executable script | `python extract.py --type ... --file ...` |
| Backend + frontend (bonus) | `server.py` (Flask API) + `web/index.html` |

## How it works

```
document text
   -> prompts.py      fill template ({document_type}, {rules}, {schema}, {examples}, {text})
   -> llm_client.py   call Gemini
   -> json_extractor  strip ```fences```/chatter, fix trailing commas, json.loads
   -> schemas.py      Pydantic validation (types, dates, ranges, no extra keys)
        |-- valid   -> print JSON
        '-- invalid -> send the error back to the model and retry (default 2 retries)
```

| File | Purpose |
|---|---|
| `extract.py` | CLI + the pipeline loop (`extract()`) |
| `prompts.py` | One reusable template, per-type rules, few-shot examples, retry template |
| `schemas.py` | Pydantic models `Invoice`, `Resume` (+ soft arithmetic checks for invoices) |
| `json_extractor.py` | Recovers a dict from messy LLM output |
| `server.py` / `web/index.html` | Flask backend API + browser frontend |
| `llm_client.py` | Gemini and OpenAI wrappers (anything callable as `llm(prompt) -> str` works) |
| `tests/` | Unit tests using a fake LLM (no API key needed) |

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your Gemini key (https://aistudio.google.com) or OpenAI key
```

## Run

```bash
python extract.py --type invoice --file samples/invoice_sample.txt
python extract.py --type resume  --file samples/resume_sample.txt --out result.json
python extract.py --type invoice --file samples/invoice_complex.txt            # EU format, discount, shipping
python extract.py --type resume  --file samples/resume_complex.txt             # messy layout
python extract.py --type invoice --file samples/invoice_sample.txt --provider openai
python extract.py --type invoice --file samples/invoice_sample.txt --verbose   # show failed attempts
python extract.py --type invoice --file samples/invoice_sample.txt --show-prompt  # print prompt, no API call
```

Options: `--provider gemini|openai`, `--model NAME`, `--retries N`, `--out FILE`, `--verbose`, `--show-prompt`. PDFs work too (needs `pypdf`).
Exit code: 0 success, 1 validation failed after all retries, 2 bad input/config.

## Web app (backend + frontend)

```bash
python server.py            # then open http://localhost:5000
```

- **Frontend:** `web/index.html` (plain HTML/CSS/JS, no build step). Pick the document type and provider,
  paste text or upload a `.txt`/`.pdf`, click **Extract JSON**. Shows the validated JSON, invoice
  arithmetic warnings, how many retries were needed, and lets you copy/download the result.
- **Backend:** `server.py` (Flask) with `POST /api/extract` (form fields: `doc_type`, `provider`,
  `text` or `file`), `GET /api/sample/<name>`, and `GET /`. Errors return JSON with
  `success: false` (400 bad input, 422 validation failed after retries, 502 LLM call failed, 503 not configured).

```
Browser (web/index.html) --POST /api/extract--> server.py --> extract.py pipeline --> Gemini/OpenAI
                         <------ validated JSON + warnings ------
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Design notes

- **Few-shot:** each document type has a worked input -> output example in the prompt. A test
  checks that these examples satisfy the schema.
- **Strict validation:** `extra="forbid"`, typed numbers (`"$49.50"` is rejected), real dates,
  3-letter currency codes, `YYYY-MM` resume dates, email format.
- **Self-correcting retries:** the validation error and the model's bad answer are appended to
  the prompt so the model can fix exactly what was wrong.
- **Soft checks:** invoice arithmetic (qty x price, line sum = subtotal, subtotal - discount + shipping + tax = total) is printed as
  warnings, not failures, since real invoices have shipping/rounding.
- **Prompt-injection hygiene:** the document sits inside `<document>` tags and the prompt says
  it is data, not instructions.
- **Limitation:** scanned/image-only PDFs are not supported (text only).
