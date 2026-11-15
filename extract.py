#!/usr/bin/env python3
"""Extract validated JSON from unstructured invoice / resume text.

Usage:
    python extract.py --type invoice --file samples/invoice_sample.txt
    python extract.py --type resume  --file samples/resume_sample.txt --out result.json
    python extract.py --type invoice --file samples/invoice_complex.txt --provider openai
    python extract.py --type invoice --file samples/invoice_sample.txt --show-prompt   # no API key needed
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from json_extractor import JSONExtractionError, extract_json
from prompts import build_prompt, build_retry_prompt
from schemas import SCHEMAS, Invoice, invoice_warnings


class ExtractionFailed(RuntimeError):
    pass


def extract(text: str, doc_type: str, llm: Callable[[str], str],
            max_retries: int = 2, log: Callable[[str], None] = lambda m: None) -> BaseModel:
    """prompt -> LLM -> clean JSON -> Pydantic validate -> (retry with error)."""
    schema = SCHEMAS[doc_type]
    base_prompt = build_prompt(doc_type, text)
    prompt = base_prompt
    last_error = ""

    for attempt in range(1, max_retries + 2):
        raw = llm(prompt)
        try:
            data = extract_json(raw)
            return schema.model_validate(data)
        except (JSONExtractionError, ValidationError) as exc:
            last_error = str(exc)
            log(f"attempt {attempt} failed:\n{last_error}\n")
            # Feed the error back so the model can correct itself.
            prompt = build_retry_prompt(base_prompt, raw, last_error)

    raise ExtractionFailed(f"Gave up after {max_retries + 1} attempts. Last error:\n{last_error}")


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            sys.exit("PDF input needs pypdf:  pip install pypdf")
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Extract validated JSON from a document using Gemini.")
    p.add_argument("--type", required=True, choices=sorted(SCHEMAS), help="document type")
    p.add_argument("--file", required=True, type=Path, help="input .txt or .pdf file")
    p.add_argument("--out", type=Path, help="also write the JSON to this file")
    p.add_argument("--retries", type=int, default=2, help="retries after a failed validation")
    p.add_argument("--provider", choices=["gemini", "openai"],
                   default=os.getenv("LLM_PROVIDER", "gemini"), help="LLM provider (default: gemini)")
    p.add_argument("--model", help="model name (default: GEMINI_MODEL / OPENAI_MODEL from .env)")
    p.add_argument("--show-prompt", action="store_true",
                   help="print the filled prompt and exit (no API call)")
    p.add_argument("--verbose", action="store_true", help="show failed attempts")
    args = p.parse_args(argv)

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 2
    text = read_document(args.file)
    if not text.strip():
        print("The document is empty.", file=sys.stderr)
        return 2

    if args.show_prompt:
        print(build_prompt(args.type, text))
        return 0

    from llm_client import make_client
    try:
        llm = make_client(args.provider, args.model)
    except (RuntimeError, ImportError) as exc:
        print(exc, file=sys.stderr)
        return 2

    log = (lambda m: print(m, file=sys.stderr)) if args.verbose else (lambda m: None)
    try:
        result = extract(text, args.type, llm, max_retries=args.retries, log=log)
    except ExtractionFailed as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output = result.model_dump_json(indent=2)
    print(output)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")

    if isinstance(result, Invoice):
        for w in invoice_warnings(result):
            print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
