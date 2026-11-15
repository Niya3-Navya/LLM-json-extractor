#!/usr/bin/env python3
"""Flask backend: exposes the extraction pipeline as an API and serves the web UI.

Run:  python server.py     then open http://localhost:5000
"""
import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from extract import ExtractionFailed, extract
from llm_client import make_client
from schemas import SCHEMAS, Invoice, invoice_warnings

BASE = Path(__file__).resolve().parent
SAMPLES = {  # whitelist: name -> file (never build paths from user input)
    "invoice": "invoice_sample.txt",
    "invoice_complex": "invoice_complex.txt",
    "resume": "resume_sample.txt",
    "resume_complex": "resume_complex.txt",
}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


def _error(message: str, status: int, **extra):
    return jsonify(success=False, error=message, **extra), status


def _document_text() -> str:
    """Text from an uploaded .txt/.pdf file (preferred) or the pasted text box."""
    upload = request.files.get("file")
    if upload and upload.filename:
        if upload.filename.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(upload.read()))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return upload.read().decode("utf-8", errors="replace")
    return request.form.get("text") or ""


@app.get("/")
def index():
    return send_from_directory(BASE / "web", "index.html")


@app.get("/api/sample/<name>")
def sample(name):
    if name not in SAMPLES:
        return _error("Unknown sample.", 404)
    return jsonify(text=(BASE / "samples" / SAMPLES[name]).read_text(encoding="utf-8"))


@app.post("/api/extract")
def api_extract():
    doc_type = request.form.get("doc_type", "")
    provider = request.form.get("provider") or os.getenv("LLM_PROVIDER", "gemini")
    if doc_type not in SCHEMAS:
        return _error(f"doc_type must be one of: {', '.join(sorted(SCHEMAS))}", 400)

    try:
        text = _document_text()
    except Exception as exc:
        return _error(f"Could not read the uploaded file: {exc}", 400)
    if not text.strip():
        return _error("No document text provided. Paste text or upload a .txt/.pdf file.", 400)

    try:
        llm = make_client(provider)
    except (RuntimeError, ImportError, ValueError) as exc:
        return _error(f"LLM is not configured: {exc}", 503)

    failed_attempts = []
    try:
        result = extract(text, doc_type, llm, max_retries=2, log=failed_attempts.append)
    except ExtractionFailed as exc:
        return _error(str(exc), 422, failed_attempts=failed_attempts)
    except Exception as exc:  # network error, quota, invalid API key...
        return _error(f"LLM call failed: {exc}", 502)

    warnings = invoice_warnings(result) if isinstance(result, Invoice) else []
    return jsonify(
        success=True,
        document_type=doc_type,
        data=result.model_dump(mode="json"),
        warnings=warnings,
        retries_used=len(failed_attempts),
        failed_attempts=failed_attempts,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
