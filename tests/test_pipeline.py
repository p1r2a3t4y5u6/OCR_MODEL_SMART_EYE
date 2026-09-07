"""
Minimal smoke tests. Run with:  python -m pytest tests/ -v
(or just:  python tests/test_pipeline.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import OCRPipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOC = REPO_ROOT / "samples" / "input" / "sample_1.png"
SAMPLE_CODE = REPO_ROOT / "samples" / "input" / "sample_2_code.png"


def test_document_mode_extracts_expected_keywords():
    pipeline = OCRPipeline(mode="document")
    result = pipeline.run(str(SAMPLE_DOC))
    assert "OCR" in result.text
    assert "Tesseract" in result.text
    assert result.num_words > 10


def test_code_mode_runs_without_error():
    pipeline = OCRPipeline(mode="code")
    result = pipeline.run(str(SAMPLE_CODE))
    assert isinstance(result.text, str)
    assert result.mode == "code"


if __name__ == "__main__":
    test_document_mode_extracts_expected_keywords()
    test_code_mode_runs_without_error()
    print("All smoke tests passed.")
