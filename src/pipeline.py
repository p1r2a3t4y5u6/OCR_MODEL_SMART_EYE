"""
pipeline.py
===========
End-to-end OCR pipeline: image in -> preprocessing -> Tesseract -> text out.

Usage (CLI):
    python -m src.pipeline samples/input/sample_1.png
    python -m src.pipeline samples/input/sample_1.png --mode code --psm 7
    python -m src.pipeline samples/input/sample_1.png --json

Usage (library):
    from src.pipeline import OCRPipeline
    pipeline = OCRPipeline(mode="document")
    result = pipeline.run("samples/input/sample_1.png")
    print(result.text)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .ocr_engine import OCRResult, TesseractEngine
from .preprocessing import preprocess_code_crop, preprocess_document


@dataclass
class PipelineResult:
    input_path: str
    mode: str
    text: str
    mean_confidence: float
    num_words: int
    elapsed_seconds: float


class OCRPipeline:
    """Wires preprocessing + the OCR engine together."""

    def __init__(self, mode: str = "document", lang: str = "eng", psm: int | None = None):
        """
        Args:
            mode: "document" for general text (screenshots, receipts,
                signage, scanned pages) or "code" for short, noisy
                alphanumeric codes (reuses the CIG_MODEL.ipynb
                morphological preprocessing).
            lang: Tesseract language model.
            psm: override the page segmentation mode; if None, a sensible
                default is chosen per mode (6 for documents, 8 for codes).
        """
        if mode not in ("document", "code"):
            raise ValueError(f"mode must be 'document' or 'code', got {mode!r}")
        self.mode = mode

        default_psm = 6 if mode == "document" else 8
        self.engine = TesseractEngine(lang=lang, psm=psm or default_psm)

    def _preprocess(self, img):
        if self.mode == "document":
            return preprocess_document(img)
        return preprocess_code_crop(img)

    def run(self, image_path: str) -> PipelineResult:
        start = time.time()

        path = Path(image_path)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        processed = self._preprocess(img)
        result: OCRResult = self.engine.extract_data(processed)

        elapsed = time.time() - start
        return PipelineResult(
            input_path=str(path),
            mode=self.mode,
            text=result.text,
            mean_confidence=result.mean_confidence,
            num_words=result.num_words,
            elapsed_seconds=round(elapsed, 3),
        )

    def run_and_save_debug_image(self, image_path: str, out_path: str) -> PipelineResult:
        """Same as run(), but also writes the preprocessed image to disk —
        useful for the README's before/after comparison."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        processed = self._preprocess(img)
        cv2.imwrite(out_path, processed)
        return self.run(image_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small OpenCV + Tesseract OCR pipeline.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--mode", choices=["document", "code"], default="document",
        help="Preprocessing mode: 'document' (default) or 'code' (short noisy codes).",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language model (default: eng).")
    parser.add_argument("--psm", type=int, default=None, help="Override Tesseract page segmentation mode.")
    parser.add_argument("--json", action="store_true", help="Print result as JSON instead of plain text.")
    parser.add_argument(
        "--save-debug", metavar="PATH", default=None,
        help="Also save the preprocessed (pre-OCR) image to PATH.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    pipeline = OCRPipeline(mode=args.mode, lang=args.lang, psm=args.psm)

    try:
        if args.save_debug:
            result = pipeline.run_and_save_debug_image(args.image, args.save_debug)
        else:
            result = pipeline.run(args.image)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(result.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
