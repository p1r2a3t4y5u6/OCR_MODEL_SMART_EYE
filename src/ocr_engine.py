"""
ocr_engine.py
=============
Thin wrapper around pytesseract so the text-extraction backend can be
swapped (Tesseract today; EasyOCR/PaddleOCR could be dropped in behind
the same `.extract_text()` / `.extract_data()` interface later) without
touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class OCRResult:
    text: str
    mean_confidence: float
    word_confidences: list = field(default_factory=list)
    num_words: int = 0


class TesseractEngine:
    """OCR backend using Tesseract via pytesseract."""

    def __init__(self, lang: str = "eng", psm: int = 6, oem: int = 3, whitelist: str | None = None):
        """
        Args:
            lang: Tesseract language model to use.
            psm: Page segmentation mode. 6 = "assume a single uniform
                block of text" (good default for documents/screenshots).
                Use 7 (single line) or 8 (single word) for short code crops.
            oem: OCR engine mode. 3 = default (LSTM + legacy).
            whitelist: optional character whitelist, e.g. for short
                alphanumeric codes: "ABCDEFGHJKMNPQRSTUVWXYZ0123456789".
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.whitelist = whitelist

    def _config(self) -> str:
        config = f"--oem {self.oem} --psm {self.psm}"
        if self.whitelist:
            config += f" -c tessedit_char_whitelist={self.whitelist}"
        return config

    def extract_text(self, image: np.ndarray) -> str:
        """Return extracted text only (stripped)."""
        raw = pytesseract.image_to_string(image, lang=self.lang, config=self._config())
        return raw.strip()

    def extract_data(self, image: np.ndarray) -> OCRResult:
        """Return extracted text plus per-word confidence scores."""
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=self._config(), output_type=Output.DICT
        )
        words, confidences = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            conf = float(conf)
            if text and conf >= 0:
                words.append(text)
                confidences.append(conf)

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return OCRResult(
            text=" ".join(words),
            mean_confidence=round(mean_conf, 2),
            word_confidences=confidences,
            num_words=len(words),
        )
